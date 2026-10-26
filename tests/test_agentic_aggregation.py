"""Aggregation: majority vote, weighted score, JSON merge, and running specialists then aggregating."""
import pytest

from llmswitch.agentic import (
    AggregatedResult,
    JsonMergeAggregator,
    MajorityVoteAggregator,
    SpecialistContract,
    WeightedScoreAggregator,
    aaggregate_specialists,
    aggregate_specialists,
)


class FakeAgent:
    """Stands in for an agent: returns a fixed status and output without any model call."""

    def __init__(self, output, status="completed"):
        self.output = output
        self.status = status

    def invoke(self, task, max_runtime_seconds=None):
        return {"status": self.status, "output": self.output, "reason": None, "steps": [], "checkpoint_id": None}

    async def ainvoke(self, task, max_runtime_seconds=None):
        return self.invoke(task, max_runtime_seconds)


def row(output, status="completed", **extra):
    return {"specialist": extra.pop("specialist", "s"), "status": status, "output": output, **extra}


def test_majority_vote_picks_the_most_common_output_and_reports_the_share():
    result = MajorityVoteAggregator().aggregate([row("a"), row("b"), row("a")])

    assert result.consensus == "a"
    assert result.strategy == "majority_vote"
    assert result.confidence == pytest.approx(2 / 3)
    assert result.metadata["votes"] == {'"a"': 2, '"b"': 1}


def test_a_tie_goes_to_the_answer_that_came_first():
    assert MajorityVoteAggregator().aggregate([row("b"), row("a")]).consensus == "b"
    assert MajorityVoteAggregator().aggregate([row("a"), row("b")]).consensus == "a"


def test_a_json_string_and_the_object_it_holds_vote_together_and_key_order_does_not_matter():
    rows = [row('{"x": 1, "y": 2}'), row({"y": 2, "x": 1}), row("other")]

    result = MajorityVoteAggregator().aggregate(rows)

    assert result.consensus == {"x": 1, "y": 2}
    assert result.confidence == pytest.approx(2 / 3)


def test_a_json_object_inside_a_code_fence_is_read_as_the_object():
    rows = [row('```json\n{"x": 1}\n```'), row('{"x": 1}'), row("```not json```")]

    result = MajorityVoteAggregator().aggregate(rows)

    assert result.consensus == {"x": 1}
    assert result.confidence == pytest.approx(2 / 3)
    assert result.metadata["votes"] == {'{"x": 1}': 2, '"```not json```"': 1}


def test_failed_rows_are_left_out_unless_asked_for():
    rows = [row("a"), row("b", status="timeout"), row("b", status="stopped"), row("b", status="error")]

    assert MajorityVoteAggregator().aggregate(rows).consensus == "a"
    assert MajorityVoteAggregator(include_failed=True).aggregate(rows).consensus == "b"


def test_completed_with_reflection_limit_is_still_an_answer():
    rows = [row("a", status="completed_with_reflection_limit")]

    assert MajorityVoteAggregator().aggregate(rows).consensus == "a"


def test_outputs_that_are_none_are_not_votes_but_count_against_confidence():
    result = MajorityVoteAggregator().aggregate([row("a"), row(None)])

    assert result.consensus == "a"
    assert result.confidence == pytest.approx(0.5)


def test_no_answer_gives_no_consensus_and_zero_confidence():
    for aggregator in (MajorityVoteAggregator(), WeightedScoreAggregator(), JsonMergeAggregator()):
        result = aggregator.aggregate([row(None), row(None, status="timeout")])

        assert result.consensus is None
        assert result.confidence == 0.0
        assert result.inputs == [row(None)]


def test_an_empty_list_of_rows_is_handled_by_every_aggregator():
    for aggregator in (MajorityVoteAggregator(), WeightedScoreAggregator(), JsonMergeAggregator()):
        result = aggregator.aggregate([])

        assert result == AggregatedResult(consensus=None, inputs=[], strategy=result.strategy, confidence=0.0)


def test_weights_can_outvote_a_larger_number_of_rows():
    rows = [row("a", weight=1.0), row("a", weight=1.0), row("b", weight=3.0)]

    result = WeightedScoreAggregator().aggregate(rows)

    assert result.consensus == "b"
    assert result.confidence == pytest.approx(0.6)
    assert result.metadata["weights"] == {'"a"': 2.0, '"b"': 3.0}


def test_a_row_without_a_weight_counts_as_one():
    result = WeightedScoreAggregator().aggregate([row("a"), row("b", weight=0.5)])

    assert result.consensus == "a"
    assert result.confidence == pytest.approx(1 / 1.5)


def test_invalid_weights_are_rejected():
    for bad in (-1.0, float("nan"), float("inf"), "heavy"):
        with pytest.raises(ValueError, match="weights"):
            WeightedScoreAggregator().aggregate([row("a", weight=bad)])


def test_contract_weights_reach_the_weighted_score_through_invoke_specialists():
    from llmswitch.agentic import attach_contract_weights, invoke_specialists

    specialists = {
        "senior": FakeAgent("plan A"),
        "junior_1": FakeAgent("plan B"),
        "junior_2": FakeAgent("plan B"),
    }
    contracts = {
        "senior": SpecialistContract(capabilities={"planning"}, weight=3.0),
        "junior_1": SpecialistContract(capabilities={"planning"}, weight=1.0),
        "junior_2": SpecialistContract(capabilities={"planning"}, weight=1.0),
    }
    rows = invoke_specialists(task="plan it", specialists=specialists, contracts=contracts)

    assert [item["weight"] for item in rows] == [3.0, 1.0, 1.0]
    assert WeightedScoreAggregator().aggregate(rows).consensus == "plan A"
    assert MajorityVoteAggregator().aggregate(rows).consensus == "plan B"
    assert attach_contract_weights(rows, contracts) == rows


def test_json_merge_combines_disjoint_keys_and_last_writer_wins_on_conflicts():
    rows = [row({"name": "x", "size": 1}), row({"size": 2, "color": "red"})]

    result = JsonMergeAggregator().aggregate(rows)

    assert result.strategy == "json_merge"
    assert result.consensus == {"name": "x", "size": 2, "color": "red"}
    assert result.confidence == 1.0
    assert result.metadata["conflicts"] == {"size": [1, 2]}
    assert result.metadata["conflict_policy"] == "last_writer_wins"


def test_first_writer_wins_keeps_the_earlier_value():
    rows = [row({"size": 1}), row({"size": 2})]

    result = JsonMergeAggregator(conflict_policy="first_writer_wins").aggregate(rows)

    assert result.consensus == {"size": 1}


def test_the_error_policy_raises_on_a_conflict_and_is_quiet_without_one():
    with pytest.raises(ValueError, match="'size'"):
        JsonMergeAggregator(conflict_policy="error").aggregate([row({"size": 1}), row({"size": 2})])

    result = JsonMergeAggregator(conflict_policy="error").aggregate([row({"size": 1}), row({"size": 1})])

    assert result.consensus == {"size": 1}


def test_json_merge_counts_only_rows_with_an_object_and_parses_json_strings():
    rows = [row('{"a": 1}'), row("plain text"), row(None)]

    result = JsonMergeAggregator().aggregate(rows)

    assert result.consensus == {"a": 1}
    assert result.confidence == pytest.approx(1 / 3)


def test_json_merge_falls_back_to_a_majority_vote_without_objects():
    rows = [row("yes"), row("yes"), row("no")]

    result = JsonMergeAggregator().aggregate(rows)

    assert result.strategy == "json_merge_fallback_majority"
    assert result.consensus == "yes"
    assert result.metadata["fallback"] is True
    assert result.metadata["votes"] == {'"yes"': 2, '"no"': 1}


def test_an_unknown_conflict_policy_is_rejected():
    with pytest.raises(ValueError, match="conflict_policy"):
        JsonMergeAggregator(conflict_policy="average")


def test_aggregate_specialists_runs_the_agents_and_merges_their_objects():
    specialists = {
        "pricing": FakeAgent({"price": 10}),
        "delivery": FakeAgent({"days": 3}),
    }

    result = aggregate_specialists(task="quote it", specialists=specialists)

    assert result.consensus == {"price": 10, "days": 3}
    assert [item["specialist"] for item in result.inputs] == ["pricing", "delivery"]


def test_aggregate_specialists_can_weigh_the_answers_by_contract():
    specialists = {"senior": FakeAgent("yes"), "junior": FakeAgent("no")}
    contracts = {
        "senior": SpecialistContract(capabilities={"review"}, weight=2.0),
        "junior": SpecialistContract(capabilities={"review"}, weight=1.0),
    }

    result = aggregate_specialists(
        task="approve?",
        specialists=specialists,
        contracts=contracts,
        aggregator=WeightedScoreAggregator(),
    )

    assert result.consensus == "yes"
    assert result.confidence == pytest.approx(2 / 3)


def test_aggregate_specialists_skips_specialists_without_the_required_capability():
    specialists = {"senior": FakeAgent("yes"), "junior": FakeAgent("no")}
    contracts = {"senior": SpecialistContract(capabilities={"review"}), "junior": SpecialistContract()}

    result = aggregate_specialists(
        task="approve?",
        specialists=specialists,
        contracts=contracts,
        required_capabilities={"review"},
        aggregator=MajorityVoteAggregator(),
    )

    assert result.consensus == "yes"
    assert [item["specialist"] for item in result.inputs] == ["senior"]


async def test_aaggregate_specialists_merges_the_async_rows():
    specialists = {"a": FakeAgent({"x": 1}), "b": FakeAgent({"y": 2}, status="error")}

    result = await aaggregate_specialists(task="go", specialists=specialists)

    assert result.consensus == {"x": 1}
    assert result.confidence == 1.0
    assert [item["status"] for item in result.inputs] == ["completed"]
