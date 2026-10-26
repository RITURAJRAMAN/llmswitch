"""Specialists: contracts, shared state, routing, running several specialists, and agents used as tools."""
import asyncio
import json
import time

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from test_agentic_agent import Scripted, call, verdict  # noqa: E402

from llmswitch.agentic import (  # noqa: E402
    Agent,
    AgentConfig,
    CostAwareRoutingPolicy,
    EscalationRoutingPolicy,
    InMemoryCheckpointStore,
    RoundRobinRoutingPolicy,
    SharedState,
    SpecialistContract,
    ainvoke_specialists,
    attach_contract_weights,
    filter_specialists_by_capabilities,
    invoke_specialists,
    make_specialist_tools,
    make_subagent_tool,
)


class Named(Scripted):
    """A scripted model with its own name, so each step's ``model_used`` says which model answered."""

    name: str = "model"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.name}


class Slow(Scripted):
    """A scripted model that takes ``delay`` seconds to answer."""

    delay: float = 0.0

    def _next(self, messages):
        time.sleep(self.delay)
        return super()._next(messages)


class AsyncSlow(Scripted):
    delay: float = 0.0

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


class Broken(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "broken"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("model unavailable")


class Recorder:
    def __init__(self):
        self.decisions = []

    def on_routing_decision(self, step, context):
        self.decisions.append((step, context["policy"], context["reason"], context["model"]))


@tool
def fail_tool(x: int) -> str:
    """Always fails."""
    raise RuntimeError("upstream down")


@tool
def ok_tool(x: int) -> str:
    """Works."""
    return "fine"


def named(name, script):
    return Named(script=list(script), name=name)


# --- contracts and shared state -----------------------------------------------------------------------

def test_a_contract_weight_must_be_finite_and_not_negative():
    with pytest.raises(ValueError):
        SpecialistContract(weight=-1)
    with pytest.raises(ValueError):
        SpecialistContract(weight=float("inf"))
    assert SpecialistContract(capabilities={"math"}, weight=2.0).weight == 2.0


def test_specialists_are_filtered_by_required_capabilities():
    specialists = {"math": "m", "poet": "p", "unlisted": "u"}
    contracts = {"math": SpecialistContract({"math", "numbers"}), "poet": SpecialistContract({"poetry"})}
    assert filter_specialists_by_capabilities(specialists, contracts, {"math"}) == {"math": "m"}
    assert filter_specialists_by_capabilities(specialists, contracts, None) == specialists
    assert filter_specialists_by_capabilities(specialists, None, {"math"}) == {}


def test_weights_and_capabilities_are_attached_to_rows():
    rows = attach_contract_weights([{"specialist": "math"}, {"specialist": "other"}],
                                   {"math": SpecialistContract({"math"}, weight=3.0)})
    assert rows[0]["weight"] == 3.0 and rows[0]["capabilities"] == ["math"]
    assert "weight" not in rows[1]


def test_shared_state_reads_writes_and_snapshots():
    state = SharedState({"a": 1})
    state.set("b", 2)
    state.update({"c": 3})
    assert (state.get("a"), state.get("missing", "dflt")) == (1, "dflt")
    snapshot = state.snapshot()
    snapshot["a"] = 99
    assert state.get("a") == 1


def test_the_prompt_projection_is_bounded_and_skips_what_does_not_fit():
    state = SharedState({"small": "x", "huge": "y" * 5000, "later": "z"})
    projected = state.project_for_prompt(max_bytes=200)
    assert "small" in projected and "huge" not in projected and "later" in projected
    assert state.project_for_prompt(allowed_keys={"huge"}, max_bytes=200) == {}
    with pytest.raises(ValueError):
        state.project_for_prompt(max_bytes=0)


# --- routing policies ---------------------------------------------------------------------------------

def test_round_robin_alternates_and_keeps_its_index_in_the_state():
    policy = RoundRobinRoutingPolicy()
    state = {}
    picks = [policy.select_llm(messages=[], step=i, had_tool_error=False, primary_llm="p", fallback_llm="f",
                               state=state).llm for i in range(1, 5)]
    assert picks == ["p", "f", "p", "f"]


def test_escalation_uses_the_fallback_only_after_a_tool_error():
    policy = EscalationRoutingPolicy()
    assert policy.select_llm(messages=[], step=1, had_tool_error=True, primary_llm="p", fallback_llm="f",
                             state={}).llm == "f"
    assert policy.select_llm(messages=[], step=1, had_tool_error=False, primary_llm="p", fallback_llm="f",
                             state={}).llm == "p"


def test_cost_aware_switches_at_the_chosen_step_or_after_an_error():
    policy = CostAwareRoutingPolicy(fallback_after_step=3)
    assert policy.select_llm(messages=[], step=2, had_tool_error=False, primary_llm="p", fallback_llm="f",
                             state={}).llm == "p"
    assert policy.select_llm(messages=[], step=3, had_tool_error=False, primary_llm="p", fallback_llm="f",
                             state={}).reason == "late_step_switch"
    with pytest.raises(ValueError):
        CostAwareRoutingPolicy(fallback_after_step=0)


# --- routing inside the agent loop --------------------------------------------------------------------

def test_escalation_moves_the_next_step_to_the_fallback_after_a_tool_error():
    primary = named("primary", [call("fail_tool", {"x": 1}, "e1")])
    fallback = named("fallback", [AIMessage(content="Recovered."), verdict(True)])
    agent = Agent(primary, tools=[fail_tool], fallback_model=fallback, routing_policy=EscalationRoutingPolicy())
    result = agent.invoke("x")
    assert result["status"] == "completed" and result["output"] == "Recovered."
    assert [s.model_used for s in result["steps"]] == ["primary", "fallback"]
    assert len(primary.seen) == 1 and len(fallback.seen) == 2  # the answer, then its reflection


def test_round_robin_records_which_model_answered_each_step():
    primary = named("p1", [call("ok_tool", {"x": 1}, "r1"), AIMessage(content="never"), verdict(True)])
    fallback = named("f1", [AIMessage(content="From the fallback."), verdict(True)])
    agent = Agent(primary, tools=[ok_tool], fallback_model=fallback, routing_policy=RoundRobinRoutingPolicy())
    result = agent.invoke("x")
    assert [s.model_used for s in result["steps"]] == ["p1", "f1"]
    assert result["output"] == "From the fallback."


def test_a_custom_router_is_called_with_the_step_number():
    primary = named("pa", [call("ok_tool", {"x": 1}, "rt1")])
    other = named("pb", [AIMessage(content="Routed."), verdict(True)])
    seen_steps = []

    def router(messages, step):
        seen_steps.append(step)
        return other if step == 2 else primary

    result = Agent(primary, tools=[ok_tool], router=router).invoke("x")
    assert seen_steps == [1, 2]
    assert result["output"] == "Routed." and result["steps"][1].model_used == "pb"


def test_routing_decisions_are_reported_to_the_event_handler():
    recorder = Recorder()
    primary = named("pc", [call("ok_tool", {"x": 1}, "ev")])
    fallback = named("fc", [AIMessage(content="ok"), verdict(True)])
    Agent(primary, tools=[ok_tool], fallback_model=fallback, routing_policy=CostAwareRoutingPolicy(2),
          event_handler=recorder).invoke("x")
    assert recorder.decisions[0] == (1, "cost_aware", "primary_default", "pc")
    assert recorder.decisions[1] == (2, "cost_aware", "late_step_switch", "fc")


def test_the_routing_rotation_survives_a_resume():
    store = InMemoryCheckpointStore()
    primary = named("rp", [call("fail_tool", {"x": 1}, "rr1")])
    fallback = named("rf", [AIMessage(content="Resumed on the fallback."), verdict(True)])
    agent = Agent(primary, tools=[fail_tool], fallback_model=fallback, routing_policy=RoundRobinRoutingPolicy(),
                  config=AgentConfig(interrupt_on_tool_error=True), checkpoint_store=store)
    interrupted = agent.invoke("x")
    assert interrupted["status"] == "interrupted"
    resumed = agent.resume(interrupted["checkpoint_id"])
    # Step 1 used index 0 (primary). The restored rotation picks index 1, the fallback, for the next step.
    assert resumed["status"] == "completed" and resumed["output"] == "Resumed on the fallback."


# --- running specialists ------------------------------------------------------------------------------

def specialist_agents():
    return {
        "math": Agent(named("math", [AIMessage(content="4")]), config=AgentConfig(reflection_enabled=False)),
        "poetry": Agent(named("poetry", [AIMessage(content="roses")]), config=AgentConfig(reflection_enabled=False)),
        "history": Agent(named("history", [AIMessage(content="1066")]), config=AgentConfig(reflection_enabled=False)),
    }


def test_specialists_run_in_parallel_and_rows_keep_the_given_order():
    rows = invoke_specialists(task="Answer briefly.", specialists=specialist_agents(), parallel=True)
    assert [row["specialist"] for row in rows] == ["math", "poetry", "history"]
    assert [row["output"] for row in rows] == ["4", "roses", "1066"]
    assert all(row["status"] == "completed" for row in rows)


def test_specialists_can_run_one_after_another():
    rows = invoke_specialists(task="x", specialists=specialist_agents(), parallel=False)
    assert [row["output"] for row in rows] == ["4", "roses", "1066"]


def test_only_capable_specialists_are_selected_and_their_weights_are_reported():
    contracts = {"math": SpecialistContract({"math"}, weight=2.0), "poetry": SpecialistContract({"poetry"})}
    rows = invoke_specialists(task="x", specialists=specialist_agents(), contracts=contracts,
                              required_capabilities={"math"})
    assert [row["specialist"] for row in rows] == ["math"]
    assert rows[0]["weight"] == 2.0


def test_no_specialist_matching_the_capabilities_gives_no_rows():
    assert invoke_specialists(task="x", specialists=specialist_agents(), contracts={},
                              required_capabilities={"astronomy"}) == []


def test_a_specialist_that_raises_is_an_error_row_and_the_others_still_run():
    agents = specialist_agents()
    agents["broken"] = Agent(Broken(), config=AgentConfig(reflection_enabled=False))
    rows = {row["specialist"]: row for row in invoke_specialists(task="x", specialists=agents)}
    assert rows["broken"]["status"] == "error" and "model unavailable" in rows["broken"]["reason"]
    assert rows["math"]["output"] == "4"


def test_shared_state_gets_each_output_and_feeds_the_next_prompt():
    state = SharedState({"topic": "history"})
    listener = named("listener", [AIMessage(content="Noted.")])
    agents = {"history": Agent(named("hist", [AIMessage(content="1066")]), config=AgentConfig(reflection_enabled=False)),
              "listen": Agent(listener, config=AgentConfig(reflection_enabled=False))}
    invoke_specialists(task="Find the date.", specialists={"history": agents["history"]}, shared_state=state)
    assert state.get("specialist_outputs.history") == "1066"
    invoke_specialists(task="Summarise.", specialists={"listen": agents["listen"]}, shared_state=state,
                       shared_state_keys={"topic", "specialist_outputs.history"})
    prompt = listener.seen[0][0].content
    assert "SharedStateContext(JSON)" in prompt and "1066" in prompt and "history" in prompt


def test_a_specialist_past_the_shared_time_limit_is_reported_as_a_timeout():
    slow = Agent(Slow(script=[AIMessage(content="late")], delay=1.0), config=AgentConfig(reflection_enabled=False))
    fast = Agent(named("quick", [AIMessage(content="early")]), config=AgentConfig(reflection_enabled=False))
    started = time.monotonic()
    rows = {r["specialist"]: r for r in invoke_specialists(
        task="x", specialists={"slow": slow, "fast": fast}, max_runtime_seconds=0.3)}
    assert time.monotonic() - started < 0.9
    assert rows["slow"]["status"] == "timeout" and rows["slow"]["output"] is None
    assert rows["fast"]["output"] == "early"


def test_a_negative_shared_time_limit_is_rejected():
    with pytest.raises(ValueError):
        invoke_specialists(task="x", specialists=specialist_agents(), max_runtime_seconds=-1)


async def test_async_specialists_run_and_slow_ones_are_cancelled_at_the_limit():
    slow = Agent(AsyncSlow(script=[AIMessage(content="late")], delay=5.0), config=AgentConfig(reflection_enabled=False))
    fast = Agent(named("qa", [AIMessage(content="ok")]), config=AgentConfig(reflection_enabled=False))
    started = time.monotonic()
    rows = {r["specialist"]: r for r in await ainvoke_specialists(
        task="x", specialists={"slow": slow, "fast": fast}, max_runtime_seconds=0.3)}
    assert time.monotonic() - started < 2.0
    assert rows["slow"]["status"] == "timeout" and rows["fast"]["output"] == "ok"


async def test_async_specialists_keep_the_order_without_a_limit():
    rows = await ainvoke_specialists(task="x", specialists=specialist_agents())
    assert [row["output"] for row in rows] == ["4", "roses", "1066"]


# --- an agent used as a tool -----------------------------------------------------------------------------

def test_a_subagent_tool_returns_a_json_summary():
    specialist = Agent(named("sub", [AIMessage(content="Paris.")]), config=AgentConfig(reflection_enabled=False))
    tool_fn = make_subagent_tool(name="ask_geography", description="Answers geography questions.", agent=specialist)
    assert tool_fn.__name__ == "ask_geography" and tool_fn.__doc__ == "Answers geography questions."
    summary = json.loads(tool_fn("Capital of France?"))
    assert summary == {"status": "completed", "output": "Paris.", "reason": None, "steps": 1}


def test_a_parent_agent_can_delegate_to_a_specialist_through_a_tool():
    specialist = Agent(named("child", [AIMessage(content="Paris.")]), config=AgentConfig(reflection_enabled=False))
    delegate = make_subagent_tool(name="ask_geography", description="Geography.", agent=specialist)
    parent = Scripted(script=[call("ask_geography", {"task": "Capital of France?"}, "d1"),
                              AIMessage(content="The capital is Paris."), verdict(True)])
    result = Agent(parent, tools=[delegate]).invoke("What is the capital of France?")
    assert result["status"] == "completed" and result["output"] == "The capital is Paris."
    assert json.loads(result["steps"][0].tool_results[0].content)["output"] == "Paris."


def test_make_specialist_tools_builds_one_tool_per_entry():
    agents = specialist_agents()
    specs = {name: (f"Handles {name}.", agent) for name, agent in agents.items()}
    tools = make_specialist_tools(specs)
    assert sorted(t.__name__ for t in tools) == ["history", "math", "poetry"]


# --- configuration -----------------------------------------------------------------------------------

def test_a_specialist_limit_of_zero_times_out_every_specialist_without_running_it():
    model = Scripted(script=[AIMessage(content="never")])
    rows = invoke_specialists(task="x", specialists={"only": Agent(model)}, max_runtime_seconds=0)
    assert rows[0]["status"] == "timeout"
    assert model.seen == []
