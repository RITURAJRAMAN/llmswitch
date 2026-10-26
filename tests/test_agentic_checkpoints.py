"""Checkpoints: the stores, the agent's interrupt and resume paths, and what a resume must never repeat."""
import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from test_agentic_agent import Scripted, call, verdict  # noqa: E402

from llmswitch.agentic import Agent, AgentConfig, FileCheckpointStore, idempotent  # noqa: E402
from llmswitch.agentic.checkpoints import InMemoryCheckpointStore, validate_checkpoint_id  # noqa: E402


def find(store, phase):
    """The id of the newest checkpoint in ``phase`` (in-memory store only)."""
    matches = [cid for cid, state in store._data.items() if state["phase"] == phase]
    assert matches, f"no checkpoint in phase {phase}"
    return matches[-1]


# --- stores -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "../escape", "a/b", "a\\b", ".hidden", "x" * 129, "has space", None])
def test_unsafe_checkpoint_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_checkpoint_id(bad)


def test_the_in_memory_store_keeps_copies():
    store = InMemoryCheckpointStore()
    state = {"phase": "tools_done", "messages": [1]}
    store.save("c1", state)
    state["messages"].append(2)
    assert store.load("c1") == {"phase": "tools_done", "messages": [1]}
    store.delete("c1")
    assert store.load("c1") is None


def test_the_file_store_round_trips_and_leaves_no_temporary_files(tmp_path):
    store = FileCheckpointStore(tmp_path / "nested")
    store.save("run-1", {"phase": "tools_done", "step": 2})
    assert store.load("run-1") == {"phase": "tools_done", "step": 2}
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["run-1.json"]
    store.delete("run-1")
    assert store.load("run-1") is None


def test_a_tampered_file_is_refused(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save("c2", {"phase": "tools_done", "step": 1})
    path = tmp_path / "c2.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["state"]["step"] = 99
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        store.load("c2")


def test_a_file_with_another_schema_version_is_refused(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save("c3", {"phase": "tools_done"})
    path = tmp_path / "c3.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["schema_version"] = 99
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="schema"):
        store.load("c3")


def test_the_file_store_rejects_unsafe_ids(tmp_path):
    store = FileCheckpointStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("../outside", {"phase": "tools_done"})
    assert not any(tmp_path.parent.glob("outside*"))


def test_a_missing_checkpoint_is_reported_on_resume():
    agent = Agent(Scripted(), config=AgentConfig())
    with pytest.raises(ValueError, match="Checkpoint not found"):
        agent.resume("never-saved")


# --- interrupts and resume ----------------------------------------------------------------------------

def failing(name):
    @tool
    def failer(x: int) -> str:
        """Fails."""
        raise RuntimeError("temporary")
    failer.name = name
    return failer


def test_a_tool_error_interrupts_with_a_checkpoint_that_resumes_to_an_answer():
    model = Scripted(script=[call("fail_once", {"x": 1}, "i1")])
    agent = Agent(model, tools=[failing("fail_once")], config=AgentConfig(interrupt_on_tool_error=True),
                  checkpoint_store=InMemoryCheckpointStore())
    result = agent.invoke("x")
    assert result["status"] == "interrupted" and result["reason"] == "tool_error"
    assert agent.checkpoint_store.load(result["checkpoint_id"])["phase"] == "tools_done"
    model.script.extend([AIMessage(content="Handled it."), verdict(True)])
    resumed = agent.resume(result["checkpoint_id"])
    assert resumed["status"] == "completed" and resumed["output"] == "Handled it."


def test_a_crash_during_a_tool_is_recovered_without_rerunning_a_finished_tool():
    runs = {"done": 0, "crash": 0}

    @tool
    def finished(x: int) -> str:
        """Finishes."""
        runs["done"] += 1
        return "finished"

    @tool
    def crashes(x: int) -> str:
        """Crashes the process once."""
        runs["crash"] += 1
        if runs["crash"] == 1:
            raise SystemExit("process killed")
        return "recovered"

    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="", tool_calls=[
        {"name": "finished", "args": {"x": 1}, "id": "f1", "type": "tool_call"},
        {"name": "crashes", "args": {"x": 2}, "id": "c1", "type": "tool_call"},
    ])])
    agent = Agent(model, tools=[finished, crashes], config=AgentConfig(parallel_tools=False), checkpoint_store=store)
    with pytest.raises(SystemExit):
        agent.invoke("x")
    checkpoint = find(store, "awaiting_tools")
    progress = store.load(checkpoint)["progress"]
    assert [entry["status"] for entry in progress] == ["completed", "started"]

    # The crashed tool is not idempotent, so the run stops rather than running it again.
    stopped = agent.resume(checkpoint)
    assert stopped["status"] == "interrupted" and stopped["reason"] == "uncertain_tool_execution"
    assert runs == {"done": 1, "crash": 1}


def test_an_idempotent_tool_that_was_interrupted_is_run_again_and_the_finished_one_is_not():
    runs = {"done": 0, "crash": 0}

    @tool
    def finished(x: int) -> str:
        """Finishes."""
        runs["done"] += 1
        return "finished"

    @idempotent
    @tool
    def crashes(x: int) -> str:
        """Crashes the process once."""
        runs["crash"] += 1
        if runs["crash"] == 1:
            raise SystemExit("process killed")
        return "recovered"

    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="", tool_calls=[
        {"name": "finished", "args": {"x": 1}, "id": "f2", "type": "tool_call"},
        {"name": "crashes", "args": {"x": 2}, "id": "c2", "type": "tool_call"},
    ]), AIMessage(content="All done."), verdict(True)])
    agent = Agent(model, tools=[finished, crashes], config=AgentConfig(parallel_tools=False), checkpoint_store=store)
    with pytest.raises(SystemExit):
        agent.invoke("x")
    result = agent.resume(find(store, "awaiting_tools"))
    assert result["status"] == "completed" and result["output"] == "All done."
    assert runs == {"done": 1, "crash": 2}
    contents = [m.content for m in result["messages"] if getattr(m, "tool_call_id", None)]
    assert contents == ["finished", "recovered"]


def test_a_reflection_interrupted_by_a_crash_resumes_to_a_verdict():
    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="Paris.")])
    agent = Agent(model, config=AgentConfig(), checkpoint_store=store)
    with pytest.raises(IndexError):  # the verdict call finds no scripted reply: the run stops here
        agent.invoke("Capital of France?")
    checkpoint = find(store, "awaiting_reflection")
    model.script.append(verdict(True))
    result = agent.resume(checkpoint)
    assert result["status"] == "completed" and result["output"] == "Paris."
    assert len(result["steps"]) == 1


def test_feedback_on_a_pending_answer_replaces_it_and_redoes_the_step():
    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="Lyon."), verdict(False, "wrong")])
    agent = Agent(model, config=AgentConfig(reflection_max_retries=0), checkpoint_store=store)
    result = agent.invoke("Capital of France?")
    assert result["status"] == "completed_with_reflection_limit"
    # Interrupt a fresh run at reflection, then send feedback instead of a verdict.
    model2 = Scripted(script=[AIMessage(content="Lyon.")])
    agent2 = Agent(model2, config=AgentConfig(), checkpoint_store=store)
    with pytest.raises(IndexError):
        agent2.invoke("Capital of France?")
    checkpoint = find(store, "awaiting_reflection")
    model2.script.extend([AIMessage(content="Paris."), verdict(True)])
    result2 = agent2.resume(checkpoint, user_feedback="Think about the capital again.")
    assert result2["status"] == "completed" and result2["output"] == "Paris."
    assert "Lyon." not in [m.content for m in result2["messages"]]


def test_feedback_cannot_change_tools_that_already_started():
    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="", tool_calls=[
        {"name": "started_tool", "args": {"x": 1}, "id": "s1", "type": "tool_call"}])])

    @tool
    def started_tool(x: int) -> str:
        """Crashes."""
        raise SystemExit("killed")

    agent = Agent(model, tools=[started_tool], config=AgentConfig(), checkpoint_store=store)
    with pytest.raises(SystemExit):
        agent.invoke("x")
    with pytest.raises(ValueError, match="may have started"):
        agent.resume(find(store, "awaiting_tools"), user_feedback="Change the tool call.")


def test_feedback_on_a_finished_step_is_added_to_the_conversation():
    store = InMemoryCheckpointStore()
    model = Scripted(script=[call("ok_tool", {"x": 1}, "o1"), AIMessage(content="ok"), verdict(True)])

    @tool
    def ok_tool(x: int) -> str:
        """Fine."""
        return "fine"

    agent = Agent(model, tools=[ok_tool], config=AgentConfig(interrupt_on_tool_error=False), checkpoint_store=store)
    result = agent.invoke("x")
    assert result["status"] == "completed"
    # Resume a tools_done checkpoint written by a run that was interrupted for another reason.
    done = find(store, "tools_done")
    model.script.extend([AIMessage(content="Noted."), verdict(True)])
    resumed = agent.resume(done, user_feedback="Also mention the time.")
    assert resumed["status"] == "completed"
    assert "Also mention the time." in [m.content for m in resumed["messages"]]


async def test_an_async_resume_continues_from_an_interrupt():
    model = Scripted(script=[call("async_fail", {"x": 1}, "af")])
    agent = Agent(model, tools=[failing("async_fail")], config=AgentConfig(interrupt_on_tool_error=True),
                  checkpoint_store=InMemoryCheckpointStore())
    result = await agent.ainvoke("x")
    assert result["status"] == "interrupted"
    model.script.extend([AIMessage(content="Async done."), verdict(True)])
    resumed = await agent.aresume(result["checkpoint_id"])
    assert resumed["status"] == "completed" and resumed["output"] == "Async done."


def test_a_checkpoint_from_another_schema_version_is_refused():
    store = InMemoryCheckpointStore()
    store.save("old", {"schema_version": 0, "phase": "tools_done"})
    with pytest.raises(ValueError, match="schema version"):
        Agent(Scripted(), checkpoint_store=store).resume("old")


def test_with_reflection_off_the_final_answer_is_recorded_as_a_step():
    model = Scripted(script=[AIMessage(content="Paris.")])
    result = Agent(model, config=AgentConfig(reflection_enabled=False), checkpoint_store=InMemoryCheckpointStore()).invoke("x")
    assert result["status"] == "completed" and result["output"] == "Paris."
    assert len(result["steps"]) == 1 and result["steps"][0].assistant_content == "Paris."
