"""
An agent with a tool that fails once, a file checkpoint, and a resume.

Requires the agentic extra:  pip install "llmswitch-sdk[agentic]"

    OPENAI_API_KEY=sk-... python examples/agentic_agent.py

The lookup tool fails on its first call. With ``interrupt_on_tool_error=True`` the run pauses, its state
is written to a checkpoint file, and the example resumes it from that file, as a new process would.
"""
import os
import tempfile

from llmswitch.agentic import Agent, AgentConfig, ChatLLMSwitch, FileCheckpointStore

_calls = {"lookup": 0}


def lookup_population(city: str) -> str:
    """Look up the population of a city."""
    _calls["lookup"] += 1
    if _calls["lookup"] == 1:
        raise RuntimeError("the population service timed out, try again")
    return f"{city} has about 2.1 million people."


def main() -> None:
    llm = ChatLLMSwitch(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    checkpoint_dir = tempfile.mkdtemp(prefix="llmswitch-checkpoints-")
    agent = Agent(
        model=llm,
        tools=[lookup_population],
        config=AgentConfig(max_iterations=6, interrupt_on_tool_error=True, reflection_enabled=False),
        checkpoint_store=FileCheckpointStore(checkpoint_dir),
    )
    try:
        result = agent.invoke("What is the population of Lyon? Answer in one sentence.")
        print("first run:", result["status"], result.get("reason"))

        if result["status"] == "interrupted":
            # A new process could do this too, because the checkpoint is a file in checkpoint_dir.
            result = agent.resume(result["checkpoint_id"])
            print("resumed:  ", result["status"])

        print("output:   ", result.get("output"))
        for step in result["steps"]:
            print(f"  step {step.step}: model={step.model_used} tools={[r.tool_name for r in step.tool_results]}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
