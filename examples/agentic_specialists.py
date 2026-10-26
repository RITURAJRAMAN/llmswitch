"""
Several specialist agents answer the same task, and their answers are combined by weight.

Requires the agentic extra:  pip install "llmswitch-sdk[agentic]"

    OPENAI_API_KEY=sk-... python examples/agentic_specialists.py

Each specialist has a contract: its capabilities decide which specialists are asked, and its weight decides
how much its answer counts. The rows show each specialist's status; the weighted score picks the answer
with the largest total weight.
"""
import os

from llmswitch.agentic import (
    Agent,
    AgentConfig,
    ChatLLMSwitch,
    SpecialistContract,
    WeightedScoreAggregator,
    aggregate_specialists,
    invoke_specialists,
)


def make_agent(user_id: str) -> Agent:
    llm = ChatLLMSwitch(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini",
                        user_id=user_id)
    return Agent(model=llm, config=AgentConfig(reflection_enabled=False, max_iterations=2))


def main() -> None:
    specialists = {
        "geography": make_agent("example-geography"),
        "trivia": make_agent("example-trivia"),
        "reviewer": make_agent("example-reviewer"),
    }
    contracts = {
        "geography": SpecialistContract(capabilities={"facts", "geography"}, weight=2.0),
        "trivia": SpecialistContract(capabilities={"facts"}, weight=1.0),
        "reviewer": SpecialistContract(capabilities={"review"}, weight=1.0),
    }
    task = "What is the capital of Australia? Answer with the city name only."

    try:
        # Only specialists with the "facts" capability are asked.
        rows = invoke_specialists(task=task, specialists=specialists, contracts=contracts,
                                  required_capabilities={"facts"}, max_runtime_seconds=60)
        for row in rows:
            print(f"{row['specialist']:>10}: status={row['status']} weight={row['weight']} output={row['output']!r}")

        result = aggregate_specialists(task=task, specialists=specialists, contracts=contracts,
                                       required_capabilities={"facts"}, aggregator=WeightedScoreAggregator(),
                                       max_runtime_seconds=60)
        print(f"consensus: {result.consensus!r} (confidence {result.confidence:.2f}, "
              f"weights {result.metadata.get('weights', {})})")
    finally:
        for agent in specialists.values():
            agent.model.close()


if __name__ == "__main__":
    main()
