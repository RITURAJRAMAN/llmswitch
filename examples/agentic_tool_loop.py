"""
A bounded tool loop with llmswitch's chat model.

Requires the agentic extra:  pip install "llmswitch-sdk[agentic]"

    OPENAI_API_KEY=sk-... python examples/agentic_tool_loop.py
"""
import os

from llmswitch.agentic import ChatLLMSwitch, run_tool_loop


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22 degrees in {city}."


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main() -> None:
    llm = ChatLLMSwitch(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    try:
        history = run_tool_loop(
            llm,
            "What is the weather in Paris, and what is 17 plus 25?",
            [get_weather, add],
            max_steps=4,
        )
        for message in history:
            print(f"{message.type:>12}: {message.content}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
