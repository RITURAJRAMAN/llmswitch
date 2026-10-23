"""
A bounded tool loop with a LangChain chat model over llmswitch.

Requires the langchain extra:  pip install "llmswitch-sdk[langchain]"

    OPENAI_API_KEY=sk-... python examples/langchain_tool_loop.py
"""
import os

from langchain_core.tools import tool

from llmswitch.integrations.langchain import ChatLLMSwitch, run_tool_loop


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22 degrees in {city}."


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def main() -> None:
    llm = ChatLLMSwitch(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini",
                        temperature=0)
    try:
        history = run_tool_loop(
            llm,
            "What is the weather in Paris, and what is 17 plus 25?",
            [get_weather, add],
            max_steps=4,
        )
        for message in history:
            print(f"{type(message).__name__:>12}: {message.content}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
