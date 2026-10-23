"""
LangChain chat model over llmswitch.

Requires the langchain extra:  pip install "llmswitch-sdk[langchain]"

Set the variables for the provider you want to try, then run:

    OPENAI_API_KEY=sk-... python examples/langchain_chat.py
"""
import asyncio
import os

from langchain_core.prompts import ChatPromptTemplate

from llmswitch.integrations.langchain import ChatLLMSwitch


def build_model() -> ChatLLMSwitch:
    return ChatLLMSwitch(provider="openai", api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini",
                         temperature=0.2, max_tokens=200)


def main() -> None:
    llm = build_model()
    try:
        reply = llm.invoke("Reply with one short sentence about LangChain.")
        print("invoke:", reply.content)
        print("usage:", reply.usage_metadata)

        prompt = ChatPromptTemplate.from_messages([("system", "Answer in five words or fewer."), ("user", "{q}")])
        chain = prompt | llm
        print("chain:", chain.invoke({"q": "What is Python?"}).content)

        print("stream: ", end="")
        for chunk in llm.stream("Count from one to three."):
            print(chunk.content, end="", flush=True)
        print()
    finally:
        llm.close()


async def amain() -> None:
    llm = build_model()
    try:
        reply = await llm.ainvoke("Say hello in one word.")
        print("async invoke:", reply.content)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    main()
    asyncio.run(amain())
