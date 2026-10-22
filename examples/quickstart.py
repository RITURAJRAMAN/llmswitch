"""
Quickstart for llmswitch 0.1.0.

Set the variables for the provider you want to try, then run:

    OPENAI_API_KEY=sk-... python examples/quickstart.py
"""
import asyncio
import os

from llmswitch import AsyncLLMSwitch, LLMSwitch

MESSAGES = [{"role": "user", "content": "Reply with one short sentence about Python."}]


def sync_demo(api_key: str) -> None:
    client = LLMSwitch(provider="openai", api_key=api_key, model="gpt-4o-mini")
    reply = client.chat.completions.create(messages=MESSAGES)
    print("sync chat:", reply.choices[0].message.content)

    vectors = client.embeddings.create(["llmswitch", "python"], model="text-embedding-3-small")
    print("sync embeddings:", len(vectors.data), "vectors of length", len(vectors.data[0].embedding))


async def async_demo(api_key: str) -> None:
    async with AsyncLLMSwitch(provider="openai", api_key=api_key, model="gpt-4o-mini") as client:
        stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
        print("async stream: ", end="")
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()


if __name__ == "__main__":
    key = os.environ["OPENAI_API_KEY"]
    sync_demo(key)
    asyncio.run(async_demo(key))
