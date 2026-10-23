"""Tokenizer selection: encoding comes from the model name, with the newest encoding as the default."""
import pytest

from llmswitch.ratelimit import tokenizer
from llmswitch.ratelimit.tokenizer import DEFAULT_ENCODING, count_tokens, truncate_message


@pytest.mark.parametrize("model, encoding", [
    ("gpt-4o", "o200k_base"),
    ("gpt-4o-mini", "o200k_base"),
    ("o3-mini", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5-turbo", "cl100k_base"),
    ("text-embedding-3-small", "cl100k_base"),
    ("text-embedding-ada-002", "cl100k_base"),
])
def test_openai_model_names_resolve_to_their_encoding(model, encoding):
    assert tokenizer._resolve_encoding(model).name == encoding


@pytest.mark.parametrize("model", ["claude-opus-4-5", "gemini-2.5-pro", "aira-gpt-5.4-mini", "Llama-4-Maverick", None])
def test_unknown_or_missing_model_uses_the_newest_default(model):
    assert tokenizer._resolve_encoding(model).name == DEFAULT_ENCODING


def test_default_is_the_newest_encoding_available():
    import tiktoken

    assert DEFAULT_ENCODING == "o200k_harmony"
    assert DEFAULT_ENCODING in tiktoken.list_encoding_names()


def test_count_tokens_uses_the_resolved_encoding():
    text = "Hello, how are you today?"
    assert count_tokens(text, "gpt-4o") == len(tokenizer._resolve_encoding("gpt-4o").encode(text))
    assert count_tokens(text, "claude-opus-4-5") == count_tokens(text, None)
    assert count_tokens(text, "claude-opus-4-5") > 0


def test_count_tokens_with_default_matches_o200k_base_for_ordinary_text():
    import tiktoken

    text = "def f(x): return x ** 2  # ordinary code and prose"
    assert count_tokens(text, "claude-opus-4-5") == len(tiktoken.get_encoding("o200k_base").encode(text))


def test_truncate_message_returns_the_message_when_it_fits():
    assert truncate_message("short text", max_tokens=100, truncate=True, model_name="gpt-4o") == "short text"


def test_truncate_message_is_a_no_op_without_the_flag():
    long_text = "word " * 500
    assert truncate_message(long_text, max_tokens=10, truncate=False, model_name="gpt-4o") == long_text


def test_truncate_message_shortens_long_text_to_the_token_budget():
    long_text = "word " * 500
    cut = truncate_message(long_text, max_tokens=20, truncate=True, model_name="gpt-4o")
    assert count_tokens(cut, "gpt-4o") <= 20
    assert len(cut) < len(long_text)


def test_the_name_based_lookup_replaces_the_old_engine_table():
    assert not hasattr(tokenizer, "get_tokenizer_name")


def test_repeated_texts_are_counted_once():
    tokenizer._count_cached.cache_clear()
    text = "A message that an agent loop sends again and again."
    first = count_tokens(text, "gpt-4o")
    second = count_tokens(text, "gpt-4o")
    info = tokenizer._count_cached.cache_info()
    assert first == second
    assert (info.hits, info.misses) == (1, 1)


def test_long_texts_are_counted_without_filling_the_cache():
    tokenizer._count_cached.cache_clear()
    long_text = "word " * 5000
    assert len(long_text) > tokenizer._CACHEABLE_CHARS
    count_tokens(long_text, "gpt-4o")
    assert tokenizer._count_cached.cache_info().currsize == 0


def test_the_cache_is_keyed_by_model():
    tokenizer._count_cached.cache_clear()
    text = "hello world " * 20
    count_tokens(text, "gpt-4o")
    count_tokens(text, "gpt-4")
    assert tokenizer._count_cached.cache_info().misses == 2
