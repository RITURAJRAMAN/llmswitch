DEFAULT_RPM = 10000
DEFAULT_TPM = 1000000
DEFAULT_EMBEDDING_RPM = 10000
DEFAULT_EMBEDDING_TPM = 1000000
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_COMPLETION_CONTEXT_WINDOW = 123000  # (based on 124000 token limit minus some buffer)

# Retry / exponential back-off defaults
DEFAULT_MAX_RETRIES = 3  # number of *retries* (not total attempts)
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds — doubling each attempt
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds — cap on the backoff ceiling

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-5"
