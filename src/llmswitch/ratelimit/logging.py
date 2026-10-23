"""
ratelimit.logging

Structured logging helper emitted through the standard ``logging`` hierarchy so
it is controlled by the same ``llmswitch`` root logger as every other module.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_structured(level: str, message: str, **kwargs: Any) -> None:
    """
    Emit a structured log record through the standard logging hierarchy.

    Args:
        level (str): Log level string — 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        message (str): Log message.
        **kwargs: Additional context fields included as a JSON suffix.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    if kwargs:
        logger.log(log_level, "%s | %s", message, json.dumps(kwargs))
    else:
        logger.log(log_level, "%s", message)
