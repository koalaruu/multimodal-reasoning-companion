"""Deterministic extraction of explicit requests from candidate text."""

from __future__ import annotations

import re
from collections.abc import Sequence

from backend.app.schemas import ExplicitSignals

_HINT_PHRASES = (
    "could i have a hint",
    "can i have a hint",
    "give me a hint",
    "help me with this",
)
_THINKING_TIME_PHRASES = (
    "give me a moment",
    "let me think",
    "i need a moment",
    "one moment please",
)
_STOP_PHRASES = (
    "stop the practice",
    "end the practice",
    "end the session",
    "stop the session",
    "i'd like to stop",
    "i want to stop",
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _contains_phrase(text: str, phrases: Sequence[str]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
        for phrase in phrases
    )


def extract_explicit_signals(text: str) -> ExplicitSignals:
    """Return only explicit requests; precedence belongs to dialogue policy."""

    normalized = _normalize(text)
    return ExplicitSignals(
        hint_requested=_contains_phrase(normalized, _HINT_PHRASES),
        thinking_time_requested=_contains_phrase(normalized, _THINKING_TIME_PHRASES),
        stop_requested=_contains_phrase(normalized, _STOP_PHRASES),
    )
