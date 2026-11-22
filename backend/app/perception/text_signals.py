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

# Questions about the tool itself, not about the candidate's project.
# Answering these with another interview question is what makes the system
# feel broken, so they get a direct explanation instead.
_META_PHRASES = (
    "what is this",
    "what's this",
    "what is it for",
    "what is this for",
    "what does this do",
    "what is the purpose",
    "how does this work",
    "what am i supposed to do",
    "what should i do",
    "ini apa",
    "ini buat apa",
    "ini untuk apa",
    "fungsinya apa",
    "fungsi chatbot",
    "chatbotnya apa",
    "apa fungsinya",
    "gimana cara pakai",
    "bagaimana cara pakai",
    "chatbot ini",
    "aplikasi ini",
    "sistem ini",
    "tool ini",
    "web ini",
    "buat apa",
    "untuk apa",
    "kegunaannya",
    "maksudnya apa",
    "ini gunanya",
)

# Indonesian phrases that wrap the noun, e.g. "fungsi chatbotnya apa".
_META_REGEX = re.compile(
    r"(?:fungsi|kegunaan|tujuan|maksud)\w*\s+\w*\s*(?:apa|apasih|itu apa)"
    r"|\b(?:apa|apasih)\s+(?:fungsi|kegunaan|tujuan|maksud)\w*"
    r"|\bini\s+(?:buat|untuk)\s+apa"
    r"|\bcara\s+(?:pakai|menggunakan|penggunaan)"
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
        meta_question_asked=(
            _contains_phrase(normalized, _META_PHRASES)
            or _META_REGEX.search(normalized) is not None
        ),
    )
