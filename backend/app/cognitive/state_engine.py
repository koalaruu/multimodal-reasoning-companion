"""Content evidence extraction and immutable session state transitions."""

from __future__ import annotations

import re
from collections.abc import Sequence

from backend.app.schemas import (
    ActionDecision,
    ContentEvidence,
    DialogueAction,
    SessionState,
    SessionStatus,
)

_PROJECT_TERMS = ("built", "developed", "implemented", "project")
_EVALUATION_TERMS = (
    "evaluated",
    "evaluation",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "test set",
    "validation set",
)
_MACRO_F1_TERMS = ("macro-f1", "macro f1", "macro f-score")
_CLASS_IMBALANCE_TERMS = ("imbalanced", "class imbalance", "minority class")
_CLASS_LEVEL_TERMS = ("per-class", "per class", "hardest class", "class-level")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_term(text: str, terms: Sequence[str]) -> bool:
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None
        for term in terms
    )


def extract_content_evidence(text: str) -> ContentEvidence:
    """Extract narrow textual evidence, not psychological user attributes."""

    normalized = _normalize(text)
    return ContentEvidence(
        project_mentioned=_contains_term(normalized, _PROJECT_TERMS),
        evaluation_mentioned=_contains_term(normalized, _EVALUATION_TERMS),
        macro_f1_mentioned=_contains_term(normalized, _MACRO_F1_TERMS),
        class_imbalance_mentioned=_contains_term(normalized, _CLASS_IMBALANCE_TERMS),
        class_level_performance_mentioned=_contains_term(
            normalized, _CLASS_LEVEL_TERMS
        ),
    )


def apply_decision(state: SessionState, decision: ActionDecision) -> SessionState:
    """Return the next state without mutating the supplied state object."""

    return state.model_copy(
        update={
            "status": (
                SessionStatus.ENDED
                if decision.action == DialogueAction.END_SESSION
                else state.status
            ),
            "turn_index": state.turn_index + 1,
            "hints_offered": state.hints_offered
            + int(decision.action == DialogueAction.OFFER_HINT),
            "followups_asked": state.followups_asked
            + int(decision.action == DialogueAction.ASK_FOLLOWUP),
            "last_action": decision.action,
        }
    )
