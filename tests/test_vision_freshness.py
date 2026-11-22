"""Facial evidence must not outlive the camera or a stale reading."""

import pytest

from backend.app.schemas import EmotionalState, EmotionLabel, SessionState
from backend.app.service import InterviewService

ANSWER = "I built a ticket classifier and evaluated it with macro-F1."


def _state(**overrides) -> SessionState:
    base = {
        "session_id": "vision-freshness",
        "emotional_state": EmotionalState(
            face_available=True,
            dominant_emotion=EmotionLabel.FEARFUL,
            confidence=0.7,
        ),
    }
    base.update(overrides)
    return SessionState(**base)


def test_explicit_null_clears_previous_facial_evidence() -> None:
    result = InterviewService().process_turn(
        _state(), ANSWER, emotional_state=None
    )

    assert result.updated_state.emotional_state is None
    assert result.decision.action.value != "give_space"


def test_omitted_argument_preserves_stored_facial_evidence() -> None:
    result = InterviewService().process_turn(_state(), ANSWER)

    assert result.updated_state.emotional_state is not None
    assert result.decision.action.value == "give_space"


def test_new_reading_replaces_previous_one() -> None:
    calm = EmotionalState(
        face_available=True,
        dominant_emotion=EmotionLabel.NEUTRAL,
        confidence=0.0,
    )
    result = InterviewService().process_turn(
        _state(), ANSWER, emotional_state=calm
    )

    assert result.updated_state.emotional_state == calm
    assert result.decision.action.value != "give_space"


def test_cleared_evidence_stops_influencing_later_turns() -> None:
    service = InterviewService()
    first = service.process_turn(_state(), ANSWER, emotional_state=None)
    second = service.process_turn(first.updated_state, ANSWER)

    assert second.updated_state.emotional_state is None
    assert second.decision.action.value != "give_space"
