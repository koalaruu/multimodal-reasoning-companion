"""Behavior tests for the deterministic text interview baseline."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from backend.app.cognitive.state_engine import apply_decision
from backend.app.schemas import (
    ActionDecision,
    DecisionReason,
    DialogueAction,
    FollowupTopic,
    SessionState,
    SessionStatus,
)
from backend.app.service import InterviewService


@pytest.fixture
def service() -> InterviewService:
    return InterviewService()


@pytest.mark.parametrize(
    ("candidate_input", "action", "reason", "topic"),
    [
        (
            "Could I have a hint?",
            DialogueAction.OFFER_HINT,
            DecisionReason.EXPLICIT_HINT_REQUEST,
            FollowupTopic.NONE,
        ),
        (
            "Give me a moment to think.",
            DialogueAction.WAIT,
            DecisionReason.EXPLICIT_THINKING_TIME_REQUEST,
            FollowupTopic.NONE,
        ),
        (
            "I built a classifier for support tickets.",
            DialogueAction.ASK_FOLLOWUP,
            DecisionReason.EVALUATION_EVIDENCE_MISSING,
            FollowupTopic.EVALUATION_METHOD,
        ),
        (
            "I used macro-F1 because the classes were imbalanced.",
            DialogueAction.ASK_FOLLOWUP,
            DecisionReason.CLASS_LEVEL_PERFORMANCE_MISSING,
            FollowupTopic.CLASS_LEVEL_PERFORMANCE,
        ),
        (
            "I'd like to stop the practice here.",
            DialogueAction.END_SESSION,
            DecisionReason.EXPLICIT_STOP_REQUEST,
            FollowupTopic.NONE,
        ),
    ],
)
def test_five_baseline_cases(
    service: InterviewService,
    candidate_input: str,
    action: DialogueAction,
    reason: DecisionReason,
    topic: FollowupTopic,
) -> None:
    result = service.process_turn(SessionState(session_id="case"), candidate_input)

    assert result.decision.action == action
    assert result.decision.reason == reason
    assert result.decision.followup_topic == topic


def test_stop_has_precedence_over_wait_and_hint(service: InterviewService) -> None:
    result = service.process_turn(
        SessionState(session_id="precedence"),
        "I'd like to stop the practice, but give me a moment and a hint.",
    )

    assert result.decision.action == DialogueAction.END_SESSION
    assert result.decision.reason == DecisionReason.EXPLICIT_STOP_REQUEST


def test_wait_has_precedence_over_hint(service: InterviewService) -> None:
    result = service.process_turn(
        SessionState(session_id="precedence"),
        "Give me a moment to think, then give me a hint.",
    )

    assert result.decision.action == DialogueAction.WAIT
    assert result.reply == "Of course—take your time."


def test_ended_session_is_terminal(service: InterviewService) -> None:
    state = SessionState(session_id="ended", status=SessionStatus.ENDED)

    result = service.process_turn(state, "Could I have a hint?")

    assert result.decision.action == DialogueAction.END_SESSION
    assert result.decision.reason == DecisionReason.SESSION_ALREADY_ENDED
    assert result.updated_state.status == SessionStatus.ENDED
    assert result.reply == "Understood. We'll end the practice here."


def test_whitespace_requests_input(service: InterviewService) -> None:
    result = service.process_turn(SessionState(session_id="empty"), "   \n")

    assert result.decision.action == DialogueAction.REQUEST_INPUT
    assert result.decision.reason == DecisionReason.INPUT_MISSING


def test_state_does_not_store_candidate_input(service: InterviewService) -> None:
    sensitive_input = "My private project description"

    result = service.process_turn(
        SessionState(session_id="privacy"), sensitive_input
    )
    serialized_state = json.dumps(result.updated_state.model_dump(mode="json"))

    assert "candidate_input" not in type(result.updated_state).model_fields
    assert sensitive_input not in serialized_state


def test_apply_decision_does_not_mutate_original_state() -> None:
    state = SessionState(session_id="immutable")
    decision = ActionDecision(
        action=DialogueAction.OFFER_HINT,
        reason=DecisionReason.EXPLICIT_HINT_REQUEST,
    )

    updated = apply_decision(state, decision)

    assert updated is not state
    assert state.turn_index == 0
    assert state.hints_offered == 0
    assert state.last_action is None
    assert updated.turn_index == 1
    assert updated.hints_offered == 1


def test_followup_decision_requires_topic() -> None:
    with pytest.raises(ValidationError):
        ActionDecision(
            action=DialogueAction.ASK_FOLLOWUP,
            reason=DecisionReason.EVALUATION_EVIDENCE_MISSING,
        )


def test_session_state_rejects_blank_id_and_negative_counter() -> None:
    with pytest.raises(ValidationError):
        SessionState(session_id="   ")
    with pytest.raises(ValidationError):
        SessionState(session_id="invalid", turn_index=-1)
