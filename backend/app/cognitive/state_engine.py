"""Immutable interview-session state transitions."""

from __future__ import annotations

from backend.app.schemas import (
    ActionDecision,
    DecisionReason,
    DialogueAction,
    EmotionalState,
    SessionState,
    SessionStatus,
)

# How many prior interviewer replies to keep for anti-repetition context.
_MAX_RECENT = 4


def apply_decision(state: SessionState, decision: ActionDecision) -> SessionState:
    """Return the next state without mutating the supplied state object."""

    advances_topic = decision.reason in {
        DecisionReason.TOPIC_ADVANCED_WITHOUT_DETAIL,
        DecisionReason.ANSWER_COMPLETE,
    }
    return state.model_copy(
        update={
            "status": (
                SessionStatus.ENDED
                if decision.action == DialogueAction.END_SESSION
                else state.status
            ),
            "turn_index": state.turn_index + 1,
            "hints_offered": (
                0
                if advances_topic
                else state.hints_offered
                + int(decision.action == DialogueAction.OFFER_HINT)
            ),
            "followups_asked": state.followups_asked
            + int(decision.action == DialogueAction.ASK_FOLLOWUP),
            "clarifications_asked": (
                0
                if advances_topic
                else state.clarifications_asked
                + int(decision.action == DialogueAction.REQUEST_CLARIFICATION)
            ),
            "last_action": decision.action,
        }
    )


def record_reply(state: SessionState, reply: str) -> SessionState:
    """Keep the last few interviewer replies so wording can avoid repeats."""

    trimmed = reply.strip()
    if not trimmed:
        return state
    return state.model_copy(
        update={"recent_replies": [*state.recent_replies, trimmed][-_MAX_RECENT:]}
    )


def update_emotional_state(
    state: SessionState,
    emotional_state: EmotionalState,
) -> SessionState:
    """Attach or replace the emotional state on a session without mutating."""

    return state.model_copy(update={"emotional_state": emotional_state})
