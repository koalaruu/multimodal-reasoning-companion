"""Immutable interview-session state transitions."""

from __future__ import annotations

from backend.app.schemas import (
    ActionDecision,
    DialogueAction,
    EmotionalState,
    SessionState,
    SessionStatus,
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
            "clarifications_asked": state.clarifications_asked
            + int(decision.action == DialogueAction.REQUEST_CLARIFICATION),
            "last_action": decision.action,
        }
    )


def update_emotional_state(
    state: SessionState,
    emotional_state: EmotionalState,
) -> SessionState:
    """Attach or replace the emotional state on a session without mutating."""

    return state.model_copy(update={"emotional_state": emotional_state})
