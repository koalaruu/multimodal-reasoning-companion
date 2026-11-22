"""Immutable interview-session state transitions."""

from __future__ import annotations

from backend.app.schemas import (
    ActionDecision,
    ContentEvidence,
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
            "audio_waits_offered": state.audio_waits_offered
            + int(
                decision.action == DialogueAction.WAIT
                and decision.reason == DecisionReason.AUDIO_TURN_INCOMPLETE
            ),
            "give_space_offered": state.give_space_offered
            + int(decision.action == DialogueAction.GIVE_SPACE),
            "last_action": decision.action,
        }
    )


def merge_evidence(
    previous: ContentEvidence | None,
    current: ContentEvidence,
) -> ContentEvidence:
    """Accumulate positive evidence across turns.

    A candidate who names the project in one turn and the metric in the next
    has answered the question. Evidence is only ever added, never revoked:
    a later vague turn must not erase what was already established.
    """

    if previous is None:
        return current

    return current.model_copy(
        update={
            "project_described": (
                previous.project_described or current.project_described
            ),
            "evaluation_method_mentioned": (
                previous.evaluation_method_mentioned
                or current.evaluation_method_mentioned
            ),
            "evaluation_result_mentioned": (
                previous.evaluation_result_mentioned
                or current.evaluation_result_mentioned
            ),
            "macro_f1_mentioned": (
                previous.macro_f1_mentioned or current.macro_f1_mentioned
            ),
            "class_imbalance_mentioned": (
                previous.class_imbalance_mentioned
                or current.class_imbalance_mentioned
            ),
            "class_level_performance_mentioned": (
                previous.class_level_performance_mentioned
                or current.class_level_performance_mentioned
            ),
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
