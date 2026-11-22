"""Deterministic dialogue action selection with explicit precedence."""

from __future__ import annotations

from backend.app.schemas import (
    ActionDecision,
    AnswerRelevance,
    ContentEvidence,
    DecisionReason,
    DialogueAction,
    ExplicitSignals,
    FollowupTopic,
    InputType,
    SessionState,
    SessionStatus,
)

# After this many clarification requests the policy stops repeating itself
# and offers a concrete hint instead.
_MAX_CLARIFICATIONS = 2

# After this many hints the policy stops hinting and moves the interview on.
_MAX_HINTS = 2

# Facial evidence of sustained high arousal (angry/fearful/disgusted) means the
# candidate is under load. Pressing another question adds pressure, so the
# policy gives space instead. This is advisory: a low-confidence or neutral
# reading never changes the action.
_HIGH_LOAD_EMOTIONS = frozenset({"angry", "fearful", "disgusted"})
_HIGH_LOAD_MIN_CONFIDENCE = 0.35
_MAX_GIVE_SPACE = 1


def decide(
    state: SessionState,
    signals: ExplicitSignals,
    evidence: ContentEvidence,
) -> ActionDecision:
    """Select exactly one action; earlier branches have higher precedence."""

    if state.status == SessionStatus.ENDED:
        return ActionDecision(
            action=DialogueAction.END_SESSION,
            reason=DecisionReason.SESSION_ALREADY_ENDED,
        )
    if signals.stop_requested:
        return ActionDecision(
            action=DialogueAction.END_SESSION,
            reason=DecisionReason.EXPLICIT_STOP_REQUEST,
        )
    if signals.thinking_time_requested:
        return ActionDecision(
            action=DialogueAction.WAIT,
            reason=DecisionReason.EXPLICIT_THINKING_TIME_REQUEST,
        )
    if signals.hint_requested:
        return ActionDecision(
            action=DialogueAction.OFFER_HINT,
            reason=DecisionReason.EXPLICIT_HINT_REQUEST,
        )
    if signals.meta_question_asked:
        return ActionDecision(
            action=DialogueAction.ANSWER_META_QUESTION,
            reason=DecisionReason.META_QUESTION_ANSWERED,
        )
    if evidence.input_type == InputType.MISSING:
        return ActionDecision(
            action=DialogueAction.REQUEST_INPUT,
            reason=DecisionReason.INPUT_MISSING,
        )
    decision = _content_decision(state, evidence)
    if _visual_load_is_high(state) and _should_yield_to_load(
        state, decision, evidence
    ):
        return ActionDecision(
            action=DialogueAction.GIVE_SPACE,
            reason=DecisionReason.VISUAL_LOAD_HIGH,
        )
    if evidence.minimal_answer:
        if state.hints_offered >= _MAX_HINTS:
            return ActionDecision(
                action=DialogueAction.CONTINUE_INTERVIEW,
                reason=DecisionReason.TOPIC_ADVANCED_WITHOUT_DETAIL,
            )
        return ActionDecision(
            action=DialogueAction.OFFER_HINT,
            reason=DecisionReason.MINIMAL_ANSWER_NEEDS_CHOICE,
        )
    if decision.action == DialogueAction.REQUEST_CLARIFICATION:
        repeats = (
            state.clarifications_asked >= _MAX_CLARIFICATIONS
            or (
                decision.reason == DecisionReason.PROJECT_EXAMPLE_NEEDED
                and state.clarifications_asked >= 1
            )
        )
        if repeats:
            if state.hints_offered >= _MAX_HINTS:
                return ActionDecision(
                    action=DialogueAction.CONTINUE_INTERVIEW,
                    reason=DecisionReason.TOPIC_ADVANCED_WITHOUT_DETAIL,
                )
            return ActionDecision(
                action=DialogueAction.OFFER_HINT,
                reason=DecisionReason.CLARIFICATION_LIMIT_REACHED,
            )
    return decision


def _visual_load_is_high(state: SessionState) -> bool:
    """True only for a confident high-arousal reading with a visible face."""

    emotional = state.emotional_state
    if emotional is None or not emotional.face_available:
        return False
    if emotional.dominant_emotion.value not in _HIGH_LOAD_EMOTIONS:
        return False
    return emotional.confidence >= _HIGH_LOAD_MIN_CONFIDENCE


def _should_yield_to_load(
    state: SessionState,
    decision: ActionDecision,
    evidence: ContentEvidence,
) -> bool:
    """Give space only when the alternative would add pressure.

    Explicit requests still win: a candidate who asks for a hint wants help,
    not silence. The yield is also capped so it cannot replace the interview.
    """

    if state.give_space_offered >= _MAX_GIVE_SPACE:
        return False
    if evidence.input_type != InputType.ANSWER:
        return False
    return decision.action in {
        DialogueAction.REQUEST_CLARIFICATION,
        DialogueAction.ASK_FOLLOWUP,
        DialogueAction.CONTINUE_INTERVIEW,
    }


def _content_decision(
    state: SessionState,
    evidence: ContentEvidence,
) -> ActionDecision:
    """Content-based decision before the clarification-repeat guard."""

    if evidence.relevance == AnswerRelevance.IRRELEVANT:
        return ActionDecision(
            action=DialogueAction.REQUEST_CLARIFICATION,
            reason=DecisionReason.ANSWER_IRRELEVANT,
        )
    if evidence.relevance == AnswerRelevance.UNCLEAR:
        return ActionDecision(
            action=DialogueAction.REQUEST_CLARIFICATION,
            reason=DecisionReason.ANSWER_UNCLEAR,
        )
    if (
        evidence.macro_f1_mentioned
        and evidence.class_imbalance_mentioned
        and not evidence.class_level_performance_mentioned
        and state.followups_asked < _MAX_HINTS
    ):
        return ActionDecision(
            action=DialogueAction.ASK_FOLLOWUP,
            reason=DecisionReason.CLASS_LEVEL_PERFORMANCE_MISSING,
            followup_topic=FollowupTopic.CLASS_LEVEL_PERFORMANCE,
        )
    if not evidence.project_described:
        if evidence.evaluation_method_mentioned:
            return ActionDecision(
                action=DialogueAction.REQUEST_CLARIFICATION,
                reason=DecisionReason.PROJECT_EXAMPLE_NEEDED,
            )
        return ActionDecision(
            action=DialogueAction.REQUEST_CLARIFICATION,
            reason=DecisionReason.PROJECT_EVIDENCE_MISSING,
        )
    if evidence.project_described and not evidence.evaluation_method_mentioned:
        return ActionDecision(
            action=DialogueAction.ASK_FOLLOWUP,
            reason=DecisionReason.EVALUATION_EVIDENCE_MISSING,
            followup_topic=FollowupTopic.EVALUATION_METHOD,
        )
    return ActionDecision(
        action=DialogueAction.CONTINUE_INTERVIEW,
        reason=DecisionReason.ANSWER_COMPLETE,
    )
