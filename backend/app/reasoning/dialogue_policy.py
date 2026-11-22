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
    if evidence.input_type == InputType.MISSING:
        return ActionDecision(
            action=DialogueAction.REQUEST_INPUT,
            reason=DecisionReason.INPUT_MISSING,
        )
    if (
        evidence.relevance in {AnswerRelevance.IRRELEVANT, AnswerRelevance.UNCLEAR}
        and state.clarifications_asked >= _MAX_CLARIFICATIONS
    ):
        return ActionDecision(
            action=DialogueAction.OFFER_HINT,
            reason=DecisionReason.CLARIFICATION_LIMIT_REACHED,
        )
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
    ):
        return ActionDecision(
            action=DialogueAction.ASK_FOLLOWUP,
            reason=DecisionReason.CLASS_LEVEL_PERFORMANCE_MISSING,
            followup_topic=FollowupTopic.CLASS_LEVEL_PERFORMANCE,
        )
    if not evidence.project_described:
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
