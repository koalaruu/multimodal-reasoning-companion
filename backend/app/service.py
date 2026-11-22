"""Application service that orchestrates one interview turn.

Groq owns the conversational turn: it decides what to ask next and how to say
it, grounded in the uploaded CV, the conversation so far, and the candidate's
proxy expression. Two things stay deterministic and local:

- explicit control intents (stop, wait, meta questions), which must not be
  reinterpreted by a model;
- the session can only end when the candidate asks to stop.
"""

from __future__ import annotations

from backend.app.cognitive.state_engine import (
    apply_decision,
    merge_evidence,
    record_reply,
    update_emotional_state,
)
from backend.app.perception.answer_quality import extract_answer_quality
from backend.app.perception.text_signals import extract_explicit_signals
from backend.app.reasoning.dialogue_policy import decide
from backend.app.reasoning.groq_interviewer import GroqInterviewer
from backend.app.reasoning.session_review import (
    GroqReviewer,
    compute_coverage,
)
from backend.app.reasoning.response_generator import generate_local_reply
from backend.app.schemas import (
    ActionDecision,
    DecisionReason,
    DialogueAction,
    EmotionalState,
    ReplyFallbackReason,
    ReplySource,
    SessionState,
    TurnResult,
)

# Actions whose wording must stay exactly as written: they express a control
# intent the candidate already stated, so a model must not reinterpret them.
_DETERMINISTIC_ACTIONS = frozenset(
    {
        DialogueAction.END_SESSION,
        DialogueAction.WAIT,
        DialogueAction.ANSWER_META_QUESTION,
        DialogueAction.REQUEST_INPUT,
        DialogueAction.GIVE_SPACE,
    }
)

_MAX_CONVERSATION_MESSAGES = 24


class InterviewService:
    """Run one interview turn with Groq as the conversational driver."""

    def __init__(
        self,
        interviewer: GroqInterviewer | None = None,
        reviewer: GroqReviewer | None = None,
    ) -> None:
        self._interviewer = interviewer
        self._reviewer = reviewer

    def review_session(self, state: SessionState):
        """Summarize a finished practice session.

        covered/missing come from accumulated evidence and are deterministic.
        suggestions/questions come from Groq and are labelled with their source.
        """

        covered, missing = compute_coverage(state.cumulative_evidence)
        if self._reviewer is None:
            from backend.app.reasoning.session_review import (
                _local_suggestions,
            )
            from backend.app.schemas import SessionReview, ReplySource

            return SessionReview(
                turns=state.turn_index,
                covered=covered,
                missing=missing,
                suggestions=_local_suggestions(missing),
                questions_to_ask=[],
                review_source=ReplySource.LOCAL,
            )
        return self._reviewer.review(
            state=state, covered=covered, missing=missing
        )

    def process_turn(
        self,
        state: SessionState,
        candidate_input: str,
        emotional_state: EmotionalState | None = None,
    ) -> TurnResult:
        signals = extract_explicit_signals(candidate_input)
        turn_evidence = extract_answer_quality(candidate_input, signals)
        cumulative_evidence = merge_evidence(
            state.cumulative_evidence, turn_evidence
        )

        # The current observation must be on the state before the policy runs,
        # otherwise the policy always sees the previous turn's face reading.
        evaluated_state = (
            update_emotional_state(state, emotional_state)
            if emotional_state is not None
            else state
        )
        decision = decide(evaluated_state, signals, cumulative_evidence)
        updated_state = apply_decision(evaluated_state, decision)
        updated_state = updated_state.model_copy(
            update={"cumulative_evidence": cumulative_evidence}
        )

        reply, source, fallback_reason = self._render(
            state=updated_state,
            candidate_input=candidate_input,
            decision=decision,
        )

        # Report who actually drove the turn. On fallback the reply text came
        # from the deterministic decision, so that decision stays truthful.
        if source == ReplySource.GROQ and decision.action not in {
            DialogueAction.GIVE_SPACE,
        }:
            decision = ActionDecision(
                action=DialogueAction.CONVERSE,
                reason=DecisionReason.GROQ_LED_TURN,
            )

        conversation = [
            *state.conversation,
            {"role": "candidate", "content": candidate_input},
            {"role": "interviewer", "content": reply},
        ][-_MAX_CONVERSATION_MESSAGES:]

        final_state = record_reply(updated_state, reply)
        final_state = final_state.model_copy(
            update={"conversation": conversation}
        )

        return TurnResult(
            signals=signals,
            content_evidence=turn_evidence,
            decision=decision,
            updated_state=final_state,
            reply=reply,
            reply_source=source,
            fallback_reason=fallback_reason,
        )

    def _render(
        self,
        *,
        state: SessionState,
        candidate_input: str,
        decision: ActionDecision,
    ) -> tuple[str, ReplySource, ReplyFallbackReason | None]:
        local_reply = generate_local_reply(decision)

        if decision.action in _DETERMINISTIC_ACTIONS:
            return local_reply, ReplySource.LOCAL, None

        if self._interviewer is None or not self._interviewer.available:
            return local_reply, ReplySource.LOCAL, None

        result = self._interviewer.respond(
            candidate_input=candidate_input,
            conversation=state.conversation,
            cv_text=state.cv_text,
            target_role=state.target_role,
            job_description=state.job_description,
            emotional_state=state.emotional_state,
            local_reply=local_reply,
        )
        return result.reply, result.source, result.fallback_reason
