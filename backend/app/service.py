"""Application service that orchestrates one interview turn."""

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
from backend.app.reasoning.llm_agent import (
    GroqReplyGenerator,
    ReplyGenerationResult,
)
from backend.app.reasoning.response_generator import generate_local_reply
from backend.app.schemas import (
    EmotionalState,
    ReplySource,
    SessionState,
    TurnResult,
)


class InterviewService:
    """Run the offline text pipeline without transport or network dependencies."""

    def __init__(
        self, reply_generator: GroqReplyGenerator | None = None
    ) -> None:
        self._reply_generator = reply_generator

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
        decision = decide(state, signals, cumulative_evidence)
        updated_state = apply_decision(state, decision)
        updated_state = updated_state.model_copy(
            update={"cumulative_evidence": cumulative_evidence}
        )
        if emotional_state is not None:
            updated_state = update_emotional_state(
                updated_state, emotional_state
            )
        local_reply = generate_local_reply(decision)
        generated_reply = (
            self._reply_generator.generate(
                decision=decision,
                candidate_input=candidate_input,
                local_reply=local_reply,
                emotional_state=updated_state.emotional_state,
                recent_replies=state.recent_replies,
            )
            if self._reply_generator is not None
            else ReplyGenerationResult(
                reply=local_reply,
                source=ReplySource.LOCAL,
            )
        )
        final_state = record_reply(updated_state, generated_reply.reply)
        return TurnResult(
            signals=signals,
            content_evidence=turn_evidence,
            decision=decision,
            updated_state=final_state,
            reply=generated_reply.reply,
            reply_source=generated_reply.source,
            fallback_reason=generated_reply.fallback_reason,
        )
