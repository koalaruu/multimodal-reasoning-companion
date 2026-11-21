"""Application service that orchestrates one interview turn."""

from __future__ import annotations

from backend.app.cognitive.state_engine import apply_decision
from backend.app.perception.answer_quality import extract_answer_quality
from backend.app.perception.text_signals import extract_explicit_signals
from backend.app.reasoning.dialogue_policy import decide
from backend.app.reasoning.llm_agent import (
    GeminiReplyGenerator,
    ReplyGenerationResult,
)
from backend.app.reasoning.response_generator import generate_local_reply
from backend.app.schemas import ReplySource, SessionState, TurnResult


class InterviewService:
    """Run the offline text pipeline without transport or network dependencies."""

    def __init__(
        self, reply_generator: GeminiReplyGenerator | None = None
    ) -> None:
        self._reply_generator = reply_generator

    def process_turn(
        self, state: SessionState, candidate_input: str
    ) -> TurnResult:
        signals = extract_explicit_signals(candidate_input)
        content_evidence = extract_answer_quality(candidate_input, signals)
        decision = decide(state, signals, content_evidence)
        updated_state = apply_decision(state, decision)
        local_reply = generate_local_reply(decision)
        generated_reply = (
            self._reply_generator.generate(
                decision=decision,
                candidate_input=candidate_input,
                local_reply=local_reply,
            )
            if self._reply_generator is not None
            else ReplyGenerationResult(
                reply=local_reply,
                source=ReplySource.LOCAL,
            )
        )
        return TurnResult(
            signals=signals,
            content_evidence=content_evidence,
            decision=decision,
            updated_state=updated_state,
            reply=generated_reply.reply,
            reply_source=generated_reply.source,
            fallback_reason=generated_reply.fallback_reason,
        )
