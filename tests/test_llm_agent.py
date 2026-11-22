"""Offline tests for the Groq-driven conversational interviewer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.app.reasoning.groq_interviewer import (
    DEFAULT_GROQ_MODEL,
    GroqInterviewer,
)
from backend.app.schemas import (
    BlendshapeReading,
    DialogueAction,
    EmotionLabel,
    ReplyFallbackReason,
    ReplySource,
    SessionState,
    VisionObservation,
)
from backend.app.perception.vision_analyzer import analyze
from backend.app.service import InterviewService


def response_with_content(content: Any) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(response, error))


def reply_json(reply: str) -> str:
    return json.dumps({"reply": reply})


def context_of(client: FakeClient) -> dict[str, Any]:
    call = client.chat.completions.calls[0]
    return json.loads(call["messages"][1]["content"])


def happy_state():
    return analyze(
        VisionObservation(
            face_available=True,
            face_count=1,
            blendshapes=[
                BlendshapeReading(name="mouthSmileLeft", score=0.85),
                BlendshapeReading(name="mouthSmileRight", score=0.8),
                BlendshapeReading(name="cheekSquintLeft", score=0.7),
                BlendshapeReading(name="cheekSquintRight", score=0.65),
            ],
        )
    )


class TestGroqInterviewer:
    def test_returns_model_reply(self) -> None:
        client = FakeClient(response_with_content(reply_json("Tell me about the model.")))
        interviewer = GroqInterviewer(client=client)

        result = interviewer.respond(
            candidate_input="I built a classifier.",
            conversation=[],
            local_reply="fallback",
        )

        assert result.reply == "Tell me about the model."
        assert result.source == ReplySource.GROQ
        assert result.fallback_reason is None
        assert client.chat.completions.calls[0]["model"] == DEFAULT_GROQ_MODEL

    def test_without_client_returns_local(self) -> None:
        interviewer = GroqInterviewer()

        result = interviewer.respond(
            candidate_input="hello",
            conversation=[],
            local_reply="local text",
        )

        assert interviewer.available is False
        assert result.reply == "local text"
        assert result.source == ReplySource.LOCAL

    def test_timeout_falls_back(self) -> None:
        client = FakeClient(error=TimeoutError("slow"))
        result = GroqInterviewer(client=client).respond(
            candidate_input="hello", conversation=[], local_reply="local text"
        )

        assert result.source == ReplySource.LOCAL_FALLBACK
        assert result.fallback_reason == ReplyFallbackReason.TIMEOUT
        assert result.reply == "local text"

    def test_sdk_error_falls_back(self) -> None:
        client = FakeClient(error=RuntimeError("network down"))
        result = GroqInterviewer(client=client).respond(
            candidate_input="hello", conversation=[], local_reply="local text"
        )

        assert result.fallback_reason == ReplyFallbackReason.SDK_ERROR

    @pytest.mark.parametrize(
        "content",
        ["not json", '{"wrong": "field"}'],
    )
    def test_malformed_payload_falls_back(self, content: str) -> None:
        client = FakeClient(response_with_content(content))
        result = GroqInterviewer(client=client).respond(
            candidate_input="hello", conversation=[], local_reply="local text"
        )

        assert result.source == ReplySource.LOCAL_FALLBACK
        assert result.fallback_reason == ReplyFallbackReason.MALFORMED_OUTPUT

    def test_empty_reply_is_invalid(self) -> None:
        client = FakeClient(response_with_content('{"reply": "   "}'))
        result = GroqInterviewer(client=client).respond(
            candidate_input="hello", conversation=[], local_reply="local text"
        )

        assert result.fallback_reason == ReplyFallbackReason.INVALID_REPLY

    def test_cv_and_role_reach_the_prompt(self) -> None:
        client = FakeClient(response_with_content(reply_json("Which metric?")))
        GroqInterviewer(client=client).respond(
            candidate_input="I built a model.",
            conversation=[{"role": "candidate", "content": "hi"}],
            cv_text="Built a ticket classifier. Macro-F1 0.81.",
            target_role="ML Engineer Intern",
            local_reply="fallback",
        )

        context = context_of(client)
        assert context["target_role"] == "ML Engineer Intern"
        assert "Macro-F1 0.81" in context["cv_summary"]
        assert context["conversation"][0]["content"] == "hi"

    def test_emotional_state_is_advisory_in_prompt(self) -> None:
        client = FakeClient(response_with_content(reply_json("Go on.")))
        GroqInterviewer(client=client).respond(
            candidate_input="done",
            conversation=[],
            emotional_state=happy_state(),
            local_reply="fallback",
        )

        context = context_of(client)
        assert context["emotional_state"]["dominant_emotion"] == "happy"
        assert "advisory" in context["emotional_state"]["note"].lower()

    def test_no_face_omits_emotional_state(self) -> None:
        client = FakeClient(response_with_content(reply_json("Go on.")))
        GroqInterviewer(client=client).respond(
            candidate_input="done",
            conversation=[],
            emotional_state=analyze(
                VisionObservation(face_available=False, face_count=0)
            ),
            local_reply="fallback",
        )

        assert "emotional_state" not in context_of(client)

    def test_cv_is_truncated_in_prompt(self) -> None:
        client = FakeClient(response_with_content(reply_json("Go on.")))
        GroqInterviewer(client=client).respond(
            candidate_input="done",
            conversation=[],
            cv_text="x" * 50_000,
            local_reply="fallback",
        )

        assert len(context_of(client)["cv_summary"]) <= 6_000

    def test_rejects_unsupported_model(self) -> None:
        with pytest.raises(ValueError, match="model must be one of"):
            GroqInterviewer(model="llama-3.1-8b-instant")


class TestServiceUsesGroqForConversation:
    def test_groq_reply_is_used_for_a_content_turn(self) -> None:
        client = FakeClient(
            response_with_content(reply_json("What metric did you track?"))
        )
        result = InterviewService(GroqInterviewer(client=client)).process_turn(
            SessionState(session_id="s1"),
            "I built a ticket classifier for support queues.",
        )

        assert result.reply == "What metric did you track?"
        assert result.reply_source == ReplySource.GROQ

    def test_conversation_accumulates_both_roles(self) -> None:
        client = FakeClient(response_with_content(reply_json("Which metric?")))
        service = InterviewService(GroqInterviewer(client=client))

        first = service.process_turn(SessionState(session_id="c1"), "I built a model.")
        second = service.process_turn(
            first.updated_state, "I used macro-F1."
        )

        conversation = second.updated_state.conversation
        roles = [entry["role"] for entry in conversation]
        assert roles == ["candidate", "interviewer", "candidate", "interviewer"]
        assert conversation[0]["content"] == "I built a model."
        assert conversation[2]["content"] == "I used macro-F1."

    def test_control_actions_never_call_groq(self) -> None:
        client = FakeClient(response_with_content(reply_json("should not appear")))
        service = InterviewService(GroqInterviewer(client=client))

        result = service.process_turn(
            SessionState(session_id="stop"),
            "I'd like to stop the practice.",
        )

        assert result.decision.action == DialogueAction.END_SESSION
        assert result.reply_source == ReplySource.LOCAL
        assert client.chat.completions.calls == []

    def test_cv_reaches_groq_through_session_state(self) -> None:
        client = FakeClient(response_with_content(reply_json("Tell me more.")))
        state = SessionState(
            session_id="cv",
            cv_text="Built a churn model. AUC 0.88.",
            target_role="Data Scientist Intern",
        )

        InterviewService(GroqInterviewer(client=client)).process_turn(
            state, "Here is my background."
        )

        context = context_of(client)
        assert context["target_role"] == "Data Scientist Intern"
        assert "AUC 0.88" in context["cv_summary"]

    def test_groq_failure_keeps_local_reply_and_reason(self) -> None:
        client = FakeClient(error=RuntimeError("down"))
        result = InterviewService(GroqInterviewer(client=client)).process_turn(
            SessionState(session_id="fallback"),
            "Could I have a hint?",
        )

        assert result.reply_source == ReplySource.LOCAL_FALLBACK
        assert result.fallback_reason == ReplyFallbackReason.SDK_ERROR
        assert result.reply
