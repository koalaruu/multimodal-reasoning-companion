"""Tests for the post-session review."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.reasoning.session_review import (
    GroqReviewer,
    compute_coverage,
)
from backend.app.schemas import (
    AnswerRelevance,
    ContentEvidence,
    InputType,
    ReplyFallbackReason,
    ReplySource,
    SessionState,
)
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


def evidence(**flags: bool) -> ContentEvidence:
    return ContentEvidence(
        input_type=InputType.ANSWER,
        relevance=AnswerRelevance.RELEVANT,
        **flags,
    )


class TestCoverageIsDeterministic:
    def test_full_evidence_covers_everything(self) -> None:
        covered, missing = compute_coverage(
            evidence(
                project_described=True,
                evaluation_method_mentioned=True,
                evaluation_result_mentioned=True,
                macro_f1_mentioned=True,
                class_imbalance_mentioned=True,
                class_level_performance_mentioned=True,
            )
        )
        assert len(covered) == 6
        assert missing == []

    def test_empty_evidence_covers_nothing(self) -> None:
        covered, missing = compute_coverage(evidence())
        assert covered == []
        assert len(missing) == 6

    def test_no_evidence_is_all_missing(self) -> None:
        covered, missing = compute_coverage(None)
        assert covered == []
        assert len(missing) == 6

    def test_partial_evidence_splits_correctly(self) -> None:
        covered, missing = compute_coverage(
            evidence(project_described=True, evaluation_method_mentioned=True)
        )
        assert "a specific project" in covered
        assert "the evaluation method" in covered
        assert "a concrete result or metric value" in missing


class TestGroqReviewer:
    def test_model_advice_is_returned(self) -> None:
        payload = json.dumps({
            "suggestions": ["Quantify the result.", "Name the threshold rule."],
            "questions_to_ask": ["How is the team structured?"],
        })
        client = FakeClient(response_with_content(payload))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r1", turn_index=4),
            covered=["a specific project"],
            missing=["the evaluation method"],
        )

        assert review.review_source == ReplySource.GROQ
        assert review.suggestions == [
            "Quantify the result.",
            "Name the threshold rule.",
        ]
        assert review.questions_to_ask == ["How is the team structured?"]
        assert review.covered == ["a specific project"]
        assert review.turns == 4

    def test_without_client_uses_local_advice(self) -> None:
        review = GroqReviewer().review(
            state=SessionState(session_id="r2", turn_index=2),
            covered=[],
            missing=["the evaluation method"],
        )

        assert review.review_source == ReplySource.LOCAL
        assert review.suggestions
        assert "which metric you used" in review.suggestions[0]

    def test_timeout_falls_back_with_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeClient(error=TimeoutError("slow"))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r3"),
            covered=[],
            missing=["a specific project"],
        )

        assert review.review_source == ReplySource.LOCAL_FALLBACK
        assert review.fallback_reason == ReplyFallbackReason.TIMEOUT
        assert "reason=timeout" in caplog.text

    def test_malformed_output_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeClient(response_with_content("not json"))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r4"),
            covered=[],
            missing=[],
        )

        assert review.fallback_reason == ReplyFallbackReason.MALFORMED_OUTPUT
        assert "reason=malformed_output" in caplog.text

    def test_items_are_trimmed_and_capped(self) -> None:
        payload = json.dumps({
            "suggestions": ["  one  ", "two", "three", "four", "five"],
            "questions_to_ask": ["a", "b", "c", "d"],
        })
        client = FakeClient(response_with_content(payload))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r5"),
            covered=[],
            missing=[],
        )

        assert len(review.suggestions) == 3
        assert review.suggestions[0] == "one"
        assert len(review.questions_to_ask) == 3

    def test_jd_and_cv_reach_the_review_prompt(self) -> None:
        payload = json.dumps({"suggestions": [], "questions_to_ask": []})
        client = FakeClient(response_with_content(payload))
        state = SessionState(
            session_id="r6",
            cv_text="Built a churn model.",
            job_description="Requires deployment experience.",
            target_role="ML Engineer Intern",
        )

        GroqReviewer(client=client).review(
            state=state, covered=[], missing=[]
        )

        context = json.loads(
            client.chat.completions.calls[0]["messages"][1]["content"]
        )
        assert context["target_role"] == "ML Engineer Intern"
        assert "deployment experience" in context["job_description"]
        assert "churn model" in context["cv_summary"]

    def test_sdk_failure_keeps_private_payload_out_of_logs_and_response(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        private_text = "PRIVATE_CV_AND_KEY_SENTINEL"
        error = groq.BadRequestError(
            f"Provider echoed {private_text}",
            response=httpx.Response(
                400,
                request=httpx.Request(
                    "POST", "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {private_text}"},
                ),
            ),
            body={"error": {"failed_generation": private_text}},
        )
        reviewer = GroqReviewer(client=FakeClient(error=error))
        state = SessionState(
            session_id="private-review",
            turn_index=2,
            cv_text=private_text,
            conversation=[{"role": "candidate", "content": private_text}],
        )
        with TestClient(create_app(InterviewService(reviewer=reviewer))) as client:
            response = client.post(
                "/v1/interview/review",
                json={"state": state.model_dump(mode="json")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["review_source"] == "local_fallback"
        assert body["fallback_reason"] == "sdk_error"
        assert "a specific project" in body["missing"]
        assert body["covered"] == []
        assert body["questions_to_ask"] == []
        assert "error_type=BadRequestError status=400" in caplog.text
        assert private_text not in caplog.text
        assert private_text not in response.text


class TestReviewEndpoint:
    def test_review_endpoint_returns_coverage(self) -> None:
        app = create_app(InterviewService())
        client = TestClient(app)
        state = SessionState(session_id="ep", turn_index=3).model_dump(mode="json")

        response = client.post(
            "/v1/interview/review", json={"state": state}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["turns"] == 3
        assert isinstance(body["covered"], list)
        assert isinstance(body["missing"], list)
        assert "review_source" in body

    def test_review_endpoint_rejects_missing_state(self) -> None:
        app = create_app(InterviewService())
        client = TestClient(app)

        response = client.post("/v1/interview/review", json={})

        assert response.status_code == 422
