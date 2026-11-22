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
        assert set(covered) == {
            "a specific work example", "how the work was checked", "a concrete outcome",
        }
        assert missing == []

    def test_empty_evidence_covers_nothing(self) -> None:
        covered, missing = compute_coverage(evidence())
        assert covered == []
        assert set(missing) == {
            "a specific work example", "how the work was checked", "a concrete outcome",
        }

    def test_no_evidence_is_all_missing(self) -> None:
        covered, missing = compute_coverage(None)
        assert covered == []
        assert set(missing) == {
            "a specific work example", "how the work was checked", "a concrete outcome",
        }

    def test_partial_evidence_splits_correctly(self) -> None:
        covered, missing = compute_coverage(
            evidence(project_described=True, evaluation_method_mentioned=True)
        )
        assert set(covered) == {"a specific work example", "how the work was checked"}
        assert missing == ["a concrete outcome"]


class TestGroqReviewer:
    def test_model_advice_is_returned(self) -> None:
        payload = json.dumps({
            "suggestions": ["Quantify the result.", "Name the threshold rule."],
            "questions_to_ask": ["How is the team structured?"],
        })
        client = FakeClient(response_with_content(payload))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r1", turn_index=4),
            covered=["a specific work example"],
            missing=["how the work was checked"],
        )

        assert review.review_source == ReplySource.GROQ
        assert review.suggestions == [
            "Quantify the result.",
            "Name the threshold rule.",
        ]
        assert review.questions_to_ask == ["How is the team structured?"]
        assert review.covered == ["a specific work example"]
        assert review.turns == 4

    def test_without_client_uses_local_advice(self) -> None:
        review = GroqReviewer().review(
            state=SessionState(session_id="r2", turn_index=2),
            covered=[],
            missing=["how the work was checked"],
        )

        assert review.review_source == ReplySource.LOCAL
        assert review.suggestions

    def test_timeout_falls_back_with_reason(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeClient(error=TimeoutError("slow"))
        review = GroqReviewer(client=client).review(
            state=SessionState(session_id="r3"),
            covered=[],
            missing=["a specific work example"],
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
            conversation=[
                {"role": "candidate", "content": "Saya membangun model churn."},
            ],
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
        assert context["conversation"] == state.conversation

    def test_long_context_fits_request_budget_and_keeps_recent_turns(self) -> None:
        # A character cap alone misses multi-byte text and JSON escaping.
        long_text = '"\\\x01📱wawancara ' * 1_000
        latest_answer = {"role": "candidate", "content": "Saya menguji alur offline."}
        latest_question = {"role": "interviewer", "content": "Apa hasilnya?"}
        state = SessionState(
            session_id="long-review",
            turn_index=12,
            target_role="Flutter Developer " + long_text,
            job_description="Build reliable offline flows. " + long_text,
            cv_text="Built a Flutter app. " + long_text,
            conversation=[
                {"role": "candidate", "content": f"Answer {i}: {long_text}"}
                for i in range(11)
            ] + [latest_answer, latest_question],
        )
        client = FakeClient(response_with_content(json.dumps({
            "suggestions": ["Quantify the offline test result."],
            "questions_to_ask": [],
        })))

        review = GroqReviewer(client=client).review(
            state=state, covered=["a specific work example"], missing=["a concrete outcome"],
        )

        assert review.review_source == ReplySource.GROQ
        assert review.turns == 12
        assert review.covered == ["a specific work example"]
        assert review.missing == ["a concrete outcome"]
        prompt = client.chat.completions.calls[0]["messages"][1]["content"]
        assert len(prompt.encode("utf-8")) <= 6_600
        context = json.loads(prompt)
        assert context["target_role"].startswith("Flutter Developer ")
        assert context["job_description"].startswith("Build reliable offline flows. ")
        assert context["cv_summary"].startswith("Built a Flutter app. ")
        assert context["conversation"][-2:] == [latest_answer, latest_question]
        assert context["conversation"][-3]["content"].startswith("Answer 10: ")
        assert context["conversation"][-3]["content"].endswith("…")
        assert all(not message["content"].startswith("Answer 0: ") for message in context["conversation"])
        assert state.conversation[-3]["content"] == f"Answer 10: {long_text}"

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
        assert "a specific work example" in body["missing"]
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
