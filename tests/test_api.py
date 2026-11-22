"""Consumer-visible tests for the stateless text interview API."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.reasoning.groq_interviewer import GroqInterviewer
from backend.app.schemas import SessionState, SessionStatus
from backend.app.service import InterviewService


class FailingCompletions:
    def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("network unavailable")


class FailingClient:
    chat = type("Chat", (), {"completions": FailingCompletions()})()


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(InterviewService())) as test_client:
        yield test_client


def request_body(candidate_input: str, session_id: str = "api-test") -> dict[str, Any]:
    return {
        "candidate_input": candidate_input,
        "state": {"session_id": session_id},
    }


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("candidate_input", "action", "reason", "topic"),
    [
        (
            "Could I have a hint?",
            "offer_hint",
            "explicit_hint_request",
            "none",
        ),
        (
            "Give me a moment to think.",
            "wait",
            "explicit_thinking_time_request",
            "none",
        ),
        (
            "I built a classifier for support tickets.",
            "ask_followup",
            "evaluation_evidence_missing",
            "evaluation_method",
        ),
        (
            "I used macro-F1 because the classes were imbalanced.",
            "ask_followup",
            "class_level_performance_missing",
            "class_level_performance",
        ),
        (
            "I'd like to stop the practice here.",
            "end_session",
            "explicit_stop_request",
            "none",
        ),
        (
            "ha test",
            "request_clarification",
            "answer_irrelevant",
            "none",
        ),
    ],
)
def test_interview_turn_exposes_policy_decision(
    client: TestClient,
    candidate_input: str,
    action: str,
    reason: str,
    topic: str,
) -> None:
    response = client.post(
        "/v1/interview/turn", json=request_body(candidate_input)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == {
        "action": action,
        "reason": reason,
        "followup_topic": topic,
    }
    assert body["updated_state"]["turn_index"] == 1
    assert body["reply_source"] == "local"
    assert body["fallback_reason"] is None

def test_interview_turn_exposes_answer_quality_evidence(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/interview/turn",
        json=request_body(
            "For my capstone, I trained a model. It reached 84% on unseen examples."
        ),
    )

    assert response.status_code == 200
    evidence = response.json()["content_evidence"]
    assert evidence["input_type"] == "answer"
    assert evidence["relevance"] == "relevant"
    assert evidence["project_described"] is True
    assert evidence["evaluation_method_mentioned"] is True
    assert evidence["evaluation_result_mentioned"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"state": {"session_id": "missing-input"}},
        {"candidate_input": "answer"},
        {"candidate_input": "answer", "state": {"session_id": "   "}},
        {
            "candidate_input": "answer",
            "state": {"session_id": "extra"},
            "unexpected": True,
        },
    ],
)
def test_invalid_request_returns_422(
    client: TestClient, body: dict[str, Any]
) -> None:
    response = client.post("/v1/interview/turn", json=body)

    assert response.status_code == 422


def test_whitespace_input_uses_request_input_policy(client: TestClient) -> None:
    response = client.post(
        "/v1/interview/turn", json=request_body("  \n  ")
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["action"] == "request_input"
    assert body["decision"]["reason"] == "input_missing"


def test_ended_session_remains_terminal_over_http(client: TestClient) -> None:
    state = SessionState(
        session_id="ended-api", status=SessionStatus.ENDED
    ).model_dump(mode="json")

    response = client.post(
        "/v1/interview/turn",
        json={"candidate_input": "Could I have a hint?", "state": state},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["action"] == "end_session"
    assert body["decision"]["reason"] == "session_already_ended"
    assert body["updated_state"]["status"] == "ended"


def test_groq_failure_returns_200_with_observable_fallback() -> None:
    service = InterviewService(
        GroqInterviewer(client=FailingClient())
    )
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/interview/turn",
            json=request_body("Could I have a hint?", "fallback-api"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["action"] == "offer_hint"
    assert body["reply_source"] == "local_fallback"
    assert body["fallback_reason"] == "sdk_error"


def test_cors_allows_only_local_vite_origin(client: TestClient) -> None:
    allowed = client.options(
        "/v1/interview/turn",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    rejected = client.options(
        "/v1/interview/turn",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == (
        "http://localhost:5173"
    )
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers
