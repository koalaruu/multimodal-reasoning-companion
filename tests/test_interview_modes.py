"""Mode validation and session round-trips without network access."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.service import InterviewService


def test_mode_survives_round_trip_and_can_switch_back() -> None:
    with TestClient(create_app(InterviewService())) as client:
        first = client.post("/v1/interview/turn", json={
            "candidate_input": "I built a ticket classifier.",
            "state": {"session_id": "mode-round-trip"},
            "interview_mode": "coach",
        }).json()
        second = client.post("/v1/interview/turn", json={
            "candidate_input": "I used macro-F1.",
            "state": first["updated_state"],
        }).json()
        third = client.post("/v1/interview/turn", json={
            "candidate_input": "It reached 0.81 on held-out tickets.",
            "state": second["updated_state"],
            "interview_mode": "strict",
        }).json()

    assert first["updated_state"]["interview_mode"] == "coach"
    assert second["updated_state"]["interview_mode"] == "coach"
    assert third["updated_state"]["interview_mode"] == "strict"
    assert third["updated_state"]["turn_index"] == 3
    history = third["updated_state"]["conversation"]
    assert history[0]["content"] == "I built a ticket classifier."
    assert history[2]["content"] == "I used macro-F1."


@pytest.mark.parametrize("endpoint", ["opening", "turn"])
def test_unknown_mode_is_rejected(endpoint: str) -> None:
    body = {"interview_mode": "unknown"}
    if endpoint == "turn":
        body.update(candidate_input="Hello", state={"session_id": "invalid-mode"})
    with TestClient(create_app(InterviewService())) as client:
        response = client.post(f"/v1/interview/{endpoint}", json=body)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "interview_mode"]


@pytest.mark.parametrize("mode", ["strict", "coach"])
def test_style_does_not_override_explicit_stop(mode: str) -> None:
    with TestClient(create_app(InterviewService())) as client:
        response = client.post("/v1/interview/turn", json={
            "candidate_input": "I'd like to stop the practice here.",
            "state": {"session_id": "stop-mode"},
            "interview_mode": mode,
        })

    assert response.status_code == 200
    assert response.json()["updated_state"]["status"] == "ended"
    assert response.json()["decision"]["action"] == "end_session"
