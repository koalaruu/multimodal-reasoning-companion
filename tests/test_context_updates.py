"""Context removal must reach returned state and the cloud model boundary."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.reasoning.groq_interviewer import GroqInterviewer
from backend.app.service import InterviewService


CONTEXT = {
    "cv_text": "PRIVATE_CV_CONTEXT",
    "target_role": "PRIVATE_ROLE_CONTEXT",
    "job_description": "PRIVATE_JOB_CONTEXT",
}


def seeded_state():
    return {
        "session_id": "clear-context",
        "turn_index": 2,
        **CONTEXT,
        "cv_filename": "private-cv.txt",
        "conversation": [
            {"role": "candidate", "content": "I built a ticket classifier."},
            {"role": "interviewer", "content": "Which metric did you use?"},
        ],
    }


@pytest.fixture
def client():
    with TestClient(create_app(InterviewService())) as client:
        yield client


@pytest.mark.parametrize("field", list(CONTEXT))
@pytest.mark.parametrize("cleared_value", [None, "", " \n\t "])
def test_explicit_clear_removes_only_selected_context(client, field, cleared_value):
    state = seeded_state()
    response = client.post("/v1/interview/turn", json={
        "state": state,
        "candidate_input": "I used macro-F1.",
        field: cleared_value,
    })

    assert response.status_code == 200
    updated = response.json()["updated_state"]
    assert updated[field] is None
    for other in CONTEXT.keys() - {field}:
        assert updated[other] == CONTEXT[other]
    assert updated["cv_filename"] == (None if field == "cv_text" else state["cv_filename"])
    assert updated["conversation"][:2] == state["conversation"]
    assert updated["turn_index"] == 3


def test_omitted_context_preserves_existing_values(client):
    state = seeded_state()
    response = client.post("/v1/interview/turn", json={
        "state": state,
        "candidate_input": "I used macro-F1.",
    })

    assert response.status_code == 200
    updated = response.json()["updated_state"]
    assert {field: updated[field] for field in CONTEXT} == CONTEXT
    assert updated["cv_filename"] == state["cv_filename"]


def test_replacement_context_is_retained_on_next_turn(client):
    replacement = {
        "cv_text": "New CV summary",
        "target_role": "New target role",
        "job_description": "New job requirements",
    }
    first = client.post("/v1/interview/turn", json={
        "state": seeded_state(),
        "candidate_input": "I used macro-F1.",
        **replacement,
    })
    assert first.status_code == 200
    second = client.post("/v1/interview/turn", json={
        "state": first.json()["updated_state"],
        "candidate_input": "It reached 0.81 on held-out tickets.",
    })

    assert second.status_code == 200
    updated = second.json()["updated_state"]
    assert {field: updated[field] for field in CONTEXT} == replacement


def test_removed_context_does_not_reach_groq():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"reply": "Which split did you use?"}))
        )])

    provider = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)
    ))
    service = InterviewService(GroqInterviewer(client=provider))
    with TestClient(create_app(service)) as client:
        response = client.post("/v1/interview/turn", json={
            "state": seeded_state(),
            "candidate_input": "I used macro-F1.",
            **dict.fromkeys(CONTEXT),
        })

    assert response.status_code == 200
    assert response.json()["reply_source"] == "groq"
    context = calls[0]["messages"][1]["content"]
    for private_value in CONTEXT.values():
        assert private_value not in context
    assert json.loads(context)["conversation"][0]["content"] == "I built a ticket classifier."
