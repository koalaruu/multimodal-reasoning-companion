"""Role-neutral coverage must not turn ML topics into universal requirements."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.reasoning.session_review import GroqReviewer
from backend.app.service import InterviewService


GENERAL_CUES = {
    "a specific work example", "how the work was checked", "a concrete outcome",
}


@pytest.mark.parametrize(
    ("role", "job", "answers"),
    [
        (
            "Flutter Developer",
            "Build Flutter screens with Provider and GoRouter, write widget tests, and improve screen load time.",
            [
                "I built a Flutter app for tracking expenses.",
                "I used widget tests and integration tests, and reduced screen load time from 800 ms to 300 ms.",
            ],
        ),
        (
            "Flutter Developer",
            "Mengembangkan aplikasi Flutter dan menguji performa tampilan.",
            [
                "Saya membuat aplikasi Flutter untuk mencatat pengeluaran.",
                "Saya melakukan pengujian widget dan integrasi; waktu muat turun dari 800 ms menjadi 300 ms.",
            ],
        ),
        (
            "Machine Learning Engineer",
            "Build demand forecasting models and measure MAE on held-out data.",
            [
                "I built a model to forecast demand.",
                "I evaluated MAE on held-out data and achieved 12% lower error.",
            ],
        ),
    ],
)
def test_complete_nonclassification_answers_have_no_ml_specific_gaps(role, job, answers):
    state = {
        "session_id": "role-review",
        "target_role": role,
        "job_description": job,
    }
    with TestClient(create_app(InterviewService(reviewer=GroqReviewer()))) as client:
        for answer in answers:
            turn = client.post("/v1/interview/turn", json={
                "state": state, "candidate_input": answer,
            })
            assert turn.status_code == 200
            state = turn.json()["updated_state"]
        response = client.post("/v1/interview/review", json={"state": state})

    assert response.status_code == 200
    review = response.json()
    assert set(review["covered"]) == GENERAL_CUES
    assert review["missing"] == []
    assert review["turns"] == 2
    assert state["cumulative_evidence"]["macro_f1_mentioned"] is False
    assert state["cumulative_evidence"]["class_imbalance_mentioned"] is False
    assert state["cumulative_evidence"]["class_level_performance_mentioned"] is False


@pytest.mark.parametrize("role", ["Flutter Developer", "Sales Associate", "ML Engineer", None])
def test_unknown_evidence_is_not_a_domain_specific_skills_verdict(role):
    with TestClient(create_app(InterviewService())) as client:
        response = client.post("/v1/interview/review", json={
            "state": {"session_id": "unknown-evidence", "target_role": role},
        })

    assert response.status_code == 200
    review = response.json()
    assert review["covered"] == []
    assert set(review["missing"]) == GENERAL_CUES
