"""Regression tests for FACS-based vision emotion analysis."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.perception.vision_analyzer import (
    analyze,
    average_emotional_state,
    update_moving_average,
)
from backend.app.schemas import (
    BlendshapeReading,
    EmotionalState,
    EmotionLabel,
    VisionObservation,
)
from backend.app.service import InterviewService


def _reading(name: str, score: float) -> BlendshapeReading:
    return BlendshapeReading(name=name, score=score)


def _obs(blendshapes: list[BlendshapeReading], **kw) -> VisionObservation:
    return VisionObservation(
        face_available=True,
        face_count=1,
        blendshapes=blendshapes,
        **kw,
    )


class TestFACSMapping:
    def test_happy_smile_and_cheek_raise(self) -> None:
        obs = _obs([
            _reading("mouthSmileLeft", 0.8),
            _reading("mouthSmileRight", 0.75),
            _reading("cheekSquintLeft", 0.6),
            _reading("cheekSquintRight", 0.55),
        ])
        state = analyze(obs)
        assert state.face_available is True
        assert state.dominant_emotion == EmotionLabel.HAPPY
        assert state.confidence > 0.5
        assert "AU6" in state.activated_aus
        assert "AU12" in state.activated_aus

    def test_partial_activation_registers_proportionally(self) -> None:
        # AU12 present, AU6 absent: happy must still register, at half strength.
        obs = _obs([_reading("mouthSmileLeft", 0.8)])
        state = analyze(obs)
        assert state.emotion_scores["happy"] == pytest.approx(0.4, abs=0.01)
        assert state.dominant_emotion == EmotionLabel.HAPPY

    def test_resting_face_stays_neutral(self) -> None:
        obs = _obs([
            _reading("mouthPucker", 0.12),
            _reading("browOuterUpLeft", 0.10),
            _reading("eyeLookDownLeft", 0.11),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.NEUTRAL
        assert state.confidence == 0.0

    def test_surprised_wide_eyes_jaw_open(self) -> None:
        obs = _obs([
            _reading("browInnerUp", 0.7),
            _reading("browOuterUpLeft", 0.6),
            _reading("browOuterUpRight", 0.65),
            _reading("eyeWideLeft", 0.8),
            _reading("eyeWideRight", 0.75),
            _reading("jawOpen", 0.9),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.SURPRISED
        assert state.confidence > 0.5

    def test_angry_brow_lower_and_squint(self) -> None:
        obs = _obs([
            _reading("browDownLeft", 0.7),
            _reading("browDownRight", 0.65),
            _reading("eyeWideLeft", 0.5),
            _reading("eyeWideRight", 0.45),
            _reading("eyeSquintLeft", 0.6),
            _reading("eyeSquintRight", 0.55),
            _reading("mouthClose", 0.7),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.ANGRY

    def test_sad_inner_brow_frown(self) -> None:
        obs = _obs([
            _reading("browInnerUp", 0.5),
            _reading("browDownLeft", 0.4),
            _reading("browDownRight", 0.35),
            _reading("mouthFrownLeft", 0.6),
            _reading("mouthFrownRight", 0.55),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.SAD

    def test_neutral_when_no_activation(self) -> None:
        obs = _obs([_reading("mouthSmileLeft", 0.05)])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.NEUTRAL
        assert state.confidence == 0.0

    def test_no_face_returns_unavailable(self) -> None:
        obs = VisionObservation(face_available=False, face_count=0)
        state = analyze(obs)
        assert state.face_available is False
        assert state.dominant_emotion == EmotionLabel.NEUTRAL
        assert state.emotion_scores == {}

    def test_disgust_nose_wrinkle(self) -> None:
        obs = _obs([
            _reading("noseSneerLeft", 0.7),
            _reading("noseSneerRight", 0.65),
            _reading("mouthFrownLeft", 0.5),
            _reading("mouthFrownRight", 0.45),
            _reading("mouthLowerDownLeft", 0.4),
            _reading("mouthLowerDownRight", 0.35),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.DISGUSTED

    def test_fear_disambiguated_from_surprise_by_brow_lower(self) -> None:
        obs = _obs([
            _reading("browInnerUp", 0.7),
            _reading("browOuterUpLeft", 0.6),
            _reading("browOuterUpRight", 0.65),
            _reading("eyeWideLeft", 0.8),
            _reading("eyeWideRight", 0.75),
            _reading("jawOpen", 0.9),
            _reading("browDownLeft", 0.5),
            _reading("browDownRight", 0.45),
        ])
        state = analyze(obs)
        assert state.dominant_emotion == EmotionLabel.FEARFUL
        assert state.emotion_scores["surprised"] == 0.0

class TestMovingAverage:
    def test_window_trims_to_size(self) -> None:
        states = [
            EmotionalState(face_available=True, dominant_emotion=EmotionLabel.HAPPY, confidence=0.5),
        ]
        for _ in range(7):
            states = update_moving_average(states, EmotionalState(
                face_available=True, dominant_emotion=EmotionLabel.NEUTRAL,
            ), window_size=5)
        assert len(states) == 5
    def test_average_only_valid_samples(self) -> None:
        history = [
            EmotionalState(
                face_available=True,
                dominant_emotion=EmotionLabel.HAPPY,
                confidence=0.6,
                emotion_scores={"happy": 0.6, "sad": 0.0},
            ),
            EmotionalState(face_available=False),
            EmotionalState(
                face_available=True,
                dominant_emotion=EmotionLabel.HAPPY,
                confidence=0.8,
                emotion_scores={"happy": 0.8, "sad": 0.0},
            ),
        ]
        avg = average_emotional_state(history)
        assert avg.face_available is True
        assert avg.dominant_emotion == EmotionLabel.HAPPY
        assert 0.69 <= avg.emotion_scores["happy"] <= 0.71

    def test_empty_history_returns_unavailable(self) -> None:
        avg = average_emotional_state([])
        assert avg.face_available is False


class TestVisionEndpoint:
    def test_analyze_endpoint_happy(self) -> None:
        app = create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/vision/analyze",
            json={
                "face_available": True,
                "face_count": 1,
                "blendshapes": [
                    {"categoryName": "mouthSmileLeft", "score": 0.8},
                    {"categoryName": "mouthSmileRight", "score": 0.75},
                    {"categoryName": "cheekSquintLeft", "score": 0.6},
                    {"categoryName": "cheekSquintRight", "score": 0.55},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["emotional_state"]["dominant_emotion"] == "happy"
        assert body["emotional_state"]["face_available"] is True

    def test_analyze_endpoint_no_face(self) -> None:
        app = create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/vision/analyze",
            json={
                "face_available": False,
                "face_count": 0,
                "blendshapes": [],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["emotional_state"]["dominant_emotion"] == "neutral"
        assert body["emotional_state"]["face_available"] is False

    def test_analyze_endpoint_ignores_extra_fields(self) -> None:
        app = create_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/vision/analyze",
            json={
                "face_available": True,
                "face_count": 1,
                "blendshapes": [],
                "top_blendshapes": [{"name": "x", "score": 0.5}],
                "availability": "available",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["emotional_state"]["face_available"] is True

    def test_one_spike_and_sustained_signal_have_different_dialogue_actions(self) -> None:
        client = TestClient(create_app(InterviewService()))
        readings = [
            _reading(name, 0.88) for name in (
                "browDownLeft", "eyeWideLeft", "eyeSquintLeft", "lipPress", "mouthClose",
            )
        ]
        quiet = [
            _obs([], observed_at_ms=at).model_dump() for at in (1_000, 2_000)
        ]
        spike = _obs(readings, observed_at_ms=3_000).model_dump()
        sustained = _obs(readings, observed_at_ms=4_000).model_dump()

        for sample, history, expected_score, expect_pause in (
            (spike, quiet, 0.293, False),
            (sustained, [*quiet, spike], 0.44, True),
        ):
            vision = client.post(
                "/v1/vision/analyze",
                json={**sample, "previous_observations": history},
            )
            assert vision.status_code == 200
            reading = vision.json()
            assert reading["sample_count"] == len(history) + 1
            emotional_state = reading["emotional_state"]
            assert emotional_state["emotion_scores"]["angry"] == pytest.approx(expected_score)

            turn = client.post(
                "/v1/interview/turn",
                json={
                    "state": {"session_id": "smoothing-test"},
                    "candidate_input": "I built a Flutter app.",
                    "emotional_state": emotional_state,
                },
            )
            assert turn.status_code == 200
            assert (turn.json()["decision"]["action"] == "give_space") is expect_pause

    @pytest.mark.parametrize("face_count", [0, 2])
    def test_current_no_face_or_multiple_faces_cannot_reuse_history(self, face_count: int) -> None:
        client = TestClient(create_app(InterviewService()))
        previous = _obs([_reading("mouthSmileLeft", 0.8)], observed_at_ms=1_000)
        current = VisionObservation(
            face_available=face_count > 0, face_count=face_count, observed_at_ms=2_000,
        )
        response = client.post(
            "/v1/vision/analyze",
            json={**current.model_dump(), "previous_observations": [previous.model_dump()]},
        )
        assert response.status_code == 200
        assert response.json()["sample_count"] == 0
        assert response.json()["emotional_state"]["face_available"] is False

    def test_window_excludes_previous_face_after_capture_gap(self) -> None:
        client = TestClient(create_app(InterviewService()))
        previous = _obs([_reading("mouthSmileLeft", 0.8)], observed_at_ms=1_000)
        current = _obs([], observed_at_ms=6_000)
        response = client.post(
            "/v1/vision/analyze",
            json={**current.model_dump(), "previous_observations": [previous.model_dump()]},
        )
        assert response.status_code == 200
        assert response.json()["sample_count"] == 1
        assert response.json()["emotional_state"]["emotion_scores"]["happy"] == 0
