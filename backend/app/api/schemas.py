"""HTTP request and health response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from backend.app.schemas import DomainModel, EmotionalState, SessionState


class HealthResponse(DomainModel):
    status: Literal["ok"] = "ok"


class InterviewTurnRequest(DomainModel):
    candidate_input: str
    state: SessionState


class TranscriptionResponse(DomainModel):
    text: str = Field(min_length=1, max_length=10_000)
    language: str = Field(min_length=1)
    language_probability: float = Field(ge=0.0, le=1.0)
    audio_duration_ms: int = Field(ge=0)
    processing_duration_ms: int = Field(ge=0)
    model: str = Field(min_length=1)


class VisionAnalyzeRequest(DomainModel):
    """Blendshape payload from the browser MediaPipe Face Landmarker."""

    face_available: bool
    face_count: int = Field(default=0, ge=0)
    inference_ms: float | None = None
    observed_at_ms: int | None = None
    blendshapes: list[dict[str, object]] = Field(default_factory=list)


class VisionAnalyzeResponse(DomainModel):
    """Emotional state derived from blendshape FACS mapping."""

    emotional_state: EmotionalState
