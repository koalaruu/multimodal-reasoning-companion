"""HTTP request and health response contracts."""

from __future__ import annotations

from typing import Literal

from backend.app.schemas import DomainModel, SessionState


class HealthResponse(DomainModel):
    status: Literal["ok"] = "ok"


class InterviewTurnRequest(DomainModel):
    candidate_input: str
    state: SessionState
