"""FastAPI routes that delegate all decisions to the domain service."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from backend.app.api.schemas import HealthResponse, InterviewTurnRequest
from backend.app.schemas import TurnResult
from backend.app.service import InterviewService

router = APIRouter()


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse()


@router.post(
    "/v1/interview/turn",
    response_model=TurnResult,
    tags=["interview"],
)
def process_interview_turn(
    request_body: InterviewTurnRequest,
    service: Annotated[InterviewService, Depends(get_interview_service)],
) -> TurnResult:
    return service.process_turn(
        state=request_body.state,
        candidate_input=request_body.candidate_input,
    )
