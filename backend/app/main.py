"""FastAPI composition root for the text interview API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.reasoning.llm_agent import GeminiReplyGenerator
from backend.app.service import InterviewService

_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def create_app(interview_service: InterviewService | None = None) -> FastAPI:
    app = FastAPI(
        title="Multimodal Reasoning Companion API",
        version="0.1.0",
        description=(
            "Stateless text interview API with deterministic dialogue policy "
            "and optional Gemini reply rendering."
        ),
    )
    app.state.interview_service = (
        interview_service
        if interview_service is not None
        else InterviewService(
            reply_generator=GeminiReplyGenerator.from_env()
        )
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
