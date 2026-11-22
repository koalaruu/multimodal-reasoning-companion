"""FastAPI composition root for the local multimodal interview API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.env_loader import load_dotenv
from backend.app.perception.speech_to_text import (
    AudioTranscriber,
    create_transcriber,
)
from backend.app.reasoning.llm_agent import GroqReplyGenerator
from backend.app.service import InterviewService

_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def create_app(
    interview_service: InterviewService | None = None,
    audio_transcriber: AudioTranscriber | None = None,
) -> FastAPI:
    load_dotenv()
    app = FastAPI(
        title="Multimodal Reasoning Companion API",
        version="0.1.0",
        description=(
            "Stateless interview API with deterministic dialogue policy, "
            "optional local Faster-Whisper transcription, and optional Groq "
            "reply rendering."
        ),
    )
    app.state.interview_service = (
        interview_service
        if interview_service is not None
        else InterviewService(reply_generator=GroqReplyGenerator.from_env())
    )
    app.state.audio_transcriber = (
        audio_transcriber
        if audio_transcriber is not None
        else create_transcriber()
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
