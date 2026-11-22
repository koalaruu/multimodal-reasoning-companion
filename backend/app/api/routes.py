"""FastAPI routes that delegate all decisions to the domain service."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.api.schemas import (
    CvUploadResponse,
    HealthResponse,
    InterviewOpeningRequest,
    InterviewOpeningResponse,
    InterviewReviewRequest,
    InterviewTurnRequest,
    TranscriptionResponse,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
)
from backend.app.perception.cv_reader import CvExtractionError, extract_cv
from backend.app.perception.speech_to_text import (
    AudioTooLongError,
    AudioTranscriber,
    InvalidAudioError,
    NoSpeechDetectedError,
    TranscriberUnavailableError,
)
from backend.app.schemas import (
    SessionReview,
    TurnResult,
    VisionObservation,
)
from backend.app.perception.vision_analyzer import analyze as analyze_vision
from backend.app.service import InterviewService

router = APIRouter()

_MAX_AUDIO_BYTES = 10 * 1024 * 1024
_SUPPORTED_AUDIO_TYPES = frozenset(
    {
        "audio/flac",
        "audio/mp4",
        "audio/ogg",
        "audio/wav",
        "audio/webm",
        "audio/x-wav",
    }
)


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service


def get_audio_transcriber(request: Request) -> AudioTranscriber:
    return request.app.state.audio_transcriber


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse()


def _audio_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    tags=["audio"],
)
async def transcribe_audio(
    request: Request,
    transcriber: Annotated[AudioTranscriber, Depends(get_audio_transcriber)],
) -> TranscriptionResponse:
    media_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if media_type not in _SUPPORTED_AUDIO_TYPES:
        raise _audio_error(
            415,
            "unsupported_audio_type",
            "Use WebM, Ogg, MP4, FLAC, or WAV audio.",
        )

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as error:
            raise _audio_error(
                400,
                "invalid_content_length",
                "Content-Length must be an integer.",
            ) from error
        if declared_size > _MAX_AUDIO_BYTES:
            raise _audio_error(
                413,
                "audio_too_large",
                "Audio must not exceed 10 MiB.",
            )

    with BytesIO() as audio:
        received_size = 0
        async for chunk in request.stream():
            received_size += len(chunk)
            if received_size > _MAX_AUDIO_BYTES:
                raise _audio_error(
                    413,
                    "audio_too_large",
                    "Audio must not exceed 10 MiB.",
                )
            audio.write(chunk)

        if received_size == 0:
            raise _audio_error(
                422,
                "empty_audio",
                "Audio body must not be empty.",
            )
        audio.seek(0)
        try:
            result = await run_in_threadpool(transcriber.transcribe, audio)
        except AudioTooLongError as error:
            raise _audio_error(
                413,
                "audio_too_long",
                "Audio must not exceed the configured duration limit.",
            ) from error
        except InvalidAudioError as error:
            raise _audio_error(
                422,
                "invalid_audio",
                "Audio could not be decoded.",
            ) from error
        except NoSpeechDetectedError as error:
            raise _audio_error(
                422,
                "no_speech_detected",
                "No speech was detected in the recording.",
            ) from error
        except TranscriberUnavailableError as error:
            raise _audio_error(
                503,
                "transcriber_unavailable",
                "Local transcription is unavailable.",
            ) from error

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        audio_duration_ms=result.audio_duration_ms,
        processing_duration_ms=result.processing_duration_ms,
        model=result.model,
    )


@router.post(
    "/v1/interview/turn",
    response_model=TurnResult,
    tags=["interview"],
)
def process_interview_turn(
    request_body: InterviewTurnRequest,
    service: Annotated[InterviewService, Depends(get_interview_service)],
) -> TurnResult:
    state = request_body.state
    updates: dict[str, object] = {}
    for field in ("cv_text", "target_role", "job_description"):
        if field in request_body.model_fields_set:
            value = getattr(request_body, field)
            updates[field] = (value.strip() if value is not None else None) or None
    if "cv_text" in updates and updates["cv_text"] is None:
        updates["cv_filename"] = None
    if request_body.interview_mode is not None:
        updates["interview_mode"] = request_body.interview_mode

    # Facial evidence follows the same convention as context: an omitted field
    # preserves the session value, an explicit null clears it. The client sends
    # null when the camera is off, the reading is stale, or analysis failed.
    emotional_state = (
        request_body.emotional_state
        if "emotional_state" in request_body.model_fields_set
        else state.emotional_state
    )
    if updates:
        state = state.model_copy(update=updates)

    return service.process_turn(
        state=state,
        candidate_input=request_body.candidate_input,
        emotional_state=emotional_state,
        turn_readiness=request_body.turn_readiness,
        trailing_silence_ms=request_body.trailing_silence_ms,
    )


@router.post(
    "/v1/interview/opening",
    response_model=InterviewOpeningResponse,
    tags=["interview"],
)
def interview_opening(
    request_body: InterviewOpeningRequest,
    service: Annotated[InterviewService, Depends(get_interview_service)],
) -> InterviewOpeningResponse:
    """First question for this candidate, grounded in CV and posting."""
    reply, source = service.opening_question(
        cv_text=request_body.cv_text,
        target_role=request_body.target_role,
        job_description=request_body.job_description,
        interview_mode=request_body.interview_mode,
    )
    return InterviewOpeningResponse(reply=reply, reply_source=source)


@router.post(
    "/v1/interview/review",
    response_model=SessionReview,
    tags=["interview"],
)
def review_interview_session(
    request_body: InterviewReviewRequest,
    service: Annotated[InterviewService, Depends(get_interview_service)],
) -> SessionReview:
    """Summarize a finished practice session.

    covered and missing are derived from accumulated evidence and are
    deterministic. suggestions and questions_to_ask are model-written.
    """
    return service.review_session(request_body.state)


@router.post(
    "/v1/cv/extract",
    response_model=CvUploadResponse,
    tags=["cv"],
)
async def extract_cv_endpoint(
    file: Annotated[UploadFile, File()],
) -> CvUploadResponse:
    """Extract plain text from an uploaded CV or job summary.

    The text is returned to the browser, which stores it in the session and
    sends it with later turns. Nothing is persisted on the server.
    """
    data = await file.read()
    try:
        document = extract_cv(file.filename or "cv", data)
    except CvExtractionError as error:
        raise HTTPException(
            status_code=422,
            detail={"code": "cv_extraction_failed", "message": str(error)},
        ) from error

    preview = document.text[:300].replace("\n", " ")
    return CvUploadResponse(
        filename=document.filename,
        characters=document.characters,
        text=document.text,
        preview=preview,
    )


@router.post(
    "/v1/vision/analyze",
    response_model=VisionAnalyzeResponse,
    tags=["vision"],
)
async def analyze_vision_endpoint(
    request_body: VisionAnalyzeRequest,
) -> VisionAnalyzeResponse:
    """Map blendshape readings to a proxy emotional state via FACS rules.

    This endpoint is stateless and does not store frames or blendshapes.
    See README for scientific limitations of FACS-based emotion inference.
    """
    from backend.app.schemas import BlendshapeReading

    readings = [
        BlendshapeReading(
            name=str(item.get("categoryName", item.get("name", ""))),
            score=float(item.get("score", 0.0)),
        )
        for item in request_body.blendshapes
        if isinstance(item, dict)
    ]
    observation = VisionObservation(
        face_available=request_body.face_available,
        face_count=request_body.face_count,
        inference_ms=request_body.inference_ms,
        observed_at_ms=request_body.observed_at_ms,
        blendshapes=readings,
    )
    emotional_state = analyze_vision(observation)
    return VisionAnalyzeResponse(emotional_state=emotional_state)
