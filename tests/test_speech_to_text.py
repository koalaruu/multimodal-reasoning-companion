"""Local transcription adapter and HTTP boundary tests."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.perception.speech_to_text import (
    AudioTooLongError,
    FasterWhisperTranscriber,
    NoSpeechDetectedError,
    TranscriberUnavailableError,
    TranscriptionResult,
)
from backend.app.service import InterviewService


class FakeModel:
    def __init__(
        self,
        segment_texts: list[str],
        *,
        language: str = "en",
        language_probability: float = 0.93,
    ) -> None:
        self.segment_texts = segment_texts
        self.language = language
        self.language_probability = language_probability
        self.calls: list[tuple[object, dict]] = []

    def transcribe(self, samples: object, **kwargs: object):
        self.calls.append((samples, kwargs))
        segments = (SimpleNamespace(text=text) for text in self.segment_texts)
        info = SimpleNamespace(
            language=self.language,
            language_probability=self.language_probability,
        )
        return segments, info


class CapturingTranscriber:
    def __init__(self) -> None:
        self.payload: bytes | None = None

    def transcribe(self, audio: BinaryIO) -> TranscriptionResult:
        self.payload = audio.read()
        return TranscriptionResult(
            text="I built a classifier.",
            language="en",
            language_probability=0.95,
            audio_duration_ms=1_200,
            processing_duration_ms=320,
            model="fake-base.en",
        )


class UnavailableTranscriber:
    def transcribe(self, audio: BinaryIO) -> TranscriptionResult:
        raise TranscriberUnavailableError("private runtime details")


def test_faster_whisper_adapter_consumes_segments_and_reuses_model() -> None:
    model = FakeModel([" I built a classifier. ", " It reached 84%. "])
    factory_calls: list[tuple[str, dict]] = []

    def model_factory(model_name: str, **kwargs: object) -> FakeModel:
        factory_calls.append((model_name, kwargs))
        return model

    transcriber = FasterWhisperTranscriber(
        model_factory=model_factory,
        audio_decoder=lambda audio, sampling_rate: range(sampling_rate * 2),
    )

    first = transcriber.transcribe(BytesIO(b"first"))
    second = transcriber.transcribe(BytesIO(b"second"))

    assert first.text == "I built a classifier. It reached 84%."
    assert first.language == "en"
    assert first.language_probability == 0.93
    assert first.audio_duration_ms == 2_000
    assert first.processing_duration_ms >= 0
    assert second.text == first.text
    assert factory_calls == [
        ("base", {"device": "cpu", "compute_type": "int8"})
    ]
    assert len(model.calls) == 2
    assert model.calls[0][1] == {
        "language": None,
        "beam_size": 1,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


def test_auto_detection_returns_indonesian_language_metadata() -> None:
    model = FakeModel(
        [" Saya membangun model klasifikasi. "],
        language="id",
        language_probability=0.91,
    )
    transcriber = FasterWhisperTranscriber(
        model_factory=lambda *args, **kwargs: model,
        audio_decoder=lambda audio, sampling_rate: range(sampling_rate),
    )

    result = transcriber.transcribe(BytesIO(b"indonesian"))

    assert result.text == "Saya membangun model klasifikasi."
    assert result.language == "id"
    assert result.language_probability == 0.91
    assert model.calls[0][1]["language"] is None


@pytest.mark.parametrize("language", ["english", "fr"])
def test_unsupported_language_configuration_is_rejected(language: str) -> None:
    with pytest.raises(ValueError, match="auto, en, or id"):
        FasterWhisperTranscriber(language=language)


def test_english_only_model_requires_pinned_english() -> None:
    with pytest.raises(ValueError, match="require language=en"):
        FasterWhisperTranscriber(model_name="base.en")

    FasterWhisperTranscriber(model_name="base.en", language="en")


def test_duration_limit_is_checked_before_model_loading() -> None:
    model_loaded = False

    def model_factory(model_name: str, **kwargs: object) -> FakeModel:
        nonlocal model_loaded
        model_loaded = True
        return FakeModel([])

    transcriber = FasterWhisperTranscriber(
        max_audio_seconds=1,
        model_factory=model_factory,
        audio_decoder=lambda audio, sampling_rate: range(sampling_rate + 1),
    )

    with pytest.raises(AudioTooLongError):
        transcriber.transcribe(BytesIO(b"long"))
    assert model_loaded is False


def test_empty_transcription_is_reported_as_no_speech() -> None:
    transcriber = FasterWhisperTranscriber(
        model_factory=lambda *args, **kwargs: FakeModel(["   "]),
        audio_decoder=lambda audio, sampling_rate: range(sampling_rate),
    )

    with pytest.raises(NoSpeechDetectedError):
        transcriber.transcribe(BytesIO(b"silence"))


def test_audio_endpoint_streams_bytes_to_injected_transcriber() -> None:
    transcriber = CapturingTranscriber()
    with TestClient(
        create_app(InterviewService(), audio_transcriber=transcriber)
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            content=b"fake-webm-audio",
            headers={"Content-Type": "audio/webm;codecs=opus"},
        )

    assert response.status_code == 200
    assert transcriber.payload == b"fake-webm-audio"
    assert response.json() == {
        "text": "I built a classifier.",
        "language": "en",
        "language_probability": 0.95,
        "audio_duration_ms": 1_200,
        "processing_duration_ms": 320,
        "model": "fake-base.en",
    }


@pytest.mark.parametrize(
    ("headers", "content", "status_code", "error_code"),
    [
        (
            {"Content-Type": "text/plain"},
            b"not-audio",
            415,
            "unsupported_audio_type",
        ),
        (
            {"Content-Type": "audio/webm"},
            b"",
            422,
            "empty_audio",
        ),
        (
            {
                "Content-Type": "audio/webm",
                "Content-Length": str(10 * 1024 * 1024 + 1),
            },
            b"x",
            413,
            "audio_too_large",
        ),
    ],
)
def test_audio_endpoint_rejects_invalid_transport_input(
    headers: dict[str, str],
    content: bytes,
    status_code: int,
    error_code: str,
) -> None:
    with TestClient(
        create_app(InterviewService(), audio_transcriber=CapturingTranscriber())
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            content=content,
            headers=headers,
        )

    assert response.status_code == status_code
    assert response.json()["detail"]["code"] == error_code


def test_audio_endpoint_hides_transcriber_failure_details() -> None:
    with TestClient(
        create_app(InterviewService(), audio_transcriber=UnavailableTranscriber())
    ) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            content=b"audio",
            headers={"Content-Type": "audio/webm"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "transcriber_unavailable",
        "message": "Local transcription is unavailable.",
    }
    assert "private runtime details" not in response.text
