"""Speech-to-text adapters: local Faster-Whisper and cloud Groq Whisper."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, BinaryIO, Protocol, Self

_SAMPLE_RATE = 16_000
_SUPPORTED_LANGUAGES = frozenset({"en", "id"})


def _normalize_language(language: str | None) -> str | None:
    normalized = language.strip().lower() if language else ""
    if normalized in {"", "auto"}:
        return None
    if normalized not in _SUPPORTED_LANGUAGES:
        raise ValueError("language must be auto, en, or id")
    return normalized


class TranscriptionError(RuntimeError):
    """Base error safe for transport-level mapping."""


class TranscriberUnavailableError(TranscriptionError):
    """The optional runtime or model is unavailable."""


class InvalidAudioError(TranscriptionError):
    """The supplied bytes cannot be decoded as audio."""


class AudioTooLongError(TranscriptionError):
    """Decoded audio exceeds the configured duration limit."""


class NoSpeechDetectedError(TranscriptionError):
    """No transcribable speech was found."""


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_duration_ms: int
    processing_duration_ms: int
    model: str


class AudioTranscriber(Protocol):
    def transcribe(self, audio: BinaryIO) -> TranscriptionResult:
        """Transcribe one seekable in-memory audio stream."""


class FasterWhisperTranscriber:
    """CPU-first transcriber that downloads and loads its model on first use."""

    def __init__(
        self,
        *,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        max_audio_seconds: float = 60.0,
        model_factory: Callable[..., Any] | None = None,
        audio_decoder: Callable[..., Any] | None = None,
    ) -> None:
        normalized_model = model_name.strip()
        if not normalized_model:
            raise ValueError("model_name must not be blank")
        if not device.strip():
            raise ValueError("device must not be blank")
        if not compute_type.strip():
            raise ValueError("compute_type must not be blank")
        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")
        if (model_factory is None) != (audio_decoder is None):
            raise ValueError(
                "model_factory and audio_decoder must be supplied together"
            )
        normalized_language = _normalize_language(language)
        if normalized_model.endswith(".en") and normalized_language != "en":
            raise ValueError(
                "English-only Whisper models require language=en"
            )

        self._model_name = normalized_model
        self._device = device.strip()
        self._compute_type = compute_type.strip()
        self._language = normalized_language
        self._max_audio_seconds = max_audio_seconds
        self._model_factory = model_factory
        self._audio_decoder = audio_decoder
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Self:
        source = os.environ if env is None else env
        raw_max_seconds = source.get("WHISPER_MAX_AUDIO_SECONDS", "60").strip()
        try:
            max_audio_seconds = float(raw_max_seconds)
        except ValueError as error:
            raise ValueError(
                "WHISPER_MAX_AUDIO_SECONDS must be a number"
            ) from error

        return cls(
            model_name=source.get("WHISPER_MODEL", "small"),
            device=source.get("WHISPER_DEVICE", "cpu"),
            compute_type=source.get("WHISPER_COMPUTE_TYPE", "int8"),
            language=source.get("WHISPER_LANGUAGE", "auto"),
            max_audio_seconds=max_audio_seconds,
        )

    def _load_runtime(self) -> None:
        if self._model_factory is not None:
            return
        try:
            from faster_whisper import WhisperModel
            from faster_whisper.audio import decode_audio
        except (ImportError, OSError) as error:
            raise TranscriberUnavailableError(
                "local transcription runtime is unavailable"
            ) from error
        self._model_factory = WhisperModel
        self._audio_decoder = decode_audio

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            self._load_runtime()
            assert self._model_factory is not None
            try:
                self._model = self._model_factory(
                    self._model_name,
                    device=self._device,
                    compute_type=self._compute_type,
                )
            except Exception as error:
                raise TranscriberUnavailableError(
                    "local transcription model is unavailable"
                ) from error
            return self._model

    def _decode(self, audio: BinaryIO) -> Any:
        self._load_runtime()
        assert self._audio_decoder is not None
        try:
            return self._audio_decoder(audio, sampling_rate=_SAMPLE_RATE)
        except Exception as error:
            raise InvalidAudioError("audio could not be decoded") from error

    def transcribe(self, audio: BinaryIO) -> TranscriptionResult:
        started_at = perf_counter()
        samples = self._decode(audio)
        audio_duration_seconds = len(samples) / _SAMPLE_RATE
        if audio_duration_seconds > self._max_audio_seconds:
            raise AudioTooLongError("audio duration exceeds the configured limit")
        if audio_duration_seconds <= 0:
            raise NoSpeechDetectedError("audio contains no samples")

        model = self._get_model()
        try:
            with self._inference_lock:
                segments, info = model.transcribe(
                    samples,
                    language=self._language,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(_segment_texts(segments)).strip()
        except Exception as error:
            raise TranscriberUnavailableError(
                "local transcription inference failed"
            ) from error

        if not text:
            raise NoSpeechDetectedError("no speech was detected")

        probability = float(getattr(info, "language_probability", 0.0))
        probability = min(1.0, max(0.0, probability))
        language = str(getattr(info, "language", None) or self._language or "unknown")
        return TranscriptionResult(
            text=text,
            language=language,
            language_probability=probability,
            audio_duration_ms=round(audio_duration_seconds * 1_000),
            processing_duration_ms=round((perf_counter() - started_at) * 1_000),
            model=self._model_name,
        )


class GroqWhisperTranscriber:
    """Cloud speech-to-text via Groq Whisper API.

    Audio bytes are uploaded to Groq. Requires GROQ_API_KEY.
    Falls back to local provider when no key is configured.
    """

    _SUPPORTED_MODELS = frozenset({
        "whisper-large-v3",
        "whisper-large-v3-turbo",
    })
    _MAX_FILE_BYTES = 25 * 1024 * 1024  # Groq free tier limit

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq STT requires a non-empty GROQ_API_KEY")
        normalized_model = model.strip()
        if normalized_model not in self._SUPPORTED_MODELS:
            supported = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"model must be one of: {supported}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        normalized_language = _normalize_language(language)

        self._api_key = api_key.strip()
        self._model = normalized_model
        self._language = normalized_language
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Self:
        source = os.environ if env is None else env
        api_key = source.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Groq STT provider requires GROQ_API_KEY; "
                "set STT_PROVIDER=local to use Faster-Whisper instead"
            )
        model = source.get("GROQ_STT_MODEL", "whisper-large-v3-turbo").strip()
        language = source.get("GROQ_STT_LANGUAGE", source.get("WHISPER_LANGUAGE", "auto"))
        raw_timeout = source.get("GROQ_STT_TIMEOUT_SECONDS", "30").strip()
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as error:
            raise ValueError("GROQ_STT_TIMEOUT_SECONDS must be a number") from error
        return cls(
            api_key=api_key,
            model=model,
            language=language,
            timeout_seconds=timeout_seconds,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
        except ImportError as error:
            raise TranscriberUnavailableError(
                "groq SDK is not installed"
            ) from error
        self._client = Groq(api_key=self._api_key, timeout=self._timeout, max_retries=0)
        return self._client

    def transcribe(self, audio: BinaryIO) -> TranscriptionResult:
        started_at = perf_counter()
        audio_bytes = audio.read()
        if not audio_bytes:
            raise NoSpeechDetectedError("audio body is empty")
        if len(audio_bytes) > self._MAX_FILE_BYTES:
            raise AudioTooLongError(
                f"audio exceeds Groq {self._MAX_FILE_BYTES // (1024*1024)} MiB limit"
            )

        client = self._get_client()
        try:
            kwargs: dict[str, Any] = {
                "file": ("recording.webm", audio_bytes),
                "model": self._model,
                "response_format": "verbose_json",
                "temperature": 0.0,
            }
            if self._language is not None:
                kwargs["language"] = self._language
            response = client.audio.transcriptions.create(**kwargs)
        except Exception as error:
            msg = str(error).lower()
            if "timeout" in msg or "timed out" in msg:
                raise TranscriberUnavailableError("Groq STT request timed out") from error
            raise TranscriberUnavailableError(
                "Groq STT request failed"
            ) from error

        text = str(getattr(response, "text", "")).strip()
        if not text:
            raise NoSpeechDetectedError("no speech was detected by Groq")

        language = str(getattr(response, "language", None) or self._language or "unknown")
        probability = float(getattr(response, "language_probability", 0.0))
        probability = min(1.0, max(0.0, probability))
        duration = float(getattr(response, "duration", 0.0))

        return TranscriptionResult(
            text=text,
            language=language,
            language_probability=probability,
            audio_duration_ms=round(duration * 1_000),
            processing_duration_ms=round((perf_counter() - started_at) * 1_000),
            model=f"groq/{self._model}",
        )


def create_transcriber(env: dict[str, str] | None = None) -> AudioTranscriber:
    """Create the configured STT provider from environment.

    STT_PROVIDER=local (default) -> FasterWhisperTranscriber
    STT_PROVIDER=groq            -> GroqWhisperTranscriber (requires GROQ_API_KEY)

    If STT_PROVIDER=groq but no API key is set, falls back to local
    with a warning rather than crashing at startup.
    """
    source = os.environ if env is None else env
    provider = source.get("STT_PROVIDER", "local").strip().lower()

    if provider == "groq":
        api_key = source.get("GROQ_API_KEY", "").strip()
        if not api_key:
            import warnings
            warnings.warn(
                "STT_PROVIDER=groq but GROQ_API_KEY is empty; "
                "falling back to local Faster-Whisper",
                RuntimeWarning,
                stacklevel=2,
            )
            return FasterWhisperTranscriber.from_env(source)
        return GroqWhisperTranscriber.from_env(source)

    return FasterWhisperTranscriber.from_env(source)


def _segment_texts(segments: Iterable[Any]) -> Iterable[str]:
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if text:
            yield text
