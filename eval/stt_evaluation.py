"""Reproducible local STT evaluation with speaker-disjoint held-out cases."""

from __future__ import annotations

import hashlib
import platform
import re
import unicodedata
import wave
from contextlib import AbstractContextManager
from io import BytesIO
from pathlib import Path
from statistics import mean, median
from typing import BinaryIO, Protocol, Self
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.perception.speech_to_text import (
    AudioTranscriber,
    TranscriberUnavailableError,
    TranscriptionError,
)

_DEFAULT_CASES_PATH = Path(__file__).with_name("stt_cases.json")
_DEFAULT_CACHE_DIR = Path(__file__).with_name("cache") / "stt"
_DEFAULT_RESULTS_PATH = Path(__file__).with_name("results") / "stt-latest.json"
_ID_PATTERN = r"^[a-z0-9_]+$"
_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_SAMPLE_RATE = 16_000


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceMetadata(_ContractModel):
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    note: str = Field(min_length=1)


class SttModelConfig(_ContractModel):
    name: str = Field(min_length=1)
    device: str = Field(min_length=1)
    compute_type: str = Field(min_length=1)
    language: str = Field(min_length=1)
    max_audio_seconds: float = Field(gt=0)


class AcceptanceTarget(_ContractModel):
    min_held_out_success_rate: float = Field(ge=0, le=1)
    max_held_out_corpus_wer: float = Field(ge=0)
    max_held_out_mean_rtf: float = Field(gt=0)


class AudioAsset(_ContractModel):
    id: str = Field(pattern=_ID_PATTERN)
    filename: str = Field(min_length=1)
    url: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(gt=0)
    media_type: str = Field(pattern=r"^audio/[a-z0-9.+-]+$")

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        if Path(self.filename).name != self.filename:
            raise ValueError("asset filename must not contain a path")
        parsed = urlparse(self.url)
        if parsed.scheme == "file":
            # Locally recorded clips stay in the working tree and are never
            # redistributed; only the digest is committed.
            if not parsed.path:
                raise ValueError("file asset url must include a path")
            return self
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("asset url must use https or file")
        return self


class SttCase(_ContractModel):
    id: str = Field(pattern=_ID_PATTERN)
    split: str = Field(pattern=r"^(calibration|held_out)$")
    speaker_id: str = Field(pattern=_ID_PATTERN)
    asset_id: str = Field(pattern=_ID_PATTERN)
    channel: str = Field(pattern=r"^(mono|left|right)$")
    reference: str = Field(min_length=1)


class SttDataset(_ContractModel):
    schema_version: str
    source: SourceMetadata
    model: SttModelConfig
    acceptance: AcceptanceTarget
    assets: list[AudioAsset] = Field(min_length=1)
    cases: list[SttCase] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if self.schema_version != "stt-eval-v1":
            raise ValueError('schema_version must be "stt-eval-v1"')

        asset_ids = [asset.id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset ids must be unique")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case ids must be unique")

        known_assets = set(asset_ids)
        for case in self.cases:
            if case.asset_id not in known_assets:
                raise ValueError(f"{case.id}: unknown asset_id")

        calibration = [case for case in self.cases if case.split == "calibration"]
        held_out = [case for case in self.cases if case.split == "held_out"]
        if not calibration or not held_out:
            raise ValueError("calibration and held_out must both be non-empty")
        calibration_speakers = {case.speaker_id for case in calibration}
        held_out_speakers = {case.speaker_id for case in held_out}
        overlap = calibration_speakers & held_out_speakers
        if overlap:
            raise ValueError("speakers must be disjoint across splits")
        return self


def load_stt_dataset(path: Path = _DEFAULT_CASES_PATH) -> SttDataset:
    return SttDataset.model_validate_json(path.read_text(encoding="utf-8"))


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = (
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join("".join(characters).split())


def word_error_count(reference: str, hypothesis: str) -> tuple[int, int]:
    reference_words = normalize_transcript(reference).split()
    hypothesis_words = normalize_transcript(hypothesis).split()
    previous = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(
            hypothesis_words, start=1
        ):
            substitution_cost = int(reference_word != hypothesis_word)
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1], len(reference_words)


class CaseAudioProvider(Protocol):
    def open_case(self, case: SttCase) -> AbstractContextManager[BinaryIO]:
        """Open one mono case stream positioned at byte zero."""


class PublicAudioCache:
    def __init__(self, dataset: SttDataset, cache_dir: Path) -> None:
        self._assets = {asset.id: asset for asset in dataset.assets}
        self._cache_dir = cache_dir

    def open_case(self, case: SttCase) -> AbstractContextManager[BinaryIO]:
        asset = self._assets[case.asset_id]
        path = self._ensure_asset(asset)
        if case.channel == "mono":
            return path.open("rb")
        return _extract_channel(path, case.channel)

    def _ensure_asset(self, asset: AudioAsset) -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / asset.filename
        if path.exists() and _asset_matches(path, asset):
            return path

        part_path = path.with_suffix(path.suffix + ".part")
        part_path.unlink(missing_ok=True)

        if urlparse(asset.url).scheme == "file":
            source_path = Path(urlparse(asset.url).path.lstrip("/"))
            if not source_path.is_file():
                raise ValueError(f"{asset.id}: local asset is missing")
            if not _asset_matches(source_path, asset):
                raise ValueError(
                    f"{asset.id}: local asset does not match its declared digest"
                )
            return source_path

        request = Request(asset.url, headers={"User-Agent": "stt-eval-v1"})
        try:
            with urlopen(request, timeout=60) as source, part_path.open("wb") as target:
                received = 0
                while chunk := source.read(_DOWNLOAD_CHUNK_BYTES):
                    received += len(chunk)
                    if received > asset.size_bytes:
                        raise ValueError(f"{asset.id}: download exceeds declared size")
                    target.write(chunk)
            if not _asset_matches(part_path, asset):
                raise ValueError(f"{asset.id}: downloaded asset failed verification")
            part_path.replace(path)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise
        return path


def _asset_matches(path: Path, asset: AudioAsset) -> bool:
    if path.stat().st_size != asset.size_bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest() == asset.sha256


def _extract_channel(path: Path, channel: str) -> BytesIO:
    try:
        import numpy as np
        from faster_whisper.audio import decode_audio
    except (ImportError, OSError) as error:
        raise TranscriberUnavailableError(
            "STT evaluation dependencies are unavailable"
        ) from error

    left, right = decode_audio(str(path), split_stereo=True)
    samples = left if channel == "left" else right
    pcm = (np.clip(samples, -1.0, 1.0) * 32_767).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    output.seek(0)
    return output


def evaluate_stt_dataset(
    dataset: SttDataset,
    transcriber: AudioTranscriber,
    audio_provider: CaseAudioProvider,
) -> dict:
    outcomes = []
    for index, case in enumerate(dataset.cases):
        reference_errors, reference_words = word_error_count(case.reference, "")
        try:
            with audio_provider.open_case(case) as audio:
                result = transcriber.transcribe(audio)
            word_errors, reference_words = word_error_count(
                case.reference, result.text
            )
            wer = word_errors / reference_words if reference_words else 0.0
            rtf = (
                result.processing_duration_ms / result.audio_duration_ms
                if result.audio_duration_ms > 0
                else None
            )
            outcome = {
                "id": case.id,
                "split": case.split,
                "speaker_id": case.speaker_id,
                "first_request": index == 0,
                "status": "passed",
                "reference": case.reference,
                "hypothesis": result.text,
                "language": result.language,
                "language_probability": result.language_probability,
                "audio_duration_ms": result.audio_duration_ms,
                "processing_duration_ms": result.processing_duration_ms,
                "rtf": _round(rtf),
                "reference_words": reference_words,
                "word_errors": word_errors,
                "wer": _round(wer),
                "failure": None,
            }
        except (OSError, ValueError, TranscriptionError) as error:
            failure = (
                "transcription_error"
                if isinstance(error, TranscriptionError)
                else "audio_or_runtime_error"
            )
            outcome = {
                "id": case.id,
                "split": case.split,
                "speaker_id": case.speaker_id,
                "first_request": index == 0,
                "status": "failed",
                "reference": case.reference,
                "hypothesis": None,
                "language": None,
                "language_probability": None,
                "audio_duration_ms": None,
                "processing_duration_ms": None,
                "rtf": None,
                "reference_words": reference_words,
                "word_errors": reference_errors,
                "wer": 1.0,
                "failure": failure,
            }
        outcomes.append(outcome)

    calibration = _summarize(
        [outcome for outcome in outcomes if outcome["split"] == "calibration"]
    )
    held_out = _summarize(
        [outcome for outcome in outcomes if outcome["split"] == "held_out"]
    )
    target = dataset.acceptance.model_dump(mode="json")
    acceptance = {
        "success_rate_passed": held_out["success_rate"]
        >= target["min_held_out_success_rate"],
        "corpus_wer_passed": held_out["corpus_wer"]
        <= target["max_held_out_corpus_wer"],
        "mean_rtf_passed": held_out["mean_rtf"] is not None
        and held_out["mean_rtf"] <= target["max_held_out_mean_rtf"],
    }
    acceptance["passed"] = all(acceptance.values())
    return {
        "schema_version": "stt-report-v1",
        "system_under_test": dataset.model.model_dump(mode="json"),
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source": dataset.source.model_dump(mode="json"),
        "target": target,
        "summary": {"calibration": calibration, "held_out": held_out},
        "acceptance": acceptance,
        "cases": outcomes,
    }


def _summarize(outcomes: list[dict]) -> dict:
    successful = [outcome for outcome in outcomes if outcome["status"] == "passed"]
    reference_words = sum(outcome["reference_words"] for outcome in outcomes)
    word_errors = sum(outcome["word_errors"] for outcome in outcomes)
    rtfs = [outcome["rtf"] for outcome in successful if outcome["rtf"] is not None]
    processing = [outcome["processing_duration_ms"] for outcome in successful]
    return {
        "total": len(outcomes),
        "successful": len(successful),
        "success_rate": _round(len(successful) / len(outcomes)),
        "reference_words": reference_words,
        "word_errors": word_errors,
        "corpus_wer": _round(word_errors / reference_words),
        "mean_case_wer": _round(mean(outcome["wer"] for outcome in outcomes)),
        "mean_rtf": _round(mean(rtfs)) if rtfs else None,
        "median_rtf": _round(median(rtfs)) if rtfs else None,
        "mean_processing_duration_ms": (
            round(mean(processing)) if processing else None
        ),
    }


def _round(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


DEFAULT_CASES_PATH = _DEFAULT_CASES_PATH
DEFAULT_CACHE_DIR = _DEFAULT_CACHE_DIR
DEFAULT_RESULTS_PATH = _DEFAULT_RESULTS_PATH
