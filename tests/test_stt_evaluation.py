"""Deterministic tests for the STT benchmark without model downloads."""

from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.perception.speech_to_text import (
    NoSpeechDetectedError,
    TranscriptionResult,
)
from eval.run_stt_cases import main
from eval.stt_evaluation import (
    PublicAudioCache,
    evaluate_stt_dataset,
    load_stt_dataset,
    normalize_transcript,
    word_error_count,
)


class InMemoryAudioProvider:
    def open_case(self, case):
        return BytesIO(case.id.encode("utf-8"))


class ReferenceTranscriber:
    def __init__(self, references: dict[str, str], fail_case: str | None = None):
        self.references = references
        self.fail_case = fail_case

    def transcribe(self, audio) -> TranscriptionResult:
        case_id = audio.read().decode("utf-8")
        if case_id == self.fail_case:
            raise NoSpeechDetectedError("private failure detail")
        return TranscriptionResult(
            text=self.references[case_id],
            language="en",
            language_probability=0.99,
            audio_duration_ms=1_000,
            processing_duration_ms=250,
            model="fake",
        )


def test_manifest_is_speaker_disjoint_and_pinned() -> None:
    dataset = load_stt_dataset()

    calibration_speakers = {
        case.speaker_id for case in dataset.cases if case.split == "calibration"
    }
    held_out_speakers = {
        case.speaker_id for case in dataset.cases if case.split == "held_out"
    }

    assert len(dataset.assets) == 2
    assert len(dataset.cases) == 3
    assert calibration_speakers.isdisjoint(held_out_speakers)
    assert all(len(asset.sha256) == 64 for asset in dataset.assets)
    assert dataset.source.commit == "6cb639e70786aea42b9233006ae60eaf07190bdb"


def test_normalization_and_word_error_count_are_deterministic() -> None:
    assert normalize_transcript("  Macro-F1: 84%! ") == "macro f1 84"
    assert word_error_count("Hello, world!", "hello word") == (1, 2)
    assert word_error_count("one two", "one two extra") == (1, 2)


def test_exact_transcripts_pass_held_out_acceptance() -> None:
    dataset = load_stt_dataset()
    references = {case.id: case.reference for case in dataset.cases}

    report = evaluate_stt_dataset(
        dataset,
        ReferenceTranscriber(references),
        InMemoryAudioProvider(),
    )

    assert report["summary"]["calibration"]["corpus_wer"] == 0
    assert report["summary"]["held_out"] == {
        "total": 2,
        "successful": 2,
        "success_rate": 1.0,
        "reference_words": 22,
        "word_errors": 0,
        "corpus_wer": 0.0,
        "mean_case_wer": 0.0,
        "mean_rtf": 0.25,
        "median_rtf": 0.25,
        "mean_processing_duration_ms": 250,
    }
    assert report["acceptance"]["passed"] is True
    assert report["cases"][0]["first_request"] is True
    assert all(case["failure"] is None for case in report["cases"])


def test_failed_held_out_case_cannot_be_hidden_from_acceptance() -> None:
    dataset = load_stt_dataset()
    references = {case.id: case.reference for case in dataset.cases}

    report = evaluate_stt_dataset(
        dataset,
        ReferenceTranscriber(references, fail_case="held_out_stereo_right"),
        InMemoryAudioProvider(),
    )

    held_out = report["summary"]["held_out"]
    failed = next(
        case for case in report["cases"] if case["id"] == "held_out_stereo_right"
    )
    assert held_out["success_rate"] == 0.5
    assert held_out["word_errors"] >= failed["reference_words"]
    assert failed["failure"] == "transcription_error"
    assert "private failure detail" not in json.dumps(report)
    assert report["acceptance"]["passed"] is False


def test_speaker_leakage_across_splits_is_rejected(tmp_path: Path) -> None:
    fixture = json.loads(Path("eval/stt_cases.json").read_text(encoding="utf-8"))
    fixture["cases"][1]["speaker_id"] = fixture["cases"][0]["speaker_id"]
    path = tmp_path / "leaked.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(ValidationError, match="speakers must be disjoint"):
        load_stt_dataset(path)


def test_validate_only_does_not_download_models_or_audio(tmp_path: Path) -> None:
    assert main(["--validate-only", "--cache-dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("status", [200, 206])
def test_archive_asset_only_accepts_the_declared_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int,
) -> None:
    dataset = load_stt_dataset()
    clip = b"exact utterance bytes"
    start = 512
    end = start + len(clip) - 1
    asset = dataset.assets[0].model_copy(update={
        "range_start": start,
        "size_bytes": len(clip),
        "sha256": sha256(clip).hexdigest(),
    })
    dataset = dataset.model_copy(update={"assets": [asset, *dataset.assets[1:]]})

    def upstream(request, timeout):
        assert request.get_header("Range") == f"bytes={start}-{end}"
        response = BytesIO(clip)
        response.status = status
        response.headers = {"Content-Range": f"bytes {start}-{end}/4096"} if status == 206 else {}
        return response

    monkeypatch.setattr("eval.stt_evaluation.urlopen", upstream)
    cache = PublicAudioCache(dataset, tmp_path)
    if status == 206:
        with cache.open_case(dataset.cases[0]) as audio:
            assert audio.read() == clip
    else:
        with pytest.raises(ValueError, match="declared byte range"):
            cache.open_case(dataset.cases[0])
        assert not list(tmp_path.iterdir())


def test_groq_benchmark_rejects_local_recordings(tmp_path: Path, capsys) -> None:
    fixture = json.loads(Path("eval/stt_cases.json").read_text(encoding="utf-8"))
    fixture["assets"][0]["url"] = (tmp_path / "private.flac").as_uri()
    manifest = tmp_path / "personal.json"
    manifest.write_text(json.dumps(fixture), encoding="utf-8")
    output = tmp_path / "report.json"

    assert main(["--cases", str(manifest), "--provider", "groq", "--output", str(output)]) == 1
    assert "only accepts public HTTPS fixtures" in capsys.readouterr().err
    assert not output.exists()
