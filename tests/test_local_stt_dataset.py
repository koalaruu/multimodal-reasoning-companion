"""Tests for the local STT dataset builder."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from eval.build_local_stt_dataset import DatasetError, build


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 16_000) -> None:
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


def _setup(tmp_path: Path, clips: dict[str, float]) -> Path:
    for case_id, seconds in clips.items():
        _write_wav(tmp_path / f"{case_id}.wav", seconds)
    (tmp_path / "references.json").write_text(
        json.dumps({case_id: f"reference for {case_id}" for case_id in clips}),
        encoding="utf-8",
    )
    return tmp_path


def _plan(**entries) -> dict[str, dict[str, str]]:
    return {
        case_id: {"split": split, "speaker_id": speaker}
        for case_id, (split, speaker) in entries.items()
    }


class TestBuildsValidDataset:
    def test_two_speakers_two_splits(self, tmp_path: Path) -> None:
        recordings = _setup(
            tmp_path, {"ind_01": 1.0, "ind_02": 1.0, "eng_01": 1.0}
        )
        plan = _plan(
            ind_01=("calibration", "speaker_a"),
            ind_02=("held_out", "speaker_b"),
            eng_01=("held_out", "speaker_c"),
        )

        dataset = build(recordings, tmp_path / "out.json", plan)

        assert len(dataset["cases"]) == 3
        assert len(dataset["assets"]) == 3
        assert dataset["model"]["language"] == "auto"
        assert all(a["url"].startswith("file://") for a in dataset["assets"])
        assert all(len(a["sha256"]) == 64 for a in dataset["assets"])

    def test_output_file_is_written(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0, "c1": 1.0})
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
            c1=("held_out", "s3"),
        )
        output = tmp_path / "dataset.json"

        build(recordings, output, plan)

        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["schema_version"] == "stt-eval-v1"
        assert len(written["cases"]) == 3


class TestRefusesUnusableDatasets:
    def test_rejects_too_few_held_out_cases(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0})
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
        )

        with pytest.raises(DatasetError, match="at least 2 held-out"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_single_held_out_speaker(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0, "c1": 1.0})
        plan = _plan(
            a1=("calibration", "s0"),
            b1=("held_out", "same"),
            c1=("held_out", "same"),
        )

        with pytest.raises(DatasetError, match="distinct held-out speakers"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_speaker_in_both_splits(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0, "c1": 1.0})
        plan = _plan(
            a1=("calibration", "leaky"),
            b1=("held_out", "leaky"),
            c1=("held_out", "other"),
        )

        with pytest.raises(DatasetError, match="leaks"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_missing_recording(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0})
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
            ghost=("held_out", "s3"),
        )

        with pytest.raises(DatasetError, match="missing recording"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_missing_reference(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 1.0, "b1": 1.0, "c1": 1.0})
        (recordings / "references.json").write_text(
            json.dumps({"a1": "one", "b1": "two"}), encoding="utf-8"
        )
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
            c1=("held_out", "s3"),
        )

        with pytest.raises(DatasetError, match="missing reference"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_too_short_clip(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"a1": 0.1, "b1": 1.0, "c1": 1.0})
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
            c1=("held_out", "s3"),
        )

        with pytest.raises(DatasetError, match="shorter than"):
            build(recordings, tmp_path / "out.json", plan)

    def test_rejects_non_wav(self, tmp_path: Path) -> None:
        recordings = _setup(tmp_path, {"b1": 1.0, "c1": 1.0})
        (recordings / "a1.wav").write_bytes(b"not a wav file")
        (recordings / "references.json").write_text(
            json.dumps({"a1": "x", "b1": "y", "c1": "z"}), encoding="utf-8"
        )
        plan = _plan(
            a1=("calibration", "s1"),
            b1=("held_out", "s2"),
            c1=("held_out", "s3"),
        )

        with pytest.raises(DatasetError, match="not a readable WAV"):
            build(recordings, tmp_path / "out.json", plan)
