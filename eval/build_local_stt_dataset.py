"""Build a local Indonesian/English STT benchmark from your own recordings.

The public Indonesian speech available on Wikimedia Commons is one-word
pronunciation clips, which cannot support a word-error-rate claim. Rather than
report a number that looks like a benchmark but measures nothing, this script
builds a dataset from recordings you make yourself.

Usage
-----
1. Record clips and put them in `eval/recordings/`:

       ind_01.wav   ind_02.wav   eng_01.wav

   WAV, mono, any sample rate (it is resampled to 16 kHz on read).

2. Write the reference transcript for each clip in `eval/recordings/references.json`:

       {
         "ind_01": "Saya membangun model klasifikasi untuk tiket dukungan.",
         "ind_02": "Saya mengevaluasinya dengan macro F1 pada data uji.",
         "eng_01": "I built a ticket classifier and measured macro F1."
       }

3. Run:

       python -m eval.build_local_stt_dataset

   It writes `eval/stt_cases_local.json` with verified SHA-256 hashes, which
   `eval.run_stt_cases --cases eval/stt_cases_local.json` then consumes.

Design notes
------------
- Speakers are split by person, not by clip: calibration and held_out never
  share a speaker id. Recording two clips with the same person and putting one
  in each split would leak.
- Clips are declared as local `file://` assets so nothing is redistributed.
- At least two held-out cases from at least two speakers are required; the
  script refuses to write a dataset that cannot support a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

DEFAULT_RECORDINGS_DIR = Path(__file__).parent / "recordings"
DEFAULT_OUTPUT = Path(__file__).parent / "stt_cases_local.json"

_MIN_HELD_OUT_CASES = 2
_MIN_HELD_OUT_SPEAKERS = 2


class DatasetError(Exception):
    """Raised when the recordings cannot support a benchmark."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_info(path: Path) -> tuple[float, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except wave.Error as error:
        raise DatasetError(f"{path.name} is not a readable WAV file") from error
    if rate <= 0 or frames <= 0:
        raise DatasetError(f"{path.name} has no audio")
    return frames / rate, rate


def build(
    recordings_dir: Path,
    output_path: Path,
    plan: dict[str, dict[str, str]],
) -> dict:
    """Build the dataset JSON from a declared split plan.

    ``plan`` maps case id -> {"split", "speaker_id", "reference"}.
    """

    references_path = recordings_dir / "references.json"
    if not references_path.is_file():
        raise DatasetError(
            f"missing {references_path}. Write the reference transcript for "
            "each clip first."
        )
    references = json.loads(references_path.read_text(encoding="utf-8"))

    assets: list[dict] = []
    cases: list[dict] = []

    for case_id, spec in sorted(plan.items()):
        clip = recordings_dir / f"{case_id}.wav"
        if not clip.is_file():
            raise DatasetError(f"missing recording: {clip.name}")
        if case_id not in references:
            raise DatasetError(f"missing reference transcript for {case_id}")

        reference = str(references[case_id]).strip()
        if not reference:
            raise DatasetError(f"empty reference transcript for {case_id}")

        duration, rate = _wav_info(clip)
        if duration < 0.4:
            raise DatasetError(f"{clip.name} is shorter than 0.4 s")

        assets.append(
            {
                "id": case_id,
                "filename": clip.name,
                "url": clip.resolve().as_uri(),
                "sha256": _sha256(clip),
                "size_bytes": clip.stat().st_size,
                "media_type": "audio/wav",
            }
        )
        cases.append(
            {
                "id": case_id,
                "split": spec["split"],
                "speaker_id": spec["speaker_id"],
                "asset_id": case_id,
                "channel": "mono",
                "reference": reference,
            }
        )

    held_out = [c for c in cases if c["split"] == "held_out"]
    if len(held_out) < _MIN_HELD_OUT_CASES:
        raise DatasetError(
            f"need at least {_MIN_HELD_OUT_CASES} held-out cases, "
            f"found {len(held_out)}"
        )
    speakers = {c["speaker_id"] for c in held_out}
    if len(speakers) < _MIN_HELD_OUT_SPEAKERS:
        raise DatasetError(
            f"need at least {_MIN_HELD_OUT_SPEAKERS} distinct held-out "
            f"speakers, found {len(speakers)}"
        )

    all_speakers: dict[str, set[str]] = {}
    for case in cases:
        all_speakers.setdefault(case["speaker_id"], set()).add(case["split"])
    leaking = [s for s, splits in all_speakers.items() if len(splits) > 1]
    if leaking:
        raise DatasetError(
            "these speakers appear in both splits, which leaks: "
            + ", ".join(sorted(leaking))
        )

    dataset = {
        "schema_version": "stt-eval-v1",
        "source": {
            "repository": "local-recordings",
            "commit": "0" * 40,
            "note": (
                "Recorded locally by the project author. Not redistributed. "
                "Speaker-disjoint splits; see eval/recordings/references.json."
            ),
        },
        "model": {
            "name": "small",
            "device": "cpu",
            "compute_type": "int8",
            "language": "auto",
            "max_audio_seconds": 60,
        },
        "acceptance": {
            "min_held_out_success_rate": 1.0,
            "max_held_out_corpus_wer": 0.35,
            "max_held_out_mean_rtf": 1.0,
        },
        "assets": assets,
        "cases": cases,
    }

    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help="directory holding .wav files and references.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR / "plan.json",
        help="JSON mapping case id to {split, speaker_id}",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.plan.is_file():
        print(
            f"missing {args.plan}. Write the split plan first; see the "
            "module docstring for the shape.",
            file=sys.stderr,
        )
        return 2

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    try:
        dataset = build(args.recordings, args.output, plan)
    except DatasetError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.output} — {len(dataset['cases'])} cases, "
        f"{len(dataset['assets'])} assets"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
