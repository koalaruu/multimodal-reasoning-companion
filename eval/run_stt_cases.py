"""Run a fixed-fixture STT benchmark against the selected app provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from backend.app.env_loader import load_dotenv
from backend.app.perception.speech_to_text import (
    FasterWhisperTranscriber,
    GroqWhisperTranscriber,
)
from eval.stt_evaluation import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CASES_PATH,
    DEFAULT_RESULTS_PATH,
    PublicAudioCache,
    evaluate_stt_dataset,
    load_stt_dataset,
)


def _write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _print_report(report: dict) -> None:
    for outcome in report["cases"]:
        if outcome["status"] == "passed":
            print(
                f"PASS {outcome['split']}/{outcome['id']} "
                f"WER={outcome['wer']:.4f} RTF={outcome['rtf']:.4f}"
            )
        else:
            print(
                f"FAIL {outcome['split']}/{outcome['id']} "
                f"failure={outcome['failure']}"
            )
    for split in ("calibration", "held_out"):
        summary = report["summary"][split]
        print(
            f"{split}: success={summary['successful']}/{summary['total']} "
            f"corpus_WER={summary['corpus_wer']:.4f} "
            f"mean_RTF={summary['mean_rtf']}"
        )
    status = "PASS" if report["acceptance"]["passed"] else "NOT MET"
    print(f"Target status: {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--provider", choices=("local", "groq"), default="local")
    parser.add_argument(
        "--groq-model",
        choices=("whisper-large-v3-turbo", "whisper-large-v3"),
        default="whisper-large-v3-turbo",
    )
    parser.add_argument(
        "--enforce-target",
        action="store_true",
        help="exit non-zero when held-out acceptance is not met",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_stt_dataset(args.cases)
    except (OSError, ValueError, ValidationError) as error:
        print(f"STT evaluation error: {error}", file=sys.stderr)
        return 1

    if args.validate_only:
        print(
            f"Valid STT dataset: {len(dataset.assets)} assets, "
            f"{len(dataset.cases)} cases"
        )
        return 0

    system = None
    if args.provider == "groq":
        if any(not asset.url.startswith("https://") for asset in dataset.assets):
            print("STT evaluation error: Groq only accepts public HTTPS fixtures.", file=sys.stderr)
            return 1
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            print("STT evaluation error: Groq requires GROQ_API_KEY.", file=sys.stderr)
            return 1
        transcriber = GroqWhisperTranscriber(
            api_key=api_key, model=args.groq_model, language=dataset.model.language,
        )
        system = {
            "provider": "groq",
            "name": args.groq_model,
            "device": "cloud",
            "language": dataset.model.language,
        }
    else:
        transcriber = FasterWhisperTranscriber(
            model_name=dataset.model.name,
            device=dataset.model.device,
            compute_type=dataset.model.compute_type,
            language=dataset.model.language,
            max_audio_seconds=dataset.model.max_audio_seconds,
        )
    provider = PublicAudioCache(dataset, args.cache_dir)
    report = evaluate_stt_dataset(dataset, transcriber, provider, system_under_test=system)
    _write_report(report, args.output)
    _print_report(report)
    if args.enforce_target and not report["acceptance"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
