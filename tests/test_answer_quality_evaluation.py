"""Contract and reporting tests for the answer-quality benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.answer_quality_evaluation import (
    evaluate_benchmark,
    load_benchmark,
    main,
    write_report,
)


def test_fixture_has_disjoint_calibration_and_held_out_splits() -> None:
    document = load_benchmark()

    calibration_ids = {case.id for case in document.calibration}
    held_out_ids = {case.id for case in document.held_out}

    assert len(calibration_ids) == 6
    assert len(held_out_ids) == 6
    assert calibration_ids.isdisjoint(held_out_ids)
    assert document.acceptance.held_out_min_exact_match_rate == 0.8
    assert document.acceptance.held_out_min_critical_action_rate == 1.0


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    source = Path("eval/answer_quality_cases.json")
    fixture = json.loads(source.read_text(encoding="utf-8"))
    fixture["held_out"][0]["id"] = fixture["calibration"][0]["id"]
    duplicate_fixture = tmp_path / "duplicate.json"
    duplicate_fixture.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(ValidationError, match="case ids must be unique"):
        load_benchmark(duplicate_fixture)


def test_current_baseline_report_is_complete_and_omits_candidate_text(
    tmp_path: Path,
) -> None:
    document = load_benchmark()
    report = evaluate_benchmark(document)
    output = tmp_path / "report.json"
    write_report(report, output)

    assert report["system_under_test"] == "deterministic-keyword-baseline"
    assert report["summary"]["calibration"]["total"] == 6
    assert report["summary"]["held_out"]["total"] == 6
    assert len(report["cases"]) == 12
    assert {case["split"] for case in report["cases"]} == {
        "calibration",
        "held_out",
    }
    assert "The weather is nice today." not in output.read_text(encoding="utf-8")


def test_report_only_mode_succeeds_while_target_is_not_met(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "report.json"

    assert main(["--output", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["acceptance"]["passed"] is False


def test_enforcement_mode_fails_while_target_is_not_met(
    tmp_path: Path,
) -> None:
    assert (
        main(
            [
                "--output",
                str(tmp_path / "report.json"),
                "--enforce-target",
            ]
        )
        == 1
    )
