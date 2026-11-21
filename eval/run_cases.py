"""Run the text baseline directly against its domain service."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from backend.app.schemas import (
    DecisionReason,
    DialogueAction,
    FollowupTopic,
    SessionState,
)
from backend.app.service import InterviewService

_CASES_PATH = Path(__file__).with_name("cases.json")
_RESULTS_PATH = Path(__file__).with_name("results") / "latest.json"


def _load_cases() -> list[dict[str, Any]]:
    with _CASES_PATH.open(encoding="utf-8") as source:
        document = json.load(source)

    if document.get("schema_version") != "1.0":
        raise ValueError('cases.json schema_version must be "1.0"')
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases.json must contain a non-empty cases list")

    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("every case must have a non-empty string id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)

        if not isinstance(case.get("candidate_input"), str):
            raise ValueError(f"{case_id}: candidate_input must be a string")
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"{case_id}: expected must be an object")
        try:
            DialogueAction(expected.get("action"))
            DecisionReason(expected.get("reason"))
            FollowupTopic(expected.get("followup_topic"))
        except ValueError as error:
            raise ValueError(f"{case_id}: invalid expected enum value") from error
        criteria = expected.get("reply_criteria")
        if not isinstance(criteria, list) or not all(
            isinstance(item, str) and item.strip() for item in criteria
        ):
            raise ValueError(f"{case_id}: reply_criteria must be non-empty strings")

    return cases


def run_cases() -> dict[str, Any]:
    service = InterviewService()
    outcomes: list[dict[str, Any]] = []

    for case in _load_cases():
        expected = case["expected"]
        result = service.process_turn(
            SessionState(session_id=f"eval-{case['id']}"),
            case["candidate_input"],
        )
        actual = {
            "action": result.decision.action.value,
            "reason": result.decision.reason.value,
            "followup_topic": result.decision.followup_topic.value,
        }
        expected_decision = {
            "action": expected["action"],
            "reason": expected["reason"],
            "followup_topic": expected["followup_topic"],
        }
        passed = actual == expected_decision
        outcomes.append(
            {
                "id": case["id"],
                "expected": expected_decision,
                "actual": actual,
                "reply": result.reply,
                "passed": passed,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'} {case['id']}")

    passed_count = sum(outcome["passed"] for outcome in outcomes)
    report = {
        "schema_version": "eval-v1",
        "summary": {
            "total": len(outcomes),
            "passed": passed_count,
            "failed": len(outcomes) - passed_count,
        },
        "cases": outcomes,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _RESULTS_PATH.open("w", encoding="utf-8") as target:
        json.dump(report, target, indent=2, ensure_ascii=False)
        target.write("\n")

    print(f"Summary: {passed_count} passed, {len(outcomes) - passed_count} failed")
    return report


def main() -> int:
    try:
        report = run_cases()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 1
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
