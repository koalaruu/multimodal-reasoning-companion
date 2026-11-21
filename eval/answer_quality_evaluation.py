"""Validate and report the answer-quality benchmark against the current baseline."""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from backend.app.schemas import SessionState
from backend.app.service import InterviewService

_DEFAULT_CASES_PATH = Path(__file__).with_name("answer_quality_cases.json")
_DEFAULT_RESULTS_PATH = Path(__file__).with_name("results") / "answer-quality-latest.json"


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputType(str, Enum):
    ANSWER = "answer"
    CONTROL = "control"
    MISSING = "missing"


class Relevance(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class ExpectedAction(str, Enum):
    OFFER_HINT = "offer_hint"
    WAIT = "wait"
    ASK_FOLLOWUP = "ask_followup"
    CONTINUE_INTERVIEW = "continue_interview"
    END_SESSION = "end_session"
    REQUEST_INPUT = "request_input"
    REQUEST_CLARIFICATION = "request_clarification"


class ExpectedTopic(str, Enum):
    NONE = "none"
    EVALUATION_METHOD = "evaluation_method"
    CLASS_LEVEL_PERFORMANCE = "class_level_performance"


class ExpectedDecision(_ContractModel):
    action: ExpectedAction
    followup_topic: ExpectedTopic

    @model_validator(mode="after")
    def validate_topic(self) -> Self:
        asks_followup = self.action == ExpectedAction.ASK_FOLLOWUP
        has_topic = self.followup_topic != ExpectedTopic.NONE
        if asks_followup != has_topic:
            raise ValueError("only ask_followup decisions may define a topic")
        return self


class ExpectedOutcome(_ContractModel):
    input_type: InputType
    relevance: Relevance
    project_described: StrictBool
    evaluation_method_mentioned: StrictBool
    evaluation_result_mentioned: StrictBool
    decision: ExpectedDecision

    @model_validator(mode="after")
    def validate_input_semantics(self) -> Self:
        if self.input_type == InputType.ANSWER:
            if self.relevance == Relevance.NOT_APPLICABLE:
                raise ValueError("answers require an applicable relevance label")
        elif self.relevance != Relevance.NOT_APPLICABLE:
            raise ValueError("control and missing inputs use not_applicable relevance")

        if self.input_type == InputType.MISSING:
            if self.decision.action != ExpectedAction.REQUEST_INPUT:
                raise ValueError("missing input must request input")
        if self.decision.action == ExpectedAction.REQUEST_CLARIFICATION:
            if self.input_type != InputType.ANSWER or self.relevance not in {
                Relevance.IRRELEVANT,
                Relevance.UNCLEAR,
            }:
                raise ValueError(
                    "clarification is reserved for irrelevant or unclear answers"
                )
        return self


class BenchmarkCase(_ContractModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str = Field(min_length=1)
    candidate_input: str
    critical_action: StrictBool
    expected: ExpectedOutcome


class AcceptanceTarget(_ContractModel):
    held_out_min_exact_match_rate: float = Field(ge=0.0, le=1.0)
    held_out_min_critical_action_rate: float = Field(ge=0.0, le=1.0)


class BenchmarkDocument(_ContractModel):
    schema_version: str
    interview_question: str = Field(min_length=1)
    acceptance: AcceptanceTarget
    calibration: list[BenchmarkCase] = Field(min_length=1)
    held_out: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        if self.schema_version != "answer-quality-v1":
            raise ValueError('schema_version must be "answer-quality-v1"')
        ids = [case.id for case in self.calibration + self.held_out]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique across both splits")
        return self


def load_benchmark(path: Path = _DEFAULT_CASES_PATH) -> BenchmarkDocument:
    """Load a strict, versioned benchmark fixture."""

    return BenchmarkDocument.model_validate_json(path.read_text(encoding="utf-8"))


def _input_type(candidate_input: str, result: object) -> InputType:
    if not candidate_input.strip():
        return InputType.MISSING
    signals = result.signals
    if signals.hint_requested or signals.thinking_time_requested or signals.stop_requested:
        return InputType.CONTROL
    return InputType.ANSWER


def _current_baseline_outcome(case: BenchmarkCase, service: InterviewService) -> dict:
    result = service.process_turn(
        SessionState(session_id=f"answer-quality-{case.id}"),
        case.candidate_input,
    )
    input_type = _input_type(case.candidate_input, result)
    evidence = result.content_evidence
    if input_type != InputType.ANSWER:
        relevance = Relevance.NOT_APPLICABLE
    elif evidence.project_mentioned or evidence.evaluation_mentioned:
        relevance = Relevance.RELEVANT
    else:
        relevance = Relevance.UNCLEAR

    return {
        "input_type": input_type.value,
        "relevance": relevance.value,
        "project_described": evidence.project_mentioned,
        "evaluation_method_mentioned": evidence.evaluation_mentioned,
        "evaluation_result_mentioned": False,
        "decision": {
            "action": result.decision.action.value,
            "followup_topic": result.decision.followup_topic.value,
        },
    }


def _summarize(outcomes: list[dict]) -> dict:
    total = len(outcomes)
    exact_matches = sum(outcome["exact_match"] for outcome in outcomes)
    decision_matches = sum(outcome["decision_match"] for outcome in outcomes)
    critical = [outcome for outcome in outcomes if outcome["critical_action"]]
    critical_matches = sum(outcome["decision_match"] for outcome in critical)
    return {
        "total": total,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / total,
        "decision_matches": decision_matches,
        "decision_match_rate": decision_matches / total,
        "critical_actions": len(critical),
        "critical_action_matches": critical_matches,
        "critical_action_match_rate": critical_matches / len(critical),
    }


def _evaluate_split(
    split: str,
    cases: list[BenchmarkCase],
    service: InterviewService,
) -> list[dict]:
    outcomes = []
    for case in cases:
        expected = case.expected.model_dump(mode="json")
        actual = _current_baseline_outcome(case, service)
        outcomes.append(
            {
                "id": case.id,
                "split": split,
                "critical_action": case.critical_action,
                "expected": expected,
                "actual": actual,
                "decision_match": actual["decision"] == expected["decision"],
                "exact_match": actual == expected,
            }
        )
    return outcomes


def evaluate_benchmark(document: BenchmarkDocument) -> dict:
    """Measure the current keyword baseline without changing its policy."""

    service = InterviewService()
    calibration = _evaluate_split("calibration", document.calibration, service)
    held_out = _evaluate_split("held_out", document.held_out, service)
    calibration_summary = _summarize(calibration)
    held_out_summary = _summarize(held_out)
    target = document.acceptance.model_dump(mode="json")
    acceptance = {
        "exact_match_rate_passed": held_out_summary["exact_match_rate"]
        >= target["held_out_min_exact_match_rate"],
        "critical_action_rate_passed": held_out_summary[
            "critical_action_match_rate"
        ]
        >= target["held_out_min_critical_action_rate"],
    }
    acceptance["passed"] = all(acceptance.values())
    return {
        "schema_version": "answer-quality-report-v1",
        "system_under_test": "deterministic-keyword-baseline",
        "target": target,
        "summary": {
            "calibration": calibration_summary,
            "held_out": held_out_summary,
        },
        "acceptance": acceptance,
        "cases": calibration + held_out,
    }


def write_report(report: dict, path: Path = _DEFAULT_RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _print_report(report: dict) -> None:
    for outcome in report["cases"]:
        exact = "PASS" if outcome["exact_match"] else "FAIL"
        decision = "PASS" if outcome["decision_match"] else "FAIL"
        print(f"{exact} {outcome['split']}/{outcome['id']} (action: {decision})")

    for split in ("calibration", "held_out"):
        summary = report["summary"][split]
        print(
            f"{split}: {summary['exact_matches']}/{summary['total']} exact, "
            f"{summary['decision_matches']}/{summary['total']} actions"
        )
    status = "PASS" if report["acceptance"]["passed"] else "NOT MET"
    print(f"Target status: {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=_DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--enforce-target",
        action="store_true",
        help="exit non-zero while the future semantic target is not met",
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate_benchmark(load_benchmark(args.cases))
        write_report(report, args.output)
    except (OSError, ValueError) as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        return 1

    _print_report(report)
    if args.enforce_target and not report["acceptance"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
