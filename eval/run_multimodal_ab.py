"""Three-condition multimodal A/B evaluation.

Conditions, run over the same held-out scenarios:

  text          audio and vision signals withheld (baseline)
  text_audio    audio turn-readiness available, vision withheld
  text_audio_visual
                audio turn-readiness and a facial high-load reading available

The question this answers: does adding a modality change the action the policy
selects, and in which direction? It does not claim either change is correct for
every candidate — that needs human raters, which the README lists as future work.

Run:
    python -m eval.run_multimodal_ab
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.perception.answer_quality import extract_answer_quality
from backend.app.perception.text_signals import extract_explicit_signals
from backend.app.reasoning.dialogue_policy import decide
from backend.app.schemas import (
    EmotionalState,
    EmotionLabel,
    SessionState,
    TurnReadiness,
)

RESULTS_PATH = Path(__file__).parent / "results" / "multimodal-ab-latest.json"

# Scenarios are written as what the candidate said plus what the sensors saw.
# The wording is fixed so the three conditions differ only in sensor input.
_SCENARIOS: list[dict] = [
    {
        "id": "complete_answer_calm",
        "input": (
            "I built a ticket classifier and evaluated it with macro-F1 on a "
            "held-out set, reaching 0.81."
        ),
        "audio": TurnReadiness.READY,
        "face": ("neutral", 0.0),
        "expect": "text-only should already be satisfied",
    },
    {
        "id": "complete_answer_tense_face",
        "input": (
            "I built a ticket classifier and evaluated it with macro-F1 on a "
            "held-out set, reaching 0.81."
        ),
        "audio": TurnReadiness.READY,
        "face": ("fearful", 0.62),
        "expect": "visual should yield space instead of pressing",
    },
    {
        "id": "thin_answer_mid_speech",
        "input": "I used accuracy.",
        "audio": TurnReadiness.THINKING_PAUSE,
        "face": ("neutral", 0.0),
        "expect": "audio should wait rather than interrupt a thinking pause",
    },
    {
        "id": "thin_answer_turn_ready",
        "input": "I used accuracy.",
        "audio": TurnReadiness.READY,
        "face": ("neutral", 0.0),
        "expect": "audio ready means normal follow-up is allowed",
    },
    {
        "id": "vague_answer_still_speaking",
        "input": "Well, it depends.",
        "audio": TurnReadiness.SPEAKING,
        "face": ("neutral", 0.0),
        "expect": "audio should wait while the candidate is still speaking",
    },
    {
        "id": "project_only_tense",
        "input": "I built a support ticket classifier for a fintech team.",
        "audio": TurnReadiness.READY,
        "face": ("angry", 0.55),
        "expect": "visual should yield space before probing evaluation",
    },
    {
        "id": "project_only_calm",
        "input": "I built a support ticket classifier for a fintech team.",
        "audio": TurnReadiness.READY,
        "face": ("neutral", 0.0),
        "expect": "no sensor pressure, normal follow-up",
    },
    {
        "id": "no_speech_detected",
        "input": "",
        "audio": TurnReadiness.LISTENING,
        "face": ("neutral", 0.0),
        "expect": "audio should wait; nothing to answer yet",
    },
    {
        "id": "answer_with_pause_but_complete",
        "input": (
            "I trained a churn model and measured AUC at 0.88 on a held-out "
            "split."
        ),
        "audio": TurnReadiness.THINKING_PAUSE,
        "face": ("happy", 0.7),
        "expect": "audio wait wins over a complete-looking answer",
    },
    {
        "id": "answer_low_confidence_face",
        "input": "I built a churn model with 0.88 AUC.",
        "audio": TurnReadiness.READY,
        "face": ("fearful", 0.2),
        "expect": "low-confidence face must not change the action",
    },
]

CONDITIONS = ("text", "text_audio", "text_audio_visual")


@dataclass
class ConditionResult:
    scenario_id: str
    condition: str
    action: str
    reason: str


@dataclass
class Report:
    conditions: list[str]
    results: list[ConditionResult] = field(default_factory=list)
    action_changes: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _emotional_state(face: tuple[str, float]) -> EmotionalState:
    label, confidence = face
    return EmotionalState(
        face_available=True,
        dominant_emotion=EmotionLabel(label),
        confidence=confidence,
    )


def _run_scenario(scenario: dict, condition: str) -> ConditionResult:
    signals = extract_explicit_signals(scenario["input"])

    if condition != "text":
        signals = signals.model_copy(
            update={"turn_readiness": scenario["audio"]}
        )

    evidence = extract_answer_quality(scenario["input"], signals)
    state = SessionState(session_id=f"{scenario['id']}-{condition}")

    if condition == "text_audio_visual":
        state = state.model_copy(
            update={"emotional_state": _emotional_state(scenario["face"])}
        )

    decision = decide(state, signals, evidence)
    return ConditionResult(
        scenario_id=scenario["id"],
        condition=condition,
        action=decision.action.value,
        reason=decision.reason.value,
    )


def build_report() -> Report:
    report = Report(conditions=list(CONDITIONS))

    by_scenario: dict[str, dict[str, ConditionResult]] = {}
    for scenario in _SCENARIOS:
        by_scenario[scenario["id"]] = {}
        for condition in CONDITIONS:
            result = _run_scenario(scenario, condition)
            report.results.append(result)
            by_scenario[scenario["id"]][condition] = result

    for scenario in _SCENARIOS:
        sid = scenario["id"]
        baseline = by_scenario[sid]["text"]
        audio = by_scenario[sid]["text_audio"]
        full = by_scenario[sid]["text_audio_visual"]
        report.action_changes.append(
            {
                "id": sid,
                "expect": scenario["expect"],
                "text": baseline.action,
                "text_audio": audio.action,
                "text_audio_visual": full.action,
                "audio_changed": baseline.action != audio.action,
                "visual_changed": audio.action != full.action,
            }
        )

    total = len(_SCENARIOS)
    report.summary = {
        "scenarios": total,
        "audio_changed_actions": sum(
            1 for c in report.action_changes if c["audio_changed"]
        ),
        "visual_changed_actions": sum(
            1 for c in report.action_changes if c["visual_changed"]
        ),
        "note": (
            "Counts describe policy behaviour under fixed synthetic sensor "
            "input, not accuracy against human judgement."
        ),
    }
    return report


def main() -> int:
    report = build_report()

    header = f"{'scenario':34} {'text':22} {'text_audio':22} {'text+audio+visual':22}"
    print(header)
    print("-" * len(header))
    for change in report.action_changes:
        print(
            f"{change['id']:34} {change['text']:22} "
            f"{change['text_audio']:22} {change['text_audio_visual']:22}"
        )

    print()
    print(json.dumps(report.summary, indent=2))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "conditions": report.conditions,
                "summary": report.summary,
                "action_changes": report.action_changes,
                "results": [
                    {
                        "scenario_id": r.scenario_id,
                        "condition": r.condition,
                        "action": r.action,
                        "reason": r.reason,
                    }
                    for r in report.results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
