"""Pilot validation script for FACS-based emotion mapping.

This script computes Cohen's kappa between the system's predicted emotion
and human-annotated labels on a small set of blendshape fixtures.

Usage:
    python -m eval.validate_emotion_mapping

The fixtures below are synthetic blendshape profiles representing clear
instances of each Ekman basic emotion. Real validation requires consented
recordings from multiple human subjects and multiple human annotators.

References:
- Ekman, P. & Friesen, W.V. (1978). Facial Action Coding System.
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
  Educational and Psychological Measurement, 20(1), 37-46.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.perception.vision_analyzer import analyze
from backend.app.schemas import BlendshapeReading, VisionObservation

# Synthetic blendshape profiles for each Ekman basic emotion.
# These are idealized activations, not real recordings.
# ponytail: real validation needs consented subjects + human annotators.
_FIXTURES: list[dict] = [
    {
        "label": "happy",
        "blendshapes": {
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.80,
            "cheekSquintLeft": 0.70,
            "cheekSquintRight": 0.65,
        },
    },
    {
        "label": "sad",
        "blendshapes": {
            "browInnerUp": 0.55,
            "browDownLeft": 0.45,
            "browDownRight": 0.40,
            "mouthFrownLeft": 0.60,
            "mouthFrownRight": 0.55,
        },
    },
    {
        "label": "surprised",
        "blendshapes": {
            "browInnerUp": 0.75,
            "browOuterUpLeft": 0.65,
            "browOuterUpRight": 0.70,
            "eyeWideLeft": 0.80,
            "eyeWideRight": 0.75,
            "jawOpen": 0.90,
        },
    },
    {
        "label": "angry",
        "blendshapes": {
            "browDownLeft": 0.70,
            "browDownRight": 0.65,
            "eyeWideLeft": 0.50,
            "eyeWideRight": 0.45,
            "eyeSquintLeft": 0.60,
            "eyeSquintRight": 0.55,
            "mouthClose": 0.70,
        },
    },
    {
        "label": "disgusted",
        "blendshapes": {
            "noseSneerLeft": 0.70,
            "noseSneerRight": 0.65,
            "mouthFrownLeft": 0.50,
            "mouthFrownRight": 0.45,
            "mouthLowerDownLeft": 0.40,
            "mouthLowerDownRight": 0.35,
        },
    },
    {
        "label": "fearful",
        "blendshapes": {
            "browInnerUp": 0.70,
            "browOuterUpLeft": 0.60,
            "browOuterUpRight": 0.65,
            "eyeWideLeft": 0.80,
            "eyeWideRight": 0.75,
            "jawOpen": 0.90,
            "browDownLeft": 0.50,
            "browDownRight": 0.45,
        },
    },
    {
        "label": "neutral",
        "blendshapes": {
            "mouthSmileLeft": 0.05,
            "eyeBlinkLeft": 0.10,
        },
    },
]


def _predict(fixture: dict) -> str:
    readings = [
        BlendshapeReading(name=name, score=score)
        for name, score in fixture["blendshapes"].items()
    ]
    obs = VisionObservation(
        face_available=True,
        face_count=1,
        blendshapes=readings,
    )
    state = analyze(obs)
    return state.dominant_emotion.value


def _cohen_kappa(
    system_labels: list[str],
    human_labels: list[str],
) -> tuple[float, dict[str, dict[str, int]]]:
    """Compute Cohen's kappa for two raters over nominal labels.

    Returns (kappa, confusion_matrix).
    """
    if len(system_labels) != len(human_labels):
        raise ValueError("label lists must have equal length")

    labels = sorted(set(system_labels) | set(human_labels))
    n = len(system_labels)

    # Build confusion matrix.
    matrix: dict[str, dict[str, int]] = {
        human: {sys: 0 for sys in labels} for human in labels
    }
    for sys, human in zip(system_labels, human_labels):
        matrix[human][sys] += 1

    # Observed agreement.
    po = sum(matrix[h][h] for h in labels) / n

    # Expected agreement by chance.
    human_marginals = {h: sum(matrix[h].values()) / n for h in labels}
    system_marginals = {
        s: sum(matrix[h][s] for h in labels) / n for s in labels
    }
    pe = sum(
        human_marginals[h] * system_marginals.get(h, 0.0) for h in labels
    )

    if pe == 1.0:
        return 1.0, matrix

    kappa = (po - pe) / (1.0 - pe)
    return kappa, matrix


def main() -> int:
    system_labels = [_predict(f) for f in _FIXTURES]
    human_labels = [f["label"] for f in _FIXTURES]

    kappa, matrix = _cohen_kappa(system_labels, human_labels)
    po = sum(matrix[h][h] for h in matrix) / len(_FIXTURES)

    report = {
        "fixture_count": len(_FIXTURES),
        "observed_agreement": round(po, 3),
        "cohen_kappa": round(kappa, 3),
        "system_predictions": system_labels,
        "human_labels": human_labels,
        "confusion_matrix": matrix,
        "note": (
            "Synthetic idealized blendshape fixtures. Not a substitute for "
            "consented multi-subject validation with multiple human annotators."
        ),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
