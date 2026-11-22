"""Map MediaPipe blendshapes to FACS AU proxies and emotion scores.

Mapping follows Ekman & Friesen (1978) documented AU-emotion combinations.
MediaPipe blendshapes are NOT 1:1 with FACS action units; each mapping is a
best-effort proxy validated against the FACS reference at
https://www.cs.cmu.edu/~face/facs.htm.

Scientific limitations are documented in the project README. These scores
represent AU-combination evidence strength, NOT direct mental state measurement.
See Cabitza et al. (2022), Cross et al. (2023), Feldman Barrett et al. (2019).
"""

from __future__ import annotations

from backend.app.schemas import (
    BlendshapeReading,
    EmotionalState,
    EmotionLabel,
    VisionObservation,
)

# Activation threshold below which a blendshape is considered inactive.
# MediaPipe scores are noisy near zero; 0.15 suppresses micro-fluctuations
# while preserving genuine activations. This is a tunable heuristic, not
# a validated psychometric threshold.
_ACTIVATION_THRESHOLD = 0.15

# MediaPipe blendshape name → FACS AU proxy.
# Based on anatomical correspondence documented at CMU FACS reference.
# Left/right asymmetric blendshapes are averaged into a single AU.
_BLENDSHAPE_TO_AU: dict[str, str] = {
    "eyeBlinkLeft": "AU43",
    "eyeBlinkRight": "AU43",
    "eyeLookUpLeft": "AU63",
    "eyeLookUpRight": "AU63",
    "eyeLookDownLeft": "AU64",
    "eyeLookDownRight": "AU64",
    "eyeLookInLeft": "AU62",
    "eyeLookInRight": "AU61",
    "eyeLookOutLeft": "AU61",
    "eyeLookOutRight": "AU62",
    "browDownLeft": "AU4",
    "browDownRight": "AU4",
    "browInnerUp": "AU1",
    "browOuterUpLeft": "AU2",
    "browOuterUpRight": "AU2",
    "cheekPuff": "AU13",
    "cheekSquintLeft": "AU6",
    "cheekSquintRight": "AU6",
    "eyeSquintLeft": "AU7",
    "eyeSquintRight": "AU7",
    "eyeWideLeft": "AU5",
    "eyeWideRight": "AU5",
    "jawOpen": "AU27",
    "jawLeft": "AU56",
    "jawRight": "AU57",
    "jawForward": "AU57",
    "mouthClose": "AU24",
    "mouthFrownLeft": "AU15",
    "mouthFrownRight": "AU15",
    "mouthDimpleLeft": "AU14",
    "mouthDimpleRight": "AU14",
    "mouthLowerDownLeft": "AU16",
    "mouthLowerDownRight": "AU16",
    "mouthPressLeft": "AU24",
    "mouthPressRight": "AU24",
    "mouthPucker": "AU18",
    "mouthRollLower": "AU28",
    "mouthRollUpper": "AU28",
    "mouthShrugLower": "AU17",
    "mouthShrugUpper": "AU10",
    "mouthSmileLeft": "AU12",
    "mouthSmileRight": "AU12",
    "mouthStretchLeft": "AU20",
    "mouthStretchRight": "AU20",
    "mouthUpperUpLeft": "AU10",
    "mouthUpperUpRight": "AU10",
    "noseSneerLeft": "AU9",
    "noseSneerRight": "AU9",
    "lipPress": "AU24",
    "lipSuck": "AU28",
    "tongueOut": "AU19",
}

# Ekman & Friesen (1978) documented AU combinations for basic emotions.
# Source: CMU FACS reference and ICCV2025 supplementary Table 2.
# Each emotion maps to a set of required AUs. Score = mean of activated AU values.
_EMOTION_AU_MAP: dict[EmotionLabel, frozenset[str]] = {
    EmotionLabel.HAPPY: frozenset({"AU6", "AU12"}),
    EmotionLabel.SAD: frozenset({"AU1", "AU4", "AU15"}),
    EmotionLabel.SURPRISED: frozenset({"AU1", "AU2", "AU5", "AU27"}),
    EmotionLabel.FEARFUL: frozenset({"AU1", "AU2", "AU5", "AU27"}),
    EmotionLabel.ANGRY: frozenset({"AU4", "AU5", "AU7", "AU23"}),
    EmotionLabel.DISGUSTED: frozenset({"AU9", "AU15", "AU16"}),
}

# Surprise and fear share the same AU set in basic emotion theory.
# We disambiguate using AU4 (brow lowerer): present → fear, absent → surprise.
# This is a simplification acknowledged in README limitations.

# AU23 (Lip Tightener) is not directly available from MediaPipe blendshapes.
# We approximate it from mouthClose + lipPress.
_AU23_BLENDSHAPES = frozenset({"mouthClose", "lipPress"})


def _blendshape_dict(reading: list[BlendshapeReading]) -> dict[str, float]:
    """Convert blendshape readings to a name→score lookup."""
    return {r.name: r.score for r in reading}


def _resolve_au23(blendshapes: dict[str, float]) -> float:
    """Approximate AU23 from available blendshape proxies."""
    values = [blendshapes.get(name, 0.0) for name in _AU23_BLENDSHAPES]
    return sum(values) / len(values) if values else 0.0


def _au_scores(blendshapes: dict[str, float]) -> dict[str, float]:
    """Map blendshape values to AU proxy scores in [0, 1].

    For asymmetric blendshapes (Left/Right), take the maximum to avoid
    averaging away genuine unilateral activations.
    """
    au_values: dict[str, list[float]] = {}
    for name, score in blendshapes.items():
        au = _BLENDSHAPE_TO_AU.get(name)
        if au is None:
            continue
        au_values.setdefault(au, []).append(score)

    scores: dict[str, float] = {}
    for au, vals in au_values.items():
        scores[au] = max(vals)

    # AU23 proxy from blendshape combination.
    if _resolve_au23(blendshapes) > _ACTIVATION_THRESHOLD:
        scores["AU23"] = _resolve_au23(blendshapes)

    return scores


def _emotion_scores(
    au_scores: dict[str, float],
) -> dict[str, float]:
    """Compute emotion scores from AU activation.

    Score = mean of AU values for each emotion's required combination.
    An emotion only scores above zero if ALL its required AUs are present.
    """
    results: dict[str, float] = {}
    for emotion, required_aus in _EMOTION_AU_MAP.items():
        activated = [au_scores.get(au, 0.0) for au in required_aus]
        # Require all AUs to be above threshold for the emotion to register.
        if all(a > _ACTIVATION_THRESHOLD for a in activated):
            results[emotion.value] = sum(activated) / len(activated)
        else:
            results[emotion.value] = 0.0
    return results


def _disambiguate_surprise_fear(
    emotion_scores: dict[str, float],
    au_scores: dict[str, float],
) -> dict[str, float]:
    """Surprise and fear share AU1+AU2+AU5+AU27. Use AU4 to split."""
    surprise = emotion_scores.get("surprised", 0.0)
    fear = emotion_scores.get("fearful", 0.0)
    if surprise <= 0.0 and fear <= 0.0:
        return emotion_scores

    au4 = au_scores.get("AU4", 0.0)
    if au4 > _ACTIVATION_THRESHOLD:
        emotion_scores["surprised"] = 0.0
    else:
        emotion_scores["fearful"] = 0.0
    return emotion_scores


def _dominant_emotion(
    emotion_scores: dict[str, float],
) -> tuple[EmotionLabel, float]:
    """Return the highest-scoring emotion and its score."""
    if not emotion_scores:
        return EmotionLabel.NEUTRAL, 0.0

    best_label, best_score = max(
        emotion_scores.items(), key=lambda kv: kv[1]
    )
    if best_score <= 0.0:
        return EmotionLabel.NEUTRAL, 0.0

    return EmotionLabel(best_label), best_score


def analyze(observation: VisionObservation) -> EmotionalState:
    """Convert a vision observation into a proxy emotional state.

    If no face is detected, returns face_available=False with neutral
    scores. Downstream consumers MUST check face_available before using
    the emotional state for any decision.
    """
    if not observation.face_available or observation.face_count == 0:
        return EmotionalState(
            face_available=False,
            dominant_emotion=EmotionLabel.NEUTRAL,
            confidence=0.0,
            emotion_scores={},
            activated_aus=[],
        )

    blendshapes = _blendshape_dict(observation.blendshapes)
    au_scores = _au_scores(blendshapes)
    activated_aus = sorted(
        au for au, score in au_scores.items() if score > _ACTIVATION_THRESHOLD
    )
    scores = _emotion_scores(au_scores)
    scores = _disambiguate_surprise_fear(scores, au_scores)

    dominant, confidence = _dominant_emotion(scores)

    return EmotionalState(
        face_available=True,
        dominant_emotion=dominant,
        confidence=round(confidence, 3),
        emotion_scores={k: round(v, 3) for k, v in scores.items()},
        activated_aus=activated_aus,
    )


def update_moving_average(
    history: list[EmotionalState],
    new_state: EmotionalState,
    *,
    window_size: int = 5,
) -> list[EmotionalState]:
    """Append a new state and return a trimmed window.

    The moving average is computed downstream from the returned list.
    This function only manages the window; it does not mutate the input.
    """
    if window_size < 1:
        window_size = 1
    updated = [*history, new_state]
    if len(updated) > window_size:
        updated = updated[-window_size:]
    return updated


def average_emotional_state(
    history: list[EmotionalState],
) -> EmotionalState:
    """Compute a moving-average emotional state from a window of samples.

    Only samples with face_available=True contribute. If no valid samples
    exist, returns a face_available=False neutral state.
    """
    valid = [s for s in history if s.face_available]
    if not valid:
        return EmotionalState(
            face_available=False,
            dominant_emotion=EmotionLabel.NEUTRAL,
            confidence=0.0,
            emotion_scores={},
            activated_aus=[],
        )

    all_emotions: set[str] = set()
    for s in valid:
        all_emotions.update(s.emotion_scores.keys())

    avg_scores: dict[str, float] = {}
    for emotion in all_emotions:
        values = [s.emotion_scores.get(emotion, 0.0) for s in valid]
        avg_scores[emotion] = round(sum(values) / len(values), 3)

    dominant, confidence = _dominant_emotion(avg_scores)

    all_aus: set[str] = set()
    for s in valid:
        all_aus.update(s.activated_aus)

    return EmotionalState(
        face_available=True,
        dominant_emotion=dominant,
        confidence=round(confidence, 3),
        emotion_scores=avg_scores,
        activated_aus=sorted(all_aus),
    )
