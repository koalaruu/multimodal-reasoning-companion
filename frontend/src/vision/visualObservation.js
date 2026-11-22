const MAX_VISIBLE_BLENDSHAPES = 3;

export function unavailableVisualObservation(permission = "idle", error = null) {
  return {
    availability: "unavailable",
    permission,
    cue: "not_enabled",
    face_count: 0,
    inference_ms: null,
    observed_at_ms: null,
    blendshapes: [],
    error,
  };
}

function roundedMilliseconds(value) {
  if (!Number.isFinite(value)) return null;
  return Math.round(Math.max(0, value) * 10) / 10;
}

function strongestBlendshapes(result, faceCount) {
  if (faceCount !== 1) return [];

  const categories = result?.faceBlendshapes?.[0]?.categories;
  if (!Array.isArray(categories)) return [];

  return categories
    .filter(
      (category) =>
        typeof category?.categoryName === "string" &&
        !category.categoryName.startsWith("_") &&
        Number.isFinite(category.score),
    )
    .sort((left, right) => right.score - left.score)
    .slice(0, MAX_VISIBLE_BLENDSHAPES)
    .map((category) => ({
      name: category.categoryName,
      score: Math.round(category.score * 1_000) / 1_000,
    }));
}

export function summarizeFaceLandmarkerResult(
  result,
  { inferenceMs = null, observedAtMs = null } = {},
) {
  const faceCount = Array.isArray(result?.faceLandmarks)
    ? result.faceLandmarks.length
    : 0;

  return {
    availability: "available",
    permission: "granted",
    cue:
      faceCount === 0
        ? "no_face"
        : faceCount === 1
          ? "face_detected"
          : "multiple_faces",
    face_count: faceCount,
    inference_ms: roundedMilliseconds(inferenceMs),
    observed_at_ms: Number.isFinite(observedAtMs) ? observedAtMs : null,
    blendshapes: strongestBlendshapes(result, faceCount),
    error: null,
  };
}
