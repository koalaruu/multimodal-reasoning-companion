import assert from "node:assert/strict";
import test from "node:test";
import {
  freshEmotionalState,
  isCameraActive,
  summarizeFaceLandmarkerResult,
  unavailableVisualObservation,
} from "./visualObservation.js";

test("disabled vision never claims that a face was observed", () => {
  assert.deepEqual(unavailableVisualObservation(), {
    availability: "unavailable",
    permission: "idle",
    cue: "not_enabled",
    face_count: 0,
    inference_ms: null,
    observed_at_ms: null,
    blendshapes: [],
    top_blendshapes: [],
    error: null,
  });
});

test("a released camera is retryable even when browser permission remains granted", () => {
  const failedObservation = unavailableVisualObservation(
    "granted",
    "Model loading failed.",
  );

  assert.equal(isCameraActive(failedObservation), false);
  assert.equal(
    isCameraActive({
      ...failedObservation,
      cue: "loading_model",
    }),
    true,
  );
});

test("an empty MediaPipe result reports no face", () => {
  const observation = summarizeFaceLandmarkerResult(
    { faceLandmarks: [], faceBlendshapes: [] },
    { inferenceMs: 12.46, observedAtMs: 1_000 },
  );

  assert.equal(observation.cue, "no_face");
  assert.equal(observation.face_count, 0);
  assert.equal(observation.inference_ms, 12.5);
  assert.equal(observation.observed_at_ms, 1_000);
});

test("one face exposes the strongest finite raw coefficients", () => {
  const observation = summarizeFaceLandmarkerResult({
    faceLandmarks: [[{ x: 0.5, y: 0.5, z: 0 }]],
    faceBlendshapes: [
      {
        categories: [
          { categoryName: "jawOpen", score: 0.45 },
          { categoryName: "eyeBlinkLeft", score: 0.91 },
          { categoryName: "browDownLeft", score: 0.7 },
          { categoryName: "mouthSmileLeft", score: 0.8 },
          { categoryName: "_neutral", score: 0.99 },
          { categoryName: "invalid", score: Number.NaN },
        ],
      },
    ],
  });

  assert.equal(observation.cue, "face_detected");
  assert.deepEqual(observation.top_blendshapes, [
    { name: "eyeBlinkLeft", score: 0.91 },
    { name: "mouthSmileLeft", score: 0.8 },
    { name: "browDownLeft", score: 0.7 },
    { name: "jawOpen", score: 0.45 },
  ]);
});

test("multiple faces suppress coefficients to avoid ambiguous attribution", () => {
  const observation = summarizeFaceLandmarkerResult({
    faceLandmarks: [[{}], [{}]],
    faceBlendshapes: [
      { categories: [{ categoryName: "jawOpen", score: 0.9 }] },
      { categories: [{ categoryName: "eyeBlinkLeft", score: 0.8 }] },
    ],
  });

  assert.equal(observation.cue, "multiple_faces");
  assert.equal(observation.face_count, 2);
  assert.deepEqual(observation.blendshapes, []);
});

test("invalid timing metadata is never reported as measured latency", () => {
  const observation = summarizeFaceLandmarkerResult(
    { faceLandmarks: [], faceBlendshapes: [] },
    { inferenceMs: Number.NaN, observedAtMs: Number.POSITIVE_INFINITY },
  );

  assert.equal(observation.inference_ms, null);
  assert.equal(observation.observed_at_ms, null);
});

test("recent facial evidence is returned unchanged", () => {
  const value = { dominant_emotion: "fearful", confidence: 0.6 };
  const entry = { value, atMs: 1_000 };

  assert.deepEqual(freshEmotionalState(entry, 3_000, 4_000), value);
});

test("facial evidence older than the limit is treated as absent", () => {
  const entry = { value: { dominant_emotion: "fearful" }, atMs: 1_000 };

  assert.equal(freshEmotionalState(entry, 6_000, 4_000), null);
});

test("missing or untimestamped facial evidence is absent, not neutral", () => {
  assert.equal(freshEmotionalState(null), null);
  assert.equal(freshEmotionalState({ value: {}, atMs: null }), null);
  assert.equal(freshEmotionalState({ value: null, atMs: 1_000 }), null);
});
