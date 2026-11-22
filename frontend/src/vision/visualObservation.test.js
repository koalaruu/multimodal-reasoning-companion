import assert from "node:assert/strict";
import test from "node:test";
import {
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
    error: null,
  });
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

test("one face exposes only the three strongest finite raw coefficients", () => {
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
  assert.deepEqual(observation.blendshapes, [
    { name: "eyeBlinkLeft", score: 0.91 },
    { name: "mouthSmileLeft", score: 0.8 },
    { name: "browDownLeft", score: 0.7 },
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
