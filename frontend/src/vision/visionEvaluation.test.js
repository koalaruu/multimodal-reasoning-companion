import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  FACE_LANDMARKER_MAX_FACES,
  FACE_LANDMARKER_MODEL_URL,
} from "./visionRuntimeConfig.js";
import {
  buildVisionReport,
  percentile,
  summarizeVisionOutcomes,
  validateVisionDataset,
} from "./visionEvaluation.js";
const manifestUrl = new URL("../../../eval/vision_cases.json", import.meta.url);
const packageUrl = new URL("../../package.json", import.meta.url);
const manifest = JSON.parse(await readFile(fileURLToPath(manifestUrl), "utf8"));
const packageManifest = JSON.parse(
  await readFile(fileURLToPath(packageUrl), "utf8"),
);

function cloneManifest() {
  return structuredClone(manifest);
}

function completed(id, split, expected, detected, times = [10, 11, 12]) {
  return {
    id,
    split,
    expected_face_count: expected,
    status: "completed",
    failure: null,
    detected_face_count: detected,
    face_counts: times.map(() => detected),
    exact_match: expected === detected,
    stable: true,
    inference_ms: times,
    median_inference_ms: percentile(times, 0.5),
    p95_inference_ms: percentile(times, 0.95),
    landmark_counts: [],
  };
}

test("benchmark runtime matches the production package and model", () => {
  assert.equal(
    manifest.runtime.version,
    packageManifest.dependencies["@mediapipe/tasks-vision"],
  );
  assert.equal(manifest.runtime.model_url, FACE_LANDMARKER_MODEL_URL);
  assert.equal(manifest.runtime.max_faces, FACE_LANDMARKER_MAX_FACES);
});

test("vision manifest is valid and identity-disjoint", () => {
  assert.equal(validateVisionDataset(cloneManifest()).cases.length, 6);
  const calibrationIdentities = new Set(
    manifest.cases.filter((item) => item.split === "calibration").flatMap((item) => item.identity_ids),
  );
  const heldOutIdentities = manifest.cases
    .filter((item) => item.split === "held_out")
    .flatMap((item) => item.identity_ids);
  assert.equal(heldOutIdentities.some((identity) => calibrationIdentities.has(identity)), false);
});

test("identity leakage across splits is rejected", () => {
  const dataset = cloneManifest();
  dataset.cases.find((item) => item.id === "held_out_grace_hopper").identity_ids = ["barack_obama"];
  assert.throws(() => validateVisionDataset(dataset), /identity crosses calibration and held-out splits/);
});

test("unknown assets and non-HTTPS URLs are rejected", () => {
  const unknownAsset = cloneManifest();
  unknownAsset.cases[0].inputs[0].asset_id = "missing";
  assert.throws(() => validateVisionDataset(unknownAsset), /unknown asset/);

  const unsafeUrl = cloneManifest();
  unsafeUrl.assets[0].url = "http://example.com/portrait.jpg";
  assert.throws(() => validateVisionDataset(unsafeUrl), /URL must use HTTPS/);
});

test("nearest-rank percentile is deterministic", () => {
  assert.equal(percentile([10, 40, 20, 30], 0.5), 20);
  assert.equal(percentile([10, 40, 20, 30], 0.95), 40);
  assert.equal(percentile([], 0.95), null);
});

test("runtime failures cannot disappear from held-out accuracy", () => {
  const outcomes = [
    completed("one", "held_out", 1, 1),
    {
      ...completed("failure", "held_out", 1, 1),
      status: "failed",
      exact_match: false,
      stable: false,
      detected_face_count: null,
      inference_ms: [],
    },
  ];
  const summary = summarizeVisionOutcomes(outcomes, "held_out");
  assert.equal(summary.runtime_success_rate, 0.5);
  assert.equal(summary.face_count_accuracy, 0.5);
  assert.equal(summary.stability_rate, 0.5);
});

test("fixed target passes only with exact stable held-out outcomes", () => {
  const outcomes = manifest.cases.map((visionCase) =>
    completed(
      visionCase.id,
      visionCase.split,
      visionCase.expected_face_count,
      visionCase.expected_face_count,
      [20, 24, 22, 21, 23],
    ),
  );
  const passing = buildVisionReport(manifest, outcomes);
  assert.equal(passing.acceptance.passed, true);

  outcomes.find((item) => item.id === "held_out_burger_no_face").detected_face_count = 1;
  outcomes.find((item) => item.id === "held_out_burger_no_face").exact_match = false;
  const failing = buildVisionReport(manifest, outcomes);
  assert.equal(failing.acceptance.passed, false);
  assert.equal(failing.summary.held_out.no_face_false_positive_rate, 1);
});
