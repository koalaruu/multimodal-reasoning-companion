import { FACE_LANDMARKER_MAX_FACES } from "./visionRuntimeConfig.js";

const SPLITS = new Set(["calibration", "held_out"]);
const LAYOUT_INPUT_COUNTS = { single: 1, horizontal: 2 };
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function assertCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function assertRate(value, name) {
  assertCondition(
    Number.isFinite(value) && value >= 0 && value <= 1,
    `${name} must be between 0 and 1`,
  );
}

function uniqueIds(items, label) {
  const ids = new Set();
  for (const item of items) {
    assertCondition(typeof item?.id === "string" && item.id.length > 0, `${label} id is required`);
    assertCondition(!ids.has(item.id), `duplicate ${label} id: ${item.id}`);
    ids.add(item.id);
  }
  return ids;
}

export function validateVisionDataset(dataset) {
  assertCondition(dataset?.schema_version === "vision-eval-v1", "unsupported vision schema version");
  assertCondition(Array.isArray(dataset.sources) && dataset.sources.length > 0, "sources must not be empty");
  assertCondition(Array.isArray(dataset.assets) && dataset.assets.length > 0, "assets must not be empty");
  assertCondition(Array.isArray(dataset.cases) && dataset.cases.length > 0, "cases must not be empty");

  const sourceIds = uniqueIds(dataset.sources, "source");
  for (const source of dataset.sources) {
    assertCondition(typeof source.license === "string" && source.license.length > 0, `source ${source.id} requires a license`);
    assertCondition(source.license_url?.startsWith("https://"), `source ${source.id} requires an HTTPS license URL`);
  }

  const runtime = dataset.runtime;
  assertCondition(runtime?.package === "@mediapipe/tasks-vision", "runtime package must match production");
  assertCondition(typeof runtime.version === "string" && runtime.version.length > 0, "runtime version is required");
  assertCondition(runtime.model_url?.startsWith("https://"), "model URL must use HTTPS");
  assertCondition(runtime.running_mode === "IMAGE", "benchmark running mode must be IMAGE");
  assertCondition(runtime.max_faces === FACE_LANDMARKER_MAX_FACES, `max_faces must match the production limit of ${FACE_LANDMARKER_MAX_FACES}`);
  assertCondition(Number.isInteger(runtime.warmup_repetitions) && runtime.warmup_repetitions >= 0, "warmup_repetitions must be a non-negative integer");
  assertCondition(Number.isInteger(runtime.measured_repetitions) && runtime.measured_repetitions >= 1, "measured_repetitions must be a positive integer");
  assertCondition(Number.isInteger(runtime.canvas_cell_size) && runtime.canvas_cell_size >= 128, "canvas_cell_size must be at least 128");

  const acceptance = dataset.acceptance;
  assertRate(acceptance?.min_held_out_runtime_success_rate, "min_held_out_runtime_success_rate");
  assertRate(acceptance?.min_held_out_face_count_accuracy, "min_held_out_face_count_accuracy");
  assertRate(acceptance?.max_held_out_no_face_false_positive_rate, "max_held_out_no_face_false_positive_rate");
  assertRate(acceptance?.min_held_out_face_detection_rate, "min_held_out_face_detection_rate");
  assertRate(acceptance?.min_held_out_stability_rate, "min_held_out_stability_rate");
  assertCondition(Number.isFinite(acceptance?.max_held_out_p95_inference_ms) && acceptance.max_held_out_p95_inference_ms > 0, "max_held_out_p95_inference_ms must be positive");

  const assetIds = uniqueIds(dataset.assets, "asset");
  for (const asset of dataset.assets) {
    assertCondition(sourceIds.has(asset.source_id), `asset ${asset.id} references an unknown source`);
    assertCondition(asset.url?.startsWith("https://"), `asset ${asset.id} URL must use HTTPS`);
    assertCondition(typeof asset.filename === "string" && !/[\\/]/.test(asset.filename), `asset ${asset.id} filename is unsafe`);
    assertCondition(SHA256_PATTERN.test(asset.sha256), `asset ${asset.id} requires a lowercase SHA-256`);
    assertCondition(Number.isInteger(asset.size_bytes) && asset.size_bytes > 0, `asset ${asset.id} size must be positive`);
    assertCondition(asset.media_type === "image/jpeg" || asset.media_type === "image/png", `asset ${asset.id} media type is unsupported`);
  }

  uniqueIds(dataset.cases, "case");
  const identitiesBySplit = { calibration: new Set(), held_out: new Set() };
  const expectedCountsBySplit = { calibration: new Set(), held_out: new Set() };
  for (const visionCase of dataset.cases) {
    assertCondition(SPLITS.has(visionCase.split), `case ${visionCase.id} has an invalid split`);
    const expectedInputs = LAYOUT_INPUT_COUNTS[visionCase.layout];
    assertCondition(expectedInputs != null, `case ${visionCase.id} has an invalid layout`);
    assertCondition(Array.isArray(visionCase.inputs) && visionCase.inputs.length === expectedInputs, `case ${visionCase.id} input count does not match its layout`);
    assertCondition(Number.isInteger(visionCase.expected_face_count) && visionCase.expected_face_count >= 0 && visionCase.expected_face_count <= runtime.max_faces, `case ${visionCase.id} expected face count is invalid`);
    assertCondition(Array.isArray(visionCase.identity_ids), `case ${visionCase.id} identity_ids must be an array`);
    assertCondition(new Set(visionCase.identity_ids).size === visionCase.identity_ids.length, `case ${visionCase.id} repeats an identity`);
    assertCondition(visionCase.identity_ids.length === visionCase.expected_face_count, `case ${visionCase.id} identity count must match expected faces`);
    for (const input of visionCase.inputs) {
      assertCondition(assetIds.has(input?.asset_id), `case ${visionCase.id} references an unknown asset`);
    }
    for (const identity of visionCase.identity_ids) {
      assertCondition(typeof identity === "string" && identity.length > 0, `case ${visionCase.id} has an invalid identity`);
      identitiesBySplit[visionCase.split].add(identity);
    }
    expectedCountsBySplit[visionCase.split].add(visionCase.expected_face_count);
  }

  assertCondition(dataset.cases.some((item) => item.split === "calibration"), "calibration split must not be empty");
  assertCondition(dataset.cases.some((item) => item.split === "held_out"), "held_out split must not be empty");
  for (const identity of identitiesBySplit.calibration) {
    assertCondition(!identitiesBySplit.held_out.has(identity), `identity crosses calibration and held-out splits: ${identity}`);
  }
  for (const expectedCount of [0, 1, 2]) {
    assertCondition(expectedCountsBySplit.held_out.has(expectedCount), `held-out split must include expected face count ${expectedCount}`);
  }

  return dataset;
}

export function percentile(values, quantile) {
  if (!Array.isArray(values) || values.length === 0) return null;
  assertCondition(Number.isFinite(quantile) && quantile >= 0 && quantile <= 1, "quantile must be between 0 and 1");
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(quantile * sorted.length) - 1);
  return Math.round(sorted[index] * 1_000) / 1_000;
}

function ratio(numerator, denominator) {
  return denominator === 0 ? null : Math.round((numerator / denominator) * 10_000) / 10_000;
}

export function summarizeVisionOutcomes(outcomes, split) {
  const selected = outcomes.filter((outcome) => outcome.split === split);
  const completed = selected.filter((outcome) => outcome.status === "completed");
  const noFace = selected.filter((outcome) => outcome.expected_face_count === 0);
  const withFaces = selected.filter((outcome) => outcome.expected_face_count > 0);
  const inferenceTimes = completed.flatMap((outcome) => outcome.inference_ms);

  return {
    total: selected.length,
    runtime_successful: completed.length,
    runtime_success_rate: ratio(completed.length, selected.length),
    face_count_correct: selected.filter((outcome) => outcome.exact_match).length,
    face_count_accuracy: ratio(selected.filter((outcome) => outcome.exact_match).length, selected.length),
    no_face_cases: noFace.length,
    no_face_false_positives: noFace.filter((outcome) => outcome.detected_face_count > 0).length,
    no_face_false_positive_rate: ratio(noFace.filter((outcome) => outcome.detected_face_count > 0).length, noFace.length),
    face_cases: withFaces.length,
    faces_detected: withFaces.filter((outcome) => outcome.detected_face_count > 0).length,
    face_detection_rate: ratio(withFaces.filter((outcome) => outcome.detected_face_count > 0).length, withFaces.length),
    stable_cases: selected.filter((outcome) => outcome.stable).length,
    stability_rate: ratio(selected.filter((outcome) => outcome.stable).length, selected.length),
    p50_inference_ms: percentile(inferenceTimes, 0.5),
    p95_inference_ms: percentile(inferenceTimes, 0.95),
  };
}

export function buildVisionReport(dataset, outcomes, runtimeMetadata = {}) {
  const summary = {
    calibration: summarizeVisionOutcomes(outcomes, "calibration"),
    held_out: summarizeVisionOutcomes(outcomes, "held_out"),
  };
  const target = dataset.acceptance;
  const actual = summary.held_out;
  const checks = {
    runtime_success_rate: actual.runtime_success_rate >= target.min_held_out_runtime_success_rate,
    face_count_accuracy: actual.face_count_accuracy >= target.min_held_out_face_count_accuracy,
    no_face_false_positive_rate: actual.no_face_false_positive_rate <= target.max_held_out_no_face_false_positive_rate,
    face_detection_rate: actual.face_detection_rate >= target.min_held_out_face_detection_rate,
    stability_rate: actual.stability_rate >= target.min_held_out_stability_rate,
    p95_inference_ms: actual.p95_inference_ms != null && actual.p95_inference_ms <= target.max_held_out_p95_inference_ms,
  };

  return {
    schema_version: "vision-eval-report-v1",
    generated_at: new Date().toISOString(),
    dataset_schema_version: dataset.schema_version,
    runtime: { ...dataset.runtime, ...runtimeMetadata },
    cases: outcomes,
    summary,
    acceptance: {
      passed: Object.values(checks).every(Boolean),
      targets: target,
      actual,
      checks,
    },
  };
}
