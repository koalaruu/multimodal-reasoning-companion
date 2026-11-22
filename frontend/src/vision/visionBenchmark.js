import { createLocalFaceLandmarker } from "./faceLandmarkerRuntime.js";
import { summarizeFaceLandmarkerResult } from "./visualObservation.js";
import { buildVisionReport, validateVisionDataset } from "./visionEvaluation.js";

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function loadAsset(asset) {
  try {
    const response = await fetch(asset.url, { mode: "cors" });
    if (!response.ok) return { failure: "download_failed" };
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== asset.size_bytes) return { failure: "size_mismatch" };
    if ((await sha256Hex(buffer)) !== asset.sha256) return { failure: "checksum_mismatch" };
    const bitmap = await createImageBitmap(new Blob([buffer], { type: asset.media_type }));
    return { bitmap };
  } catch {
    return { failure: "download_failed" };
  }
}

function drawContained(context, bitmap, left, top, width, height) {
  const scale = Math.min(width / bitmap.width, height / bitmap.height);
  const drawWidth = bitmap.width * scale;
  const drawHeight = bitmap.height * scale;
  context.drawImage(
    bitmap,
    left + (width - drawWidth) / 2,
    top + (height - drawHeight) / 2,
    drawWidth,
    drawHeight,
  );
}

function caseInput(visionCase, assets, cellSize) {
  const inputs = visionCase.inputs.map((input) => assets.get(input.asset_id)?.bitmap);
  if (inputs.some((bitmap) => !bitmap)) return null;
  if (visionCase.layout === "single") return inputs[0];

  const canvas = document.createElement("canvas");
  canvas.width = cellSize * inputs.length;
  canvas.height = cellSize;
  const context = canvas.getContext("2d", { alpha: false });
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  inputs.forEach((bitmap, index) => {
    drawContained(context, bitmap, index * cellSize, 0, cellSize, cellSize);
  });
  return canvas;
}

function mostFrequent(values) {
  const frequencies = new Map();
  for (const value of values) frequencies.set(value, (frequencies.get(value) ?? 0) + 1);
  return [...frequencies.entries()].sort(
    ([leftValue, leftCount], [rightValue, rightCount]) =>
      rightCount - leftCount || leftValue - rightValue,
  )[0]?.[0] ?? null;
}

function failedOutcome(visionCase, failure) {
  return {
    id: visionCase.id,
    split: visionCase.split,
    expected_face_count: visionCase.expected_face_count,
    status: "failed",
    failure,
    detected_face_count: null,
    face_counts: [],
    exact_match: false,
    stable: false,
    inference_ms: [],
    median_inference_ms: null,
    p95_inference_ms: null,
    landmark_counts: [],
  };
}

function nearestRank(values, quantile) {
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(quantile * sorted.length) - 1);
  return Math.round(sorted[index] * 1_000) / 1_000;
}

function evaluateCase(landmarker, visionCase, input, runtime) {
  try {
    for (let index = 0; index < runtime.warmup_repetitions; index += 1) {
      landmarker.detect(input);
    }

    const faceCounts = [];
    const inferenceTimes = [];
    const landmarkCounts = [];
    for (let index = 0; index < runtime.measured_repetitions; index += 1) {
      const startedAt = performance.now();
      const result = landmarker.detect(input);
      inferenceTimes.push(Math.round((performance.now() - startedAt) * 1_000) / 1_000);
      const observation = summarizeFaceLandmarkerResult(result);
      faceCounts.push(observation.face_count);
      landmarkCounts.push(result.faceLandmarks.map((landmarks) => landmarks.length));
    }

    const detectedFaceCount = mostFrequent(faceCounts);
    const stable = new Set(faceCounts).size === 1;
    return {
      id: visionCase.id,
      split: visionCase.split,
      expected_face_count: visionCase.expected_face_count,
      status: "completed",
      failure: null,
      detected_face_count: detectedFaceCount,
      face_counts: faceCounts,
      exact_match: stable && detectedFaceCount === visionCase.expected_face_count,
      stable,
      inference_ms: inferenceTimes,
      median_inference_ms: nearestRank(inferenceTimes, 0.5),
      p95_inference_ms: nearestRank(inferenceTimes, 0.95),
      landmark_counts: landmarkCounts,
    };
  } catch {
    return failedOutcome(visionCase, "inference_failed");
  }
}

export async function runVisionBenchmark(rawDataset) {
  const dataset = validateVisionDataset(rawDataset);
  const assets = new Map();
  await Promise.all(
    dataset.assets.map(async (asset) => {
      assets.set(asset.id, await loadAsset(asset));
    }),
  );

  let landmarker = null;
  let model_load_ms = null;
  const outcomes = [];
  try {
    const modelStartedAt = performance.now();
    try {
      landmarker = await createLocalFaceLandmarker({ runningMode: "IMAGE" });
      model_load_ms = Math.round((performance.now() - modelStartedAt) * 1_000) / 1_000;
    } catch {
      for (const visionCase of dataset.cases) {
        outcomes.push(failedOutcome(visionCase, "model_unavailable"));
      }
      return buildVisionReport(dataset, outcomes, {
        model_load_ms,
        user_agent: navigator.userAgent,
        hardware_concurrency: navigator.hardwareConcurrency ?? null,
      });
    }

    for (const visionCase of dataset.cases) {
      const missingAsset = visionCase.inputs
        .map((input) => assets.get(input.asset_id))
        .find((asset) => asset?.failure);
      if (missingAsset) {
        outcomes.push(failedOutcome(visionCase, missingAsset.failure));
        continue;
      }
      const input = caseInput(visionCase, assets, dataset.runtime.canvas_cell_size);
      outcomes.push(
        input
          ? evaluateCase(landmarker, visionCase, input, dataset.runtime)
          : failedOutcome(visionCase, "asset_unavailable"),
      );
    }

    return buildVisionReport(dataset, outcomes, {
      model_load_ms,
      user_agent: navigator.userAgent,
      hardware_concurrency: navigator.hardwareConcurrency ?? null,
    });
  } finally {
    for (const asset of assets.values()) asset.bitmap?.close();
    try {
      landmarker?.close();
    } catch {
      // Completed measurements remain valid after a runtime cleanup failure.
    }
  }
}
