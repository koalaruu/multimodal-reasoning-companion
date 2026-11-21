import { readFile } from "node:fs/promises";
import { deriveTurnReadiness } from "../frontend/src/audio/turnReadiness.js";

const SPLITS = new Set(["calibration", "held_out"]);
const AVAILABILITY = new Set(["available", "unavailable"]);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export async function loadAudioDataset(path) {
  const dataset = JSON.parse(await readFile(path, "utf8"));
  validateDataset(dataset);
  return dataset;
}

export function validateDataset(dataset) {
  requireCondition(
    dataset.schema_version === "audio-cases-v1",
    'schema_version must be "audio-cases-v1"',
  );
  requireCondition(
    Number.isInteger(dataset.step_ms) && dataset.step_ms > 0,
    "step_ms must be a positive integer",
  );
  requireCondition(
    Array.isArray(dataset.thresholds_ms) && dataset.thresholds_ms.length > 0,
    "thresholds_ms must be a non-empty array",
  );
  dataset.thresholds_ms.forEach((threshold) => {
    requireCondition(
      Number.isInteger(threshold) && threshold > 0,
      "every threshold must be a positive integer",
    );
  });
  requireCondition(
    Array.isArray(dataset.cases) && dataset.cases.length > 0,
    "cases must be a non-empty array",
  );

  const ids = new Set();
  for (const testCase of dataset.cases) {
    requireCondition(
      typeof testCase.id === "string" && testCase.id.length > 0,
      "every case needs an id",
    );
    requireCondition(!ids.has(testCase.id), `duplicate case id: ${testCase.id}`);
    ids.add(testCase.id);
    requireCondition(SPLITS.has(testCase.split), `${testCase.id}: invalid split`);
    requireCondition(
      AVAILABILITY.has(testCase.availability),
      `${testCase.id}: invalid availability`,
    );
    requireCondition(
      Number.isInteger(testCase.duration_ms) &&
        testCase.duration_ms > 0 &&
        testCase.duration_ms % dataset.step_ms === 0,
      `${testCase.id}: duration must align with step_ms`,
    );
    requireCondition(Array.isArray(testCase.events), `${testCase.id}: events must be an array`);

    let previousAt = -1;
    for (const event of testCase.events) {
      requireCondition(
        Number.isInteger(event.at_ms) &&
          event.at_ms >= 0 &&
          event.at_ms <= testCase.duration_ms &&
          event.at_ms % dataset.step_ms === 0,
        `${testCase.id}: event time must align with step_ms`,
      );
      requireCondition(
        event.at_ms > previousAt,
        `${testCase.id}: events must be strictly ordered`,
      );
      requireCondition(
        typeof event.speaking === "boolean",
        `${testCase.id}: speaking must be boolean`,
      );
      previousAt = event.at_ms;
    }
    if (testCase.availability === "unavailable") {
      requireCondition(
        testCase.events.length === 0,
        `${testCase.id}: unavailable audio cannot contain events`,
      );
    }

    const turnEnd = testCase.ground_truth_turn_end_ms;
    requireCondition(
      turnEnd === null ||
        (Number.isInteger(turnEnd) &&
          turnEnd >= 0 &&
          turnEnd <= testCase.duration_ms &&
          turnEnd % dataset.step_ms === 0),
      `${testCase.id}: invalid ground_truth_turn_end_ms`,
    );
    if (turnEnd !== null) {
      requireCondition(
        testCase.events.some(
          (event) => event.at_ms === turnEnd && event.speaking === false,
        ),
        `${testCase.id}: ground-truth end must match a speech-stop event`,
      );
    }
  }

  const acceptance = dataset.acceptance;
  requireCondition(
    acceptance &&
      acceptance.max_false_ready_rate >= 0 &&
      acceptance.max_false_ready_rate <= 1,
    "acceptance.max_false_ready_rate must be between 0 and 1",
  );
  requireCondition(
    Number.isFinite(acceptance.max_mean_detection_delay_ms) &&
      acceptance.max_mean_detection_delay_ms >= 0,
    "acceptance.max_mean_detection_delay_ms must be non-negative",
  );
}

export function simulateCase(testCase, thresholdMs, stepMs) {
  if (testCase.availability === "unavailable") {
    return {
      id: testCase.id,
      split: testCase.split,
      availability: testCase.availability,
      ground_truth_turn_end_ms: testCase.ground_truth_turn_end_ms,
      had_speech: false,
      false_ready: false,
      first_false_ready_at_ms: null,
      first_ready_after_end_ms: null,
      detection_delay_ms: null,
      unavailable_correct: true,
      passed: true,
    };
  }

  let speaking = false;
  let hasSpeech = false;
  let lastSpeechAt = null;
  let eventIndex = 0;
  let falseReady = false;
  let firstFalseReadyAt = null;
  let firstReadyAfterEnd = null;

  for (let atMs = 0; atMs <= testCase.duration_ms; atMs += stepMs) {
    while (
      eventIndex < testCase.events.length &&
      testCase.events[eventIndex].at_ms === atMs
    ) {
      speaking = testCase.events[eventIndex].speaking;
      if (speaking) hasSpeech = true;
      lastSpeechAt = hasSpeech ? atMs : null;
      eventIndex += 1;
    }
    if (speaking) {
      hasSpeech = true;
      lastSpeechAt = atMs;
    }

    const observation = {
      availability: "available",
      has_speech: hasSpeech,
      is_speaking: speaking,
      trailing_silence_ms:
        hasSpeech && !speaking ? atMs - lastSpeechAt : speaking ? 0 : null,
    };
    const readiness = deriveTurnReadiness(observation, thresholdMs);
    if (readiness.status !== "ready") continue;

    const turnEnd = testCase.ground_truth_turn_end_ms;
    if (turnEnd === null || atMs < turnEnd) {
      falseReady = true;
      if (firstFalseReadyAt === null) firstFalseReadyAt = atMs;
    } else if (firstReadyAfterEnd === null) {
      firstReadyAfterEnd = atMs;
    }
  }

  const turnEnd = testCase.ground_truth_turn_end_ms;
  const detectionDelay =
    turnEnd !== null && firstReadyAfterEnd !== null
      ? firstReadyAfterEnd - turnEnd
      : null;
  const passed =
    turnEnd === null
      ? !falseReady
      : !falseReady && firstReadyAfterEnd !== null;

  return {
    id: testCase.id,
    split: testCase.split,
    availability: testCase.availability,
    ground_truth_turn_end_ms: testCase.ground_truth_turn_end_ms,
    had_speech: hasSpeech,
    false_ready: falseReady,
    first_false_ready_at_ms: firstFalseReadyAt,
    first_ready_after_end_ms: firstReadyAfterEnd,
    detection_delay_ms: detectionDelay,
    unavailable_correct: null,
    passed,
  };
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarizeOutcomes(outcomes) {
  const riskCases = outcomes.filter(
    (outcome) => outcome.availability === "available" && outcome.had_speech,
  );
  const completedTurns = outcomes.filter(
    (outcome) => outcome.detection_delay_ms !== null,
  );
  const expectedCompletedTurns = outcomes.filter(
    (outcome) =>
      outcome.availability === "available" &&
      outcome.ground_truth_turn_end_ms !== null,
  );
  const unavailableCases = outcomes.filter(
    (outcome) => outcome.availability === "unavailable",
  );
  const falseReadyCount = riskCases.filter(
    (outcome) => outcome.false_ready,
  ).length;
  const delays = completedTurns.map((outcome) => outcome.detection_delay_ms);

  return {
    total_cases: outcomes.length,
    passed_cases: outcomes.filter((outcome) => outcome.passed).length,
    false_ready_count: falseReadyCount,
    false_ready_rate:
      riskCases.length === 0 ? 0 : round(falseReadyCount / riskCases.length),
    completed_turns_detected: completedTurns.length,
    completed_turn_detection_rate:
      expectedCompletedTurns.length === 0
        ? 1
        : round(completedTurns.length / expectedCompletedTurns.length),
    mean_detection_delay_ms:
      delays.length === 0
        ? null
        : round(delays.reduce((sum, delay) => sum + delay, 0) / delays.length, 1),
    median_detection_delay_ms: median(delays),
    unavailable_correct: unavailableCases.every(
      (outcome) => outcome.unavailable_correct,
    ),
  };
}

export function evaluateThreshold(cases, thresholdMs, stepMs) {
  const outcomes = cases.map((testCase) =>
    simulateCase(testCase, thresholdMs, stepMs),
  );
  return {
    threshold_ms: thresholdMs,
    summary: summarizeOutcomes(outcomes),
    cases: outcomes,
  };
}

export function selectThreshold(calibrationResults) {
  return [...calibrationResults].sort((left, right) => {
    const falseReadyDifference =
      left.summary.false_ready_count - right.summary.false_ready_count;
    if (falseReadyDifference !== 0) return falseReadyDifference;
    const leftDelay = left.summary.mean_detection_delay_ms ?? Infinity;
    const rightDelay = right.summary.mean_detection_delay_ms ?? Infinity;
    if (leftDelay !== rightDelay) return leftDelay - rightDelay;
    return left.threshold_ms - right.threshold_ms;
  })[0].threshold_ms;
}

export function buildAudioReport(dataset) {
  const calibrationCases = dataset.cases.filter(
    (testCase) => testCase.split === "calibration",
  );
  const heldOutCases = dataset.cases.filter(
    (testCase) => testCase.split === "held_out",
  );
  const thresholdResults = dataset.thresholds_ms.map((threshold) => ({
    threshold_ms: threshold,
    calibration: evaluateThreshold(
      calibrationCases,
      threshold,
      dataset.step_ms,
    ).summary,
    held_out: evaluateThreshold(heldOutCases, threshold, dataset.step_ms).summary,
  }));
  const calibrationResults = dataset.thresholds_ms.map((threshold) =>
    evaluateThreshold(calibrationCases, threshold, dataset.step_ms),
  );
  const selectedThreshold = selectThreshold(calibrationResults);
  const selectedHeldOut = evaluateThreshold(
    heldOutCases,
    selectedThreshold,
    dataset.step_ms,
  );
  const acceptance = {
    false_ready_rate_within_limit:
      selectedHeldOut.summary.false_ready_rate <=
      dataset.acceptance.max_false_ready_rate,
    detection_delay_within_limit:
      selectedHeldOut.summary.mean_detection_delay_ms <=
      dataset.acceptance.max_mean_detection_delay_ms,
    all_completed_turns_detected:
      selectedHeldOut.summary.completed_turn_detection_rate === 1,
    unavailable_handled_correctly: selectedHeldOut.summary.unavailable_correct,
  };
  acceptance.passed = Object.values(acceptance).every(Boolean);

  return {
    schema_version: "audio-eval-v1",
    selection_rule:
      "minimize calibration false-ready count, then mean detection delay",
    selected_threshold_ms: selectedThreshold,
    acceptance,
    selected_held_out_summary: selectedHeldOut.summary,
    selected_held_out_cases: selectedHeldOut.cases,
    threshold_results: thresholdResults,
  };
}
