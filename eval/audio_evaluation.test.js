import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import path from "node:path";
import {
  buildAudioReport,
  evaluateThreshold,
  loadAudioDataset,
  simulateCase,
} from "./audio_evaluation.js";

const directory = path.dirname(fileURLToPath(import.meta.url));
const datasetPath = path.join(directory, "audio_cases.json");

const dataset = await loadAudioDataset(datasetPath);

test("calibration selects the lowest-delay threshold without false-ready cases", () => {
  const report = buildAudioReport(dataset);

  assert.equal(report.selected_threshold_ms, 1600);
});

test("held-out acceptance reports the known edge case without hiding it", () => {
  const report = buildAudioReport(dataset);

  assert.equal(report.selected_held_out_summary.false_ready_count, 1);
  assert.equal(report.selected_held_out_summary.false_ready_rate, 0.25);
  assert.equal(report.selected_held_out_summary.mean_detection_delay_ms, 1800);
  assert.equal(report.acceptance.passed, true);
});

test("a 1700 ms held-out pause exposes the 1600 ms heuristic limitation", () => {
  const edgeCase = dataset.cases.find(
    (testCase) => testCase.id === "held_out_pause_1700ms",
  );
  const outcome = simulateCase(edgeCase, 1600, dataset.step_ms);

  assert.equal(outcome.false_ready, true);
  assert.equal(outcome.first_false_ready_at_ms, 3600);
  assert.equal(outcome.passed, false);
});

test("unavailable microphone is never interpreted as silence or readiness", () => {
  const unavailableCase = dataset.cases.find(
    (testCase) => testCase.id === "held_out_microphone_unavailable",
  );
  const outcome = simulateCase(unavailableCase, 1600, dataset.step_ms);

  assert.equal(outcome.availability, "unavailable");
  assert.equal(outcome.false_ready, false);
  assert.equal(outcome.unavailable_correct, true);
  assert.equal(outcome.passed, true);
});

test("no observed speech never becomes ready", () => {
  const noSpeechCase = dataset.cases.find(
    (testCase) => testCase.id === "held_out_no_speech",
  );
  const outcome = simulateCase(noSpeechCase, 1600, dataset.step_ms);

  assert.equal(outcome.had_speech, false);
  assert.equal(outcome.first_ready_after_end_ms, null);
  assert.equal(outcome.false_ready, false);
  assert.equal(outcome.passed, true);
});

test("a trace ending before the threshold lowers completion detection rate", () => {
  const result = evaluateThreshold(
    [
      {
        id: "truncated_completed_turn",
        split: "held_out",
        availability: "available",
        duration_ms: 2000,
        events: [
          { at_ms: 0, speaking: true },
          { at_ms: 1000, speaking: false },
        ],
        ground_truth_turn_end_ms: 1000,
      },
    ],
    1600,
    100,
  );

  assert.equal(result.summary.completed_turns_detected, 0);
  assert.equal(result.summary.completed_turn_detection_rate, 0);
});
