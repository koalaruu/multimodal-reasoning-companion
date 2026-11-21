import test from "node:test";
import assert from "node:assert/strict";
import {
  TURN_END_SILENCE_MS,
  deriveTurnReadiness,
} from "./turnReadiness.js";

function observation(overrides = {}) {
  return {
    availability: "available",
    has_speech: true,
    is_speaking: false,
    trailing_silence_ms: 0,
    ...overrides,
  };
}

test("unavailable audio never pretends the candidate is silent", () => {
  const result = deriveTurnReadiness(
    observation({ availability: "unavailable" }),
  );

  assert.equal(result.status, "unavailable");
  assert.equal(result.shouldWait, false);
});

test("active speech requires waiting", () => {
  const result = deriveTurnReadiness(observation({ is_speaking: true }));

  assert.equal(result.status, "speaking");
  assert.equal(result.shouldWait, true);
});

test("available microphone without observed speech keeps listening", () => {
  const result = deriveTurnReadiness(observation({ has_speech: false }));

  assert.equal(result.status, "listening");
  assert.equal(result.shouldWait, true);
});

test("short silence is treated as a thinking pause", () => {
  const result = deriveTurnReadiness(
    observation({ trailing_silence_ms: TURN_END_SILENCE_MS - 1 }),
  );

  assert.equal(result.status, "thinking_pause");
  assert.equal(result.shouldWait, true);
});

test("silence at the threshold marks the turn as potentially complete", () => {
  const result = deriveTurnReadiness(
    observation({ trailing_silence_ms: TURN_END_SILENCE_MS }),
  );

  assert.equal(result.status, "ready");
  assert.equal(result.shouldWait, false);
});
