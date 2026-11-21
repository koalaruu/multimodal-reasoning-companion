import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { transcribeAudio } from "./interviewApi.js";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("transcription sends the audio blob without JSON conversion", async () => {
  const audio = new Blob(["audio-bytes"], { type: "audio/webm;codecs=opus" });
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url, options };
    return new Response(
      JSON.stringify({
        text: "I built a classifier.",
        language: "en",
        language_probability: 0.98,
        audio_duration_ms: 1200,
        processing_duration_ms: 300,
        model: "base.en",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };

  const result = await transcribeAudio({ audio });

  assert.equal(request.url, "http://127.0.0.1:8000/v1/audio/transcriptions");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers["Content-Type"], "audio/webm;codecs=opus");
  assert.strictEqual(request.options.body, audio);
  assert.equal(result.text, "I built a classifier.");
});

test("transcription exposes the safe API error message", async () => {
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        detail: {
          code: "transcriber_unavailable",
          message: "Local transcription is unavailable.",
        },
      }),
      { status: 503, headers: { "Content-Type": "application/json" } },
    );

  await assert.rejects(
    transcribeAudio({
      audio: new Blob(["audio"], { type: "audio/webm" }),
    }),
    /Local transcription is unavailable/,
  );
});

test("transcription rejects malformed success payloads", async () => {
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ text: "missing metadata" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });

  await assert.rejects(
    transcribeAudio({
      audio: new Blob(["audio"], { type: "audio/webm" }),
    }),
    /invalid transcription response/,
  );
});
