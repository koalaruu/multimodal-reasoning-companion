import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { analyzeVision, transcribeAudio } from "./interviewApi.js";

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
        text: "Saya membangun model klasifikasi.",
        language: "id",
        language_probability: 0.98,
        audio_duration_ms: 1200,
        processing_duration_ms: 300,
        model: "small",
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };

  const result = await transcribeAudio({ audio });

  assert.equal(request.url, "http://127.0.0.1:8000/v1/audio/transcriptions");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers["Content-Type"], "audio/webm;codecs=opus");
  assert.strictEqual(request.options.body, audio);
  assert.equal(result.text, "Saya membangun model klasifikasi.");
  assert.equal(result.language, "id");
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

test("vision analysis derives face_available from the observation shape", async () => {
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url, options };
    return new Response(
      JSON.stringify({
        emotional_state: {
          dominant_emotion: "neutral",
          confidence: 0,
          emotion_scores: {},
          activated_aus: [],
          face_available: false,
        },
        sample_count: 0,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };

  await analyzeVision({
    observation: {
      availability: "available",
      permission: "granted",
      cue: "no_face",
      face_count: 0,
      inference_ms: 17.1,
      observed_at_ms: 1795317614545,
      blendshapes: [],
      top_blendshapes: [],
      error: null,
    },
  });

  const body = JSON.parse(request.options.body);
  assert.equal(body.face_available, false);
  assert.equal(body.face_count, 0);
  assert.ok(!("top_blendshapes" in body));
  assert.ok(!("availability" in body));
});

test("vision analysis sends recent single-face readings in capture order", async () => {
  let request;
  globalThis.fetch = async (url, options) => {
    request = { url, options };
    return new Response(
      JSON.stringify({
        emotional_state: {
          dominant_emotion: "happy",
          confidence: 0.7,
          emotion_scores: { happy: 0.7 },
          activated_aus: ["AU6", "AU12"],
          face_available: true,
        },
        sample_count: 3,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  };

  const result = await analyzeVision({
    previousObservations: [1_795_317_612_545, 1_795_317_613_545].map((at) => ({
      availability: "available",
      face_count: 1,
      observed_at_ms: at,
      blendshapes: [{ name: "mouthSmileLeft", score: 0.6 }],
      top_blendshapes: [],
    })),
    observation: {
      availability: "available",
      permission: "granted",
      cue: "face_detected",
      face_count: 1,
      inference_ms: 12.5,
      observed_at_ms: 1795317614545,
      blendshapes: [{ name: "mouthSmileLeft", score: 0.8 }],
      top_blendshapes: [{ name: "mouthSmileLeft", score: 0.8 }],
      error: null,
    },
  });

  const body = JSON.parse(request.options.body);
  assert.equal(body.face_available, true);
  assert.deepEqual(body.blendshapes, [
    { name: "mouthSmileLeft", score: 0.8 },
  ]);
  assert.deepEqual(body.previous_observations, [
    {
      face_available: true,
      face_count: 1,
      observed_at_ms: 1_795_317_612_545,
      blendshapes: [{ name: "mouthSmileLeft", score: 0.6 }],
    },
    {
      face_available: true,
      face_count: 1,
      observed_at_ms: 1_795_317_613_545,
      blendshapes: [{ name: "mouthSmileLeft", score: 0.6 }],
    },
  ]);
  assert.equal(result.sample_count, 3);
});
