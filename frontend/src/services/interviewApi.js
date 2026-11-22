const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function submitInterviewTurn({
  candidateInput,
  state,
  emotionalState,
  signal,
}) {
  const response = await fetch(`${API_BASE_URL}/v1/interview/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidate_input: candidateInput,
      state,
      ...(emotionalState ? { emotional_state: emotionalState } : {}),
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Interview service rejected the turn (${response.status}).`);
  }

  const result = await response.json();
  if (
    typeof result.reply !== "string" ||
    result.decision == null ||
    result.updated_state == null
  ) {
    throw new Error("Interview service returned an invalid response.");
  }
  return result;
}

export async function transcribeAudio({ audio, signal }) {
  const response = await fetch(`${API_BASE_URL}/v1/audio/transcriptions`, {
    method: "POST",
    headers: { "Content-Type": audio.type || "audio/webm" },
    body: audio,
    signal,
  });

  if (!response.ok) {
    let message = `Transcription service rejected the audio (${response.status}).`;
    try {
      const errorBody = await response.json();
      if (typeof errorBody?.detail?.message === "string") {
        message = errorBody.detail.message;
      }
    } catch {
      // Keep the status-only fallback for non-JSON failures.
    }
    throw new Error(message);
  }

  const result = await response.json();
  if (
    typeof result.text !== "string" ||
    typeof result.language !== "string" ||
    typeof result.language_probability !== "number" ||
    typeof result.audio_duration_ms !== "number" ||
    typeof result.processing_duration_ms !== "number" ||
    typeof result.model !== "string"
  ) {
    throw new Error("Transcription service returned an invalid transcription response.");
  }
  return result;
}

export async function analyzeVision({ observation, signal }) {
  const response = await fetch(`${API_BASE_URL}/v1/vision/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      face_available:
        observation.face_available ??
        (observation.availability === "available" &&
          observation.face_count > 0),
      face_count: observation.face_count,
      inference_ms: observation.inference_ms,
      observed_at_ms: observation.observed_at_ms,
      blendshapes: observation.blendshapes,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Vision analysis failed (${response.status}).`);
  }

  const result = await response.json();
  if (result.emotional_state == null) {
    throw new Error("Vision service returned an invalid response.");
  }
  return result.emotional_state;
}
