const API_BASE_URL = (
  import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function fetchOpeningQuestion({
  cvText,
  targetRole,
  jobDescription,
  interviewMode,
  signal,
}) {
  const response = await fetch(`${API_BASE_URL}/v1/interview/opening`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...(cvText ? { cv_text: cvText } : {}),
      ...(targetRole ? { target_role: targetRole } : {}),
      ...(jobDescription ? { job_description: jobDescription } : {}),
      ...(interviewMode ? { interview_mode: interviewMode } : {}),
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Opening question failed (${response.status}).`);
  }
  return response.json();
}

export async function fetchSessionReview({ state, signal }) {
  const response = await fetch(`${API_BASE_URL}/v1/interview/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Review failed (${response.status}).`);
  }
  return response.json();
}

export async function extractCv({ file, signal }) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${API_BASE_URL}/v1/cv/extract`, {
    method: "POST",
    body,
    signal,
  });

  if (!response.ok) {
    let message = `CV extraction failed (${response.status}).`;
    try {
      const payload = await response.json();
      if (payload?.detail?.message) message = payload.detail.message;
    } catch {
      // Keep the status-based message.
    }
    throw new Error(message);
  }

  return response.json();
}

export async function submitInterviewTurn({
  candidateInput,
  state,
  emotionalState,
  cvText,
  targetRole,
  jobDescription,
  interviewMode,
  turnReadiness,
  trailingSilenceMs,
  signal,
}) {
  const response = await fetch(`${API_BASE_URL}/v1/interview/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidate_input: candidateInput,
      state,
      ...(emotionalState ? { emotional_state: emotionalState } : {}),
      ...(cvText ? { cv_text: cvText } : {}),
      ...(targetRole ? { target_role: targetRole } : {}),
      ...(jobDescription ? { job_description: jobDescription } : {}),
      ...(interviewMode ? { interview_mode: interviewMode } : {}),
      ...(turnReadiness ? { turn_readiness: turnReadiness } : {}),
      ...(trailingSilenceMs != null
        ? { trailing_silence_ms: trailingSilenceMs }
        : {}),
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
