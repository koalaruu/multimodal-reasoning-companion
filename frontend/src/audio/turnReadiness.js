export const TURN_END_SILENCE_MS = 1_600;

export function deriveTurnReadiness(
  observation,
  silenceThresholdMs = TURN_END_SILENCE_MS,
) {
  if (observation.availability !== "available") {
    return {
      status: "unavailable",
      shouldWait: false,
      label: "Manual turn control",
    };
  }
  if (observation.is_speaking) {
    return {
      status: "speaking",
      shouldWait: true,
      label: "Candidate is speaking",
    };
  }
  if (!observation.has_speech) {
    return {
      status: "listening",
      shouldWait: true,
      label: "Listening for speech",
    };
  }
  if (observation.trailing_silence_ms < silenceThresholdMs) {
    return {
      status: "thinking_pause",
      shouldWait: true,
      label: "Short pause — keep waiting",
    };
  }
  return {
    status: "ready",
    shouldWait: false,
    label: "Turn may be complete",
  };
}
