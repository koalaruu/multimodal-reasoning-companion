const LANGUAGE_LABELS = {
  en: "English",
  id: "Bahasa Indonesia",
};

function formatDetectedLanguage(result) {
  const label = LANGUAGE_LABELS[result.language] ?? result.language;
  return `${label} (${Math.round(result.language_probability * 100)}%)`;
}

const VOICE_LABELS = {
  idle: "Voice answer ready",
  recording: "Recording voice answer",
  transcribing: "Transcribing voice answer",
  ready: "Transcript added to Your answer",
  error: "Voice transcription unavailable",
};

function formatDuration(milliseconds) {
  if (milliseconds == null) return "Not observed";
  return `${(milliseconds / 1_000).toFixed(1)} s`;
}

export default function AudioControls({
  observation,
  readiness,
  onEnable,
  onDisable,
  onStartRecording,
  onStopAndTranscribe,
  voiceDisabled,
  voiceError,
  voiceResult,
  voiceStatus,
}) {
  const enabled = observation.availability === "available";
  const requesting = observation.permission === "requesting";
  const recording = voiceStatus === "recording";
  const transcribing = voiceStatus === "transcribing";

  return (
    <section className="audio-panel" aria-labelledby="audio-heading">
      <div className="audio-summary">
        <div>
          <p className="eyebrow">Voice input</p>
          <h2 id="audio-heading">Speak your answer</h2>
        </div>
        <div className="audio-actions">
          <button
            className={enabled ? "secondary-button" : "primary-button"}
            type="button"
            onClick={enabled ? onDisable : onEnable}
            disabled={requesting}
          >
            {requesting
              ? "Requesting permission..."
              : enabled
                ? "Stop microphone"
                : "Enable microphone"}
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={recording ? onStopAndTranscribe : onStartRecording}
            disabled={
              !enabled || transcribing || (voiceDisabled && !recording)
            }
          >
            {recording
              ? "Stop and transcribe"
              : transcribing
                ? "Transcribing..."
                : "Start voice answer"}
          </button>
        </div>
      </div>

      <div className="audio-readout" aria-live="polite">
        <div className={`audio-state ${readiness.status}`}>
          <span className="audio-state-dot" />
          <div>
            <strong>{readiness.label}</strong>
            <small>
              {enabled
                ? "This cue never submits an answer automatically."
                : "Text interview controls remain available."}
            </small>
          </div>
        </div>
        <div className={`voice-state ${voiceStatus}`}>
          <strong>{VOICE_LABELS[voiceStatus] ?? VOICE_LABELS.idle}</strong>
          {voiceResult && (
            <small>
              {formatDuration(voiceResult.audio_duration_ms)} audio in{" "}
              {formatDuration(voiceResult.processing_duration_ms)} using{" "}
              {voiceResult.model}. Detected{" "}
              {formatDetectedLanguage(voiceResult)}.
            </small>
          )}
          {voiceResult && (
            <small className="stt-privacy-note">
              {voiceResult.model?.startsWith("groq/")
                ? "Audio was uploaded to Groq for transcription."
                : "Audio was transcribed on this device."}
            </small>
          )}
        </div>
        <div className="level-meter" aria-label="Current microphone activity">
          <span style={{ width: `${Math.round(observation.level * 100)}%` }} />
        </div>
        <details className="sensor-details">
          <summary>Voice activity details</summary>
        <dl className="audio-metrics">
          <div>
            <dt>Speech observed</dt>
            <dd>{observation.has_speech ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Speech duration</dt>
            <dd>{formatDuration(observation.speech_duration_ms)}</dd>
          </div>
          <div>
            <dt>Trailing silence</dt>
            <dd>{formatDuration(observation.trailing_silence_ms)}</dd>
          </div>
        </dl>
        </details>
      </div>
      <p className="observation-note">
        Review the transcript before sending. Recording never sends a chat message automatically.
      </p>

      {(observation.error || voiceError) && (
        <p className="audio-error" role="status">
          {observation.error || voiceError}
        </p>
      )}
    </section>
  );
}
