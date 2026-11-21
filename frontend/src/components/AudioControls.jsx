const VOICE_LABELS = {
  idle: "Voice answer ready",
  recording: "Recording voice answer",
  transcribing: "Transcribing locally",
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
          <p className="eyebrow">Optional local audio</p>
          <h2 id="audio-heading">Turn-taking and voice answer</h2>
          <p>
            Voice activity stays in the browser. An explicit voice recording is
            sent only to local FastAPI, transcribed in memory, and never submits
            an answer automatically.
          </p>
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
              {voiceResult.model}
            </small>
          )}
        </div>
        <div className="level-meter" aria-label="Current microphone activity">
          <span style={{ width: `${Math.round(observation.level * 100)}%` }} />
        </div>
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
      </div>

      {(observation.error || voiceError) && (
        <p className="audio-error" role="status">
          {observation.error || voiceError}
        </p>
      )}
    </section>
  );
}
