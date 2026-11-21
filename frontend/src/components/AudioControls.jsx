function formatDuration(milliseconds) {
  if (milliseconds == null) return "Not observed";
  return `${(milliseconds / 1_000).toFixed(1)} s`;
}

export default function AudioControls({
  observation,
  readiness,
  onEnable,
  onDisable,
}) {
  const enabled = observation.availability === "available";
  const requesting = observation.permission === "requesting";

  return (
    <section className="audio-panel" aria-labelledby="audio-heading">
      <div className="audio-summary">
        <div>
          <p className="eyebrow">Optional audio perception</p>
          <h2 id="audio-heading">Turn-taking cue</h2>
          <p>
            Local voice activity only. Audio is not transcribed, uploaded, or
            stored. Manual Send remains authoritative.
          </p>
        </div>
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

      {observation.error && (
        <p className="audio-error" role="status">
          {observation.error}
        </p>
      )}
    </section>
  );
}
