import { isCameraActive } from "../vision/visualObservation.js";

const CUE_LABELS = {
  not_enabled: "Camera not enabled",
  loading_model: "Loading local face model",
  no_face: "No face detected",
  face_detected: "One face detected",
  multiple_faces: "Multiple faces detected",
};

function formatInference(milliseconds) {
  return milliseconds == null ? "Not observed" : `${milliseconds.toFixed(1)} ms`;
}

export default function VisualControls({
  observation,
  emotionalState,
  videoRef,
  onEnable,
  onDisable,
}) {
  const cameraOn = isCameraActive(observation);
  const requesting = observation.permission === "requesting";
  const modelReady = observation.availability === "available";

  return (
    <section className="vision-panel" aria-labelledby="vision-heading">
      <div className="vision-summary">
        <div>
          <p className="eyebrow">Local perception</p>
          <h2 id="vision-heading">Camera &amp; expression</h2>
        </div>
        <button
          className={cameraOn ? "secondary-button" : "primary-button"}
          type="button"
          onClick={cameraOn ? onDisable : onEnable}
          disabled={requesting}
        >
          {requesting
            ? "Requesting permission..."
            : cameraOn
              ? "Stop camera"
              : "Enable camera"}
        </button>
      </div>

      <div className="vision-content">
        <div className="camera-preview">
          <video ref={videoRef} autoPlay muted playsInline aria-label="Local camera preview" />
          {!cameraOn && (
            <div className="camera-placeholder">
              <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <rect x="8" y="14" width="26" height="22" rx="5" />
                <path d="m34 21 7-4v16l-7-4 M18 14l3-5h8l3 5" />
              </svg>
              <strong>Your camera is off</strong>
              <span>Enable it for optional facial cues</span>
            </div>
          )}
        </div>

        <div className="vision-readout" aria-live="polite">
          <div className={`vision-state ${observation.cue}`}>
            <span className="vision-state-dot" />
            <div>
              <strong>
                {CUE_LABELS[observation.cue] ?? "Visual observation unavailable"}
              </strong>
              <small>
                {modelReady
                  ? "Advisory facial cues, not a mental-state assessment."
                  : "Text and audio remain available."}
              </small>
            </div>
          </div>

          <details className="sensor-details">
            <summary>Landmarks &amp; inference details</summary>
          <dl className="vision-metrics">
            <div>
              <dt>Faces</dt>
              <dd>{modelReady ? observation.face_count : "Not observed"}</dd>
            </div>
            <div>
              <dt>Inference</dt>
              <dd>{formatInference(observation.inference_ms)}</dd>
            </div>
          </dl>

          <div className="blendshape-summary">
            <h3>Strongest raw coefficients</h3>
            {observation.top_blendshapes?.length > 0 ? (
              <ul>
                {observation.top_blendshapes.map((blendshape) => (
                  <li key={blendshape.name}>
                    <span>{blendshape.name}</span>
                    <strong>{blendshape.score.toFixed(3)}</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p>Available after exactly one face is detected.</p>
            )}
          </div>
          </details>

          {emotionalState && emotionalState.face_available && (
            <div className="emotion-readout">
              <h3>Proxy emotional state</h3>
              <div className="emotion-dominant">
                <span className="emotion-label">
                  {emotionalState.dominant_emotion}
                </span>
                <small>
                  confidence {(emotionalState.confidence * 100).toFixed(0)}%
                </small>
              </div>
              {emotionalState.activated_aus?.length > 0 && (
                <p className="activated-aus">
                  Activated AU proxies: {emotionalState.activated_aus.join(", ")}
                </p>
              )}
              {Object.entries(emotionalState.emotion_scores)
                .filter(([, v]) => v > 0)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 4)
                .map(([emotion, score]) => (
                  <div key={emotion} className="emotion-bar-row">
                    <span>{emotion}</span>
                    <div className="emotion-bar">
                      <div
                        className="emotion-bar-fill"
                        style={{ width: `${Math.round(score * 100)}%` }}
                      />
                    </div>
                    <small>{(score * 100).toFixed(0)}%</small>
                  </div>
                ))}
              {emotionalState.dominant_emotion === "neutral" && (
                <p className="emotion-disclaimer">
                  No AU combination cleared the activation threshold, so the
                  resting face reads as neutral.
                </p>
              )}
              <p className="emotion-disclaimer">
                FACS AU proxy mapping (Ekman &amp; Friesen, 1978).
                Not a direct measure of mental state.
              </p>
            </div>
          )}
        </div>
      </div>

      {observation.error && (
        <p className="vision-error" role="status">
          {observation.error}
        </p>
      )}
      <p className="observation-note">
        Video stays on this device. Facial coefficients are sent to the local API.
      </p>
    </section>
  );
}
