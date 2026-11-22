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
          <p className="eyebrow">Optional local vision</p>
          <h2 id="vision-heading">Observable face landmarks</h2>
          <p>
            MediaPipe runs in this browser and reports face count plus raw facial
            coefficients. It does not infer mood, stress, focus, or suitability.
          </p>
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
            <div className="camera-placeholder">Camera preview stays local</div>
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
                  ? "Advisory perception only; dialogue policy is unchanged."
                  : "Text and audio remain available."}
              </small>
            </div>
          </div>

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
        Video frames are not uploaded. Blendshape coefficients are sent to the
        local backend for FACS proxy mapping; the MediaPipe runtime is bundled.
      </p>
    </section>
  );
}
