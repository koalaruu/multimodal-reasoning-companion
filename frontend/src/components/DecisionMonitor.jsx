function readable(value) {
  return value ? value.replaceAll("_", " ") : "Not available";
}

function EvidenceItem({ label, active }) {
  return (
    <li className={active ? "evidence-active" : ""}>
      <span>{label}</span>
      <strong>{active ? "Observed" : "No evidence"}</strong>
    </li>
  );
}

export default function DecisionMonitor({
  result,
  session,
  onReset,
  audioObservation,
  audioReadiness,
  visualObservation,
}) {
  const decision = result?.decision;
  const evidence = result?.content_evidence;
  const signals = result?.signals;

  return (
    <aside className="monitor-panel" aria-labelledby="monitor-heading">
      <div className="panel-heading monitor-heading">
        <div>
          <p className="eyebrow">System transparency</p>
          <h2 id="monitor-heading">Decision monitor</h2>
        </div>
        <button className="secondary-button" type="button" onClick={onReset}>
          Reset session
        </button>
      </div>

      <section className="decision-card">
        <h3>Latest policy decision</h3>
        {decision ? (
          <dl className="decision-grid">
            <div>
              <dt>Action</dt>
              <dd>{readable(decision.action)}</dd>
            </div>
            <div>
              <dt>Reason</dt>
              <dd>{readable(decision.reason)}</dd>
            </div>
            <div>
              <dt>Follow-up topic</dt>
              <dd>{readable(decision.followup_topic)}</dd>
            </div>
            <div>
              <dt>Reply source</dt>
              <dd>{readable(result.reply_source)}</dd>
            </div>
          </dl>
        ) : (
          <p className="empty-state">
            Submit an answer to inspect the evidence and local policy decision.
          </p>
        )}
        {result?.fallback_reason && (
          <p className="fallback-notice">
            Local fallback: {readable(result.fallback_reason)}
          </p>
        )}
      </section>

      <section className="evidence-card">
        <h3>Observed evidence</h3>
        <dl className="decision-grid">
          <div>
            <dt>Input type</dt>
            <dd>{readable(evidence?.input_type)}</dd>
          </div>
          <div>
            <dt>Answer relevance</dt>
            <dd>{readable(evidence?.relevance)}</dd>
          </div>
        </dl>
        <ul className="evidence-list">
          <EvidenceItem label="Hint requested" active={signals?.hint_requested} />
          <EvidenceItem
            label="Thinking time requested"
            active={signals?.thinking_time_requested}
          />
          <EvidenceItem label="Stop requested" active={signals?.stop_requested} />
          <EvidenceItem
            label="Project described"
            active={evidence?.project_described}
          />
          <EvidenceItem
            label="Evaluation method mentioned"
            active={evidence?.evaluation_method_mentioned}
          />
          <EvidenceItem
            label="Evaluation result mentioned"
            active={evidence?.evaluation_result_mentioned}
          />
          <EvidenceItem
            label="Macro-F1 mentioned"
            active={evidence?.macro_f1_mentioned}
          />
          <EvidenceItem
            label="Class imbalance mentioned"
            active={evidence?.class_imbalance_mentioned}
          />
          <EvidenceItem
            label="Class-level result mentioned"
            active={evidence?.class_level_performance_mentioned}
          />
        </ul>
      </section>

      <section className="audio-evidence-card">
        <h3>Audio turn evidence</h3>
        <dl className="session-grid">
          <div>
            <dt>Availability</dt>
            <dd>{readable(audioObservation.availability)}</dd>
          </div>
          <div>
            <dt>Turn cue</dt>
            <dd>{readable(audioReadiness.status)}</dd>
          </div>
          <div>
            <dt>Speaking now</dt>
            <dd>{audioObservation.is_speaking ? "Yes" : "No"}</dd>
          </div>
          <div>
            <dt>Trailing silence</dt>
            <dd>
              {audioObservation.trailing_silence_ms == null
                ? "Not observed"
                : `${audioObservation.trailing_silence_ms} ms`}
            </dd>
          </div>
        </dl>
        <p className="observation-note">
          Voice activity is an advisory timing signal, not an emotion or
          attention measurement.
        </p>
      </section>
      <section className="vision-evidence-card">
        <h3>Visual perception evidence</h3>
        <dl className="session-grid">
          <div>
            <dt>Availability</dt>
            <dd>{readable(visualObservation.availability)}</dd>
          </div>
          <div>
            <dt>Visual cue</dt>
            <dd>{readable(visualObservation.cue)}</dd>
          </div>
          <div>
            <dt>Faces</dt>
            <dd>
              {visualObservation.availability === "available"
                ? visualObservation.face_count
                : "Not observed"}
            </dd>
          </div>
          <div>
            <dt>Inference</dt>
            <dd>
              {visualObservation.inference_ms == null
                ? "Not observed"
                : `${visualObservation.inference_ms.toFixed(1)} ms`}
            </dd>
          </div>
        </dl>
        <p className="observation-note">
          Face landmarks are observable coefficients, not emotion or attention
          measurements. They do not change the dialogue policy in this baseline.
        </p>
      </section>
      <section className="session-card">
        <h3>Session state</h3>
        <dl className="session-grid">
          <div>
            <dt>Status</dt>
            <dd>{session.status ?? "active"}</dd>
          </div>
          <div>
            <dt>Turn</dt>
            <dd>{session.turn_index ?? 0}</dd>
          </div>
          <div>
            <dt>Hints</dt>
            <dd>{session.hints_offered ?? 0}</dd>
          </div>
          <div>
            <dt>Follow-ups</dt>
            <dd>{session.followups_asked ?? 0}</dd>
          </div>
        </dl>
        <p className="session-id" title={session.session_id}>
          Session {session.session_id.slice(0, 8)}
        </p>
      </section>
    </aside>
  );
}
