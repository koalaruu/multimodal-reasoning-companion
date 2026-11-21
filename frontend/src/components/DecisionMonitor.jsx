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

export default function DecisionMonitor({ result, session, onReset }) {
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
        <ul className="evidence-list">
          <EvidenceItem label="Hint requested" active={signals?.hint_requested} />
          <EvidenceItem
            label="Thinking time requested"
            active={signals?.thinking_time_requested}
          />
          <EvidenceItem label="Stop requested" active={signals?.stop_requested} />
          <EvidenceItem
            label="Project described"
            active={evidence?.project_mentioned}
          />
          <EvidenceItem
            label="Evaluation discussed"
            active={evidence?.evaluation_mentioned}
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
