import { useEffect, useRef, useState } from "react";
import { fetchSessionReview } from "../services/interviewApi.js";

const FALLBACK_MESSAGES = {
  timeout: "Groq timed out while reviewing this session.",
  sdk_error: "The Groq request failed. Check the backend log for its error type and HTTP status.",
  malformed_output: "Groq returned a review that did not match the expected format.",
};

function sameReviewInputs(left, right) {
  return left.session_id === right.session_id &&
    left.turn_index === right.turn_index &&
    left.conversation === right.conversation &&
    left.cumulative_evidence === right.cumulative_evidence &&
    left.cv_text === right.cv_text &&
    left.target_role === right.target_role &&
    left.job_description === right.job_description;
}

export default function SessionReview({ session, disabled, turnPending = false, onReviewLoaded }) {
  const [snapshot, setSnapshot] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef(null);
  const review = snapshot?.result;
  const outdated = !!snapshot && (turnPending || !sameReviewInputs(snapshot.session, session));

  useEffect(() => () => requestRef.current?.abort(), []);

  async function handleReview() {
    if (requestRef.current || disabled || turnPending) return;
    const controller = new AbortController();
    const requestedSession = session;
    requestRef.current = controller;
    setBusy(true);
    setError("");
    try {
      const result = await fetchSessionReview({
        state: requestedSession,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setSnapshot({ result, session: requestedSession });
      onReviewLoaded?.(result);
    } catch (reviewError) {
      if (!controller.signal.aborted) setError(reviewError.message);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        if (!controller.signal.aborted) setBusy(false);
      }
    }
  }

  return (
    <section className="panel session-review">
      <header className="panel-heading">
        <p className="eyebrow">After the practice</p>
        <h2>Session review</h2>
        <p className="panel-subtitle">
          General answer cues: a work example, how you checked it, and an outcome.
          Groq tailors advice to your role and job description; this is not a skills score.
        </p>
      </header>

      <button
        type="button"
        className="primary-button"
        onClick={handleReview}
        disabled={busy || disabled || turnPending}
      >
        {busy ? "Reviewing…" : outdated ? "Update review" : "Review this session"}
      </button>

      {disabled && (
        <p className="cv-status">Answer at least two questions first.</p>
      )}
      {turnPending && !review && (
        <p className="cv-status" role="status">Wait for the pending answer to finish before reviewing.</p>
      )}
      {error && <p className="vision-error">{error}</p>}

      {review && (
        <div className="review-body">
          <p className={`review-freshness ${outdated ? "outdated" : "current"}`} role="status">
            {outdated ? "Outdated review" : "Current review"} · {review.turns} completed turn{review.turns === 1 ? "" : "s"}
          </p>
          {outdated && (
            <p className="review-stale-notice" role="status">
              {turnPending
                ? "A new answer is being processed."
                : "Your completed answers or application context have changed."}{" "}
              This review describes an earlier snapshot. Select Update review
              {turnPending ? " after the answer finishes" : ""} to review the latest completed answers and context.
            </p>
          )}
          <p className="cv-status">
            {review.review_source === "groq"
              ? "suggestions from Groq"
              : review.review_source === "local_fallback"
                ? "local fallback suggestions"
                : "local suggestions"}
          </p>
          {review.review_source === "local_fallback" && (
            <p className="fallback-notice" role="status">
              {FALLBACK_MESSAGES[review.fallback_reason] ?? "Groq could not complete this review."}{" "}
              Showing local suggestions; coverage still comes from your answers.
              Use the review button to try again.
            </p>
          )}
          {review.review_source === "local" && (
            <p className="observation-note">
              Local review mode. Configure Groq on the backend for model-written suggestions.
            </p>
          )}

          <div className="review-columns">
            <div>
              <h3>Observed answer cues</h3>
              {review.covered.length > 0 ? (
                <ul className="review-list covered">
                  {review.covered.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="cv-status">No general answer cues detected yet.</p>
              )}
            </div>
            <div>
              <h3>Not detected yet</h3>
              {review.missing.length > 0 ? (
                <ul className="review-list missing">
                  {review.missing.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="cv-status">All three general answer cues were detected.</p>
              )}
            </div>
          </div>

          {review.suggestions.length > 0 && (
            <div>
              <h3>Advice for your interview</h3>
              <ul className="review-list">
                {review.suggestions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {review.questions_to_ask.length > 0 && (
            <div>
              <h3>Questions you could ask</h3>
              <ul className="review-list">
                {review.questions_to_ask.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
