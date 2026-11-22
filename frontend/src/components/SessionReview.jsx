import { useState } from "react";
import { fetchSessionReview } from "../services/interviewApi.js";

export default function SessionReview({ session, disabled, onReviewLoaded }) {
  const [review, setReview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleReview() {
    setBusy(true);
    setError("");
    try {
      const result = await fetchSessionReview({ state: session });
      setReview(result);
      onReviewLoaded?.(result);
    } catch (reviewError) {
      setError(reviewError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel session-review">
      <header className="panel-heading">
        <p className="eyebrow">After the practice</p>
        <h2>Session review</h2>
        <p className="panel-subtitle">
          What you covered is derived from your own answers. Suggestions are
          written by the model from the same facts.
        </p>
      </header>

      <button
        type="button"
        className="primary-button"
        onClick={handleReview}
        disabled={busy || disabled}
      >
        {busy ? "Reviewing…" : "Review this session"}
      </button>

      {disabled && (
        <p className="cv-status">Answer at least two questions first.</p>
      )}
      {error && <p className="vision-error">{error}</p>}

      {review && (
        <div className="review-body">
          <p className="cv-status">
            {review.turns} turn{review.turns === 1 ? "" : "s"} ·{" "}
            {review.review_source === "groq"
              ? "suggestions from Groq"
              : "local suggestions"}
          </p>

          <div className="review-columns">
            <div>
              <h3>Covered</h3>
              {review.covered.length > 0 ? (
                <ul className="review-list covered">
                  {review.covered.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="cv-status">Nothing established yet.</p>
              )}
            </div>
            <div>
              <h3>Missing</h3>
              {review.missing.length > 0 ? (
                <ul className="review-list missing">
                  {review.missing.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="cv-status">Nothing missing.</p>
              )}
            </div>
          </div>

          {review.suggestions.length > 0 && (
            <div>
              <h3>Suggestions</h3>
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
