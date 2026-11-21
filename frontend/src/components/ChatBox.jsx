import { useEffect, useRef } from "react";

export default function ChatBox({
  messages,
  draft,
  onDraftChange,
  onSubmit,
  sending,
  ended,
}) {
  const messageEndRef = useRef(null);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function handleSubmit(event) {
    event.preventDefault();
    const answer = draft.trim();
    if (answer && !sending && !ended) {
      onSubmit(answer);
    }
  }

  return (
    <section className="chat-panel" aria-labelledby="practice-heading">
      <div className="panel-heading chat-heading">
        <div>
          <p className="eyebrow">Interview practice</p>
          <h2 id="practice-heading">AI project discussion</h2>
        </div>
        <span className={`session-status ${ended ? "ended" : "active"}`}>
          {ended ? "Session ended" : "Session active"}
        </span>
      </div>

      <div className="message-list" aria-live="polite">
        {messages.map((message) => (
          <article
            className={`message ${message.role}`}
            key={message.id}
          >
            <span className="message-role">
              {message.role === "assistant" ? "Interviewer" : "You"}
            </span>
            <p>{message.text}</p>
            {message.source && (
              <span className={`reply-source ${message.source}`}>
                {message.source.replaceAll("_", " ")}
              </span>
            )}
          </article>
        ))}
        {sending && (
          <div className="thinking" role="status">
            <span />
            <span />
            <span />
            <span className="sr-only">Preparing the next response</span>
          </div>
        )}
        <div ref={messageEndRef} />
      </div>

      <form className="answer-form" onSubmit={handleSubmit}>
        <label htmlFor="candidate-answer">Your answer</label>
        <textarea
          id="candidate-answer"
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          placeholder={
            ended
              ? "Reset the session to continue practicing."
              : "Describe your project, evaluation method, and results..."
          }
          rows="4"
          disabled={sending || ended}
        />
        <div className="form-footer">
          <span>{draft.length} characters</span>
          <button
            className="primary-button"
            type="submit"
            disabled={!draft.trim() || sending || ended}
          >
            {sending ? "Sending..." : "Send answer"}
          </button>
        </div>
      </form>
    </section>
  );
}
