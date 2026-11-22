import { useEffect, useRef } from "react";

export default function ChatBox({
  title = "Interview practice",
  messages,
  draft,
  onDraftChange,
  onSubmit,
  sending,
  openingBusy = false,
  ended,
}) {
  const messageListRef = useRef(null);

  useEffect(() => {
    const list = messageListRef.current;
    list?.scrollTo({
      top: list.scrollHeight,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
    });
  }, [messages, sending]);

  function handleSubmit(event) {
    event.preventDefault();
    const answer = draft.trim();
    if (answer && !sending && !openingBusy && !ended) {
      onSubmit(answer);
    }
  }

  return (
    <section className="chat-panel" aria-labelledby="practice-heading">
      <div className="panel-heading chat-heading">
        <div>
          <p className="eyebrow">Interview practice</p>
          <h2 id="practice-heading">{title}</h2>
        </div>
        <span className={`session-status ${ended ? "ended" : "active"}`}>
          {ended ? "Session ended" : "Session active"}
        </span>
      </div>

      <div className="message-list" ref={messageListRef} aria-live="polite" aria-busy={sending}>
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
              : "Share your answer, using relevant experience where appropriate..."
          }
          rows="4"
          disabled={sending || openingBusy || ended}
        />
        <div className="form-footer">
          <span>{draft.length} characters</span>
          <button
            className="primary-button"
            type="submit"
            disabled={!draft.trim() || sending || openingBusy || ended}
          >
            {openingBusy ? "Generating opening..." : sending ? "Sending..." : "Send answer"}
          </button>
        </div>
      </form>
    </section>
  );
}
