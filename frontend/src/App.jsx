import { useState } from "react";
import useAudioActivity from "./audio/useAudioActivity.js";
import AudioControls from "./components/AudioControls.jsx";
import ChatBox from "./components/ChatBox.jsx";
import DecisionMonitor from "./components/DecisionMonitor.jsx";
import { submitInterviewTurn } from "./services/interviewApi.js";

const INTERVIEW_QUESTION =
  "Tell me about an AI project you built. How did you evaluate it?";

function createSession() {
  return { session_id: crypto.randomUUID() };
}

function initialMessages() {
  return [
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: INTERVIEW_QUESTION,
      source: "local",
    },
  ];
}

export default function App() {
  const [session, setSession] = useState(createSession);
  const [messages, setMessages] = useState(initialMessages);
  const [latestResult, setLatestResult] = useState(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const audio = useAudioActivity();

  const ended = session.status === "ended";

  async function handleSubmit(answer) {
    const userMessage = {
      id: crypto.randomUUID(),
      role: "candidate",
      text: answer,
    };
    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setError("");
    setSending(true);

    try {
      const result = await submitInterviewTurn({
        candidateInput: answer,
        state: session,
      });
      setSession(result.updated_state);
      setLatestResult(result);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.reply,
          source: result.reply_source,
        },
      ]);
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "Cannot reach the interview API. Confirm that FastAPI is running on port 8000."
          : requestError.message,
      );
      setDraft(answer);
    } finally {
      setSending(false);
    }
  }

  function handleReset() {
    setSession(createSession());
    setMessages(initialMessages());
    setLatestResult(null);
    setDraft("");
    setError("");
    setSending(false);
    audio.disableMicrophone();
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <div>
          <p className="brand-kicker">Multimodal reasoning companion</p>
          <h1>Practice explaining your AI work.</h1>
          <p className="site-intro">
            A transparent interview partner that shows the evidence behind its
            next conversational action.
          </p>
        </div>
        <div className="privacy-badge">
          <span />
          Session text is not persisted
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          <strong>Turn not sent.</strong>
          <span>{error}</span>
        </div>
      )}
      <AudioControls
        observation={audio.observation}
        readiness={audio.readiness}
        onEnable={audio.enableMicrophone}
        onDisable={audio.disableMicrophone}
      />


      <div className="workspace">
        <ChatBox
          messages={messages}
          draft={draft}
          onDraftChange={setDraft}
          onSubmit={handleSubmit}
          sending={sending}
          ended={ended}
        />
        <DecisionMonitor
          result={latestResult}
          session={session}
          onReset={handleReset}
          audioObservation={audio.observation}
          audioReadiness={audio.readiness}
        />
      </div>

      <footer>
        Dialogue policy and voice activity stay local. Gemini may rewrite eligible replies only.
      </footer>
    </main>
  );
}
