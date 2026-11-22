import { useState } from "react";
import useAudioActivity from "./audio/useAudioActivity.js";
import useVoiceTranscription from "./audio/useVoiceTranscription.js";
import useVisualPerception from "./vision/useVisualPerception.js";
import AudioControls from "./components/AudioControls.jsx";
import ChatBox from "./components/ChatBox.jsx";
import DecisionMonitor from "./components/DecisionMonitor.jsx";
import VisualControls from "./components/VisualControls.jsx";
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
  const voice = useVoiceTranscription({
    getStream: audio.getStream,
    onTranscript: (transcript) => {
      setDraft((current) =>
        [current.trim(), transcript.trim()].filter(Boolean).join(" "),
      );
    },
  });
  const vision = useVisualPerception();

  const ended = session.status === "ended";

  async function handleSubmit(answer) {
    voice.cancel();
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

  function handleDisableMicrophone() {
    voice.cancel();
    audio.disableMicrophone();
  }

  function handleReset() {
    voice.cancel();
    setSession(createSession());
    setMessages(initialMessages());
    setLatestResult(null);
    setDraft("");
    setError("");
    setSending(false);
    audio.disableMicrophone();
    vision.disableCamera();
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
        onDisable={handleDisableMicrophone}
        onStartRecording={voice.startRecording}
        onStopAndTranscribe={voice.stopAndTranscribe}
        voiceDisabled={sending || ended}
        voiceError={voice.error}
        voiceResult={voice.result}
        voiceStatus={voice.status}
      />
      <VisualControls
        observation={vision.observation}
        emotionalState={vision.emotionalState}
        videoRef={vision.videoRef}
        onEnable={vision.enableCamera}
        onDisable={vision.disableCamera}
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
          visualObservation={vision.observation}
        />
      </div>

      <footer>
        Dialogue policy, voice activity, transcription, and visual perception
        stay local. Groq may rewrite eligible replies only.
      </footer>
    </main>
  );
}
