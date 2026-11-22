import { useRef, useState } from "react";
import useAudioActivity from "./audio/useAudioActivity.js";
import useVoiceTranscription from "./audio/useVoiceTranscription.js";
import useVisualPerception from "./vision/useVisualPerception.js";
import { isCameraActive } from "./vision/visualObservation.js";
import AudioControls from "./components/AudioControls.jsx";
import ChatBox from "./components/ChatBox.jsx";
import DecisionMonitor from "./components/DecisionMonitor.jsx";
import CvUpload from "./components/CvUpload.jsx";
import SessionReview from "./components/SessionReview.jsx";
import { deriveTurnReadiness } from "./audio/turnReadiness.js";
import VisualControls from "./components/VisualControls.jsx";
import {
  fetchOpeningQuestion,
  submitInterviewTurn,
} from "./services/interviewApi.js";

const INTERVIEW_QUESTION =
  "Tell me about an AI project you built. How did you evaluate it?";

const DASHBOARD_VIEWS = [
  { id: "practice", label: "Practice room", icon: "M5 4h14v12H9l-4 4V4Z M9 8h6 M9 12h4" },
  { id: "context", label: "Application context", icon: "M7 3h7l4 4v14H6V3h1Z M14 3v5h4 M9 12h6 M9 16h4" },
  { id: "insights", label: "Session insights", icon: "M3 12h4l3-7 4 14 3-7h4" },
];

function createSession() {
  return {
    session_id: crypto.randomUUID(),
    conversation: [
      { role: "interviewer", content: INTERVIEW_QUESTION },
    ],
  };
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
  const [cv, setCv] = useState(null);
  const [targetRole, setTargetRole] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [openingBusy, setOpeningBusy] = useState(false);
  const [interviewMode, setInterviewMode] = useState("strict");
  const [activeView, setActiveView] = useState("practice");
  const mainRef = useRef(null);
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

  function navigateTo(view) {
    setActiveView(view);
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "instant" });
  }

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
      const readiness = deriveTurnReadiness(audio.observation);
      const result = await submitInterviewTurn({
        candidateInput: answer,
        state: session,
        emotionalState: vision.emotionalState,
        turnReadiness: readiness.status,
        trailingSilenceMs: audio.observation?.trailing_silence_ms ?? null,
        cvText: cv?.text,
        targetRole: targetRole.trim(),
        jobDescription: jobDescription.trim(),
        interviewMode,
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

  async function handleGenerateOpening() {
    setOpeningBusy(true);
    setError("");
    try {
      const result = await fetchOpeningQuestion({
        cvText: cv?.text,
        targetRole: targetRole.trim(),
        jobDescription: jobDescription.trim(),
        interviewMode,
      });
      setMessages((current) => {
        const rest = current.slice(1);
        return [
          {
            id: crypto.randomUUID(),
            role: "assistant",
            text: result.reply,
            source: result.reply_source,
          },
          ...rest,
        ];
      });
      setSession((current) => ({
        ...current,
        conversation: [
          { role: "interviewer", content: result.reply },
          ...(current.conversation ?? []).slice(1),
        ],
      }));
      navigateTo("practice");
    } catch (openingError) {
      setError(openingError.message);
    } finally {
      setOpeningBusy(false);
    }
  }

  function handleReset() {
    voice.cancel();
    setSession(createSession());
    setMessages(initialMessages());
    setLatestResult(null);
    setDraft("");
    setError("");
    setSending(false);
    setCv(null);
    setTargetRole("");
    setJobDescription("");
    setInterviewMode("strict");
    navigateTo("practice");
    audio.disableMicrophone();
    vision.disableCamera();
  }

  return (
    <div className="dashboard-shell">
      <a className="skip-link" href="#main-content">Skip to workspace</a>
      <aside className="dashboard-sidebar">
        <a className="brand" href="#main-content" onClick={() => navigateTo("practice")}>
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M4 6h6v6H4z M14 6h6v6h-6z M4 16h6 M14 16h6 M7 3v3 M17 3v3" />
            </svg>
          </span>
          <span>Companion<small>INTERVIEW WORKSPACE</small></span>
        </a>

        <p className="nav-caption">Your workspace</p>
        <nav className="dashboard-nav" aria-label="Workspace navigation">
          {DASHBOARD_VIEWS.map((view) => (
            <button
              key={view.id}
              type="button"
              aria-current={activeView === view.id ? "page" : undefined}
              onClick={() => navigateTo(view.id)}
              aria-controls={`${view.id}-view`}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d={view.icon} />
              </svg>
              {view.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-session">
          <span className="sidebar-label">CURRENT SESSION</span>
          <strong>{targetRole.trim() || "Interview practice"}</strong>
          <p>{session.turn_index ?? 0} turns · {interviewMode === "coach" ? "Coach" : "Strict"} style</p>
          <button type="button" onClick={handleReset} disabled={sending || openingBusy}>
            <span aria-hidden="true">＋</span> New session
          </button>
        </div>
        <div className="sidebar-note">
          <span className="privacy-dot" />
          <div><strong>Private by design</strong><p>No server-side chat storage.<br />Camera frames stay local.</p></div>
        </div>
        <div className="sidebar-footer">Perception → State → Dialogue</div>
      </aside>

      <main className="dashboard-main" id="main-content" ref={mainRef} tabIndex={-1}>
        <header className="dashboard-header">
          <div>
            <p className="breadcrumb">WORKSPACE <span>/</span> {DASHBOARD_VIEWS.find((view) => view.id === activeView).label}</p>
            <h1>{activeView === "practice" ? "Let’s practice." : activeView === "context" ? "Make it your interview." : "See what’s behind the conversation."}</h1>
            <p className="site-intro">
              {activeView === "practice"
                ? "A focused space to explain your experience, one answer at a time."
                : activeView === "context"
                  ? "Add the role, job description, and your background to guide the questions."
                  : "Review your answers and inspect the evidence used by the system."}
            </p>
          </div>
          <div className="header-status">
            <span className={`session-status ${ended ? "ended" : "active"}`}>
              <span className="status-dot" />{ended ? "Session ended" : "Session active"}
            </span>
            <button className="sensor-status" type="button" onClick={() => navigateTo("practice")} title="Open the practice room for sensor controls">
              {isCameraActive(vision.observation) ? "Camera on" : "Camera off"} · {voice.status === "recording" ? "Recording voice" : audio.observation.availability === "available" ? "Mic on" : "Mic off"}
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <strong>Request failed.</strong><span>{error}</span>
          </div>
        )}

        <section id="practice-view" hidden={activeView !== "practice"} aria-label="Practice room">
          <div className="session-brief">
            <div className="brief-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                <path d="M4 7h16v13H4z M9 7V4h6v3 M4 12h16 M10 12v3h4v-3" />
              </svg>
            </div>
            <div className="brief-context">
              <span className="eyebrow">Your interview</span>
              <strong>{targetRole.trim() || "Set up your target role"}</strong>
              <p>{cv ? "CV added" : "No CV added"} <span>·</span> {jobDescription.trim() ? "Job description added" : "No job description"} <span>·</span> {interviewMode === "coach" ? "Coach" : "Strict"} mode</p>
            </div>
            <div className="brief-actions">
              <button type="button" className="secondary-button" onClick={() => navigateTo("context")}>Edit context</button>
              <button type="button" className="primary-button" onClick={handleGenerateOpening} disabled={openingBusy || sending || ended}>
                {openingBusy ? "Generating…" : "Generate opening"} <span aria-hidden="true">↗</span>
              </button>
            </div>
          </div>

          <div className="practice-grid">
            <ChatBox
              title={targetRole.trim() || "Interview practice"}
              messages={messages}
              draft={draft}
              onDraftChange={setDraft}
              onSubmit={handleSubmit}
              sending={sending}
              ended={ended}
            />
            <aside className="sensor-column" aria-label="Optional perception controls">
              <div className="column-label"><span>LIVE PERCEPTION</span><span>OPT-IN</span></div>
              <VisualControls
                observation={vision.observation}
                emotionalState={vision.emotionalState}
                videoRef={vision.videoRef}
                onEnable={vision.enableCamera}
                onDisable={vision.disableCamera}
              />
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
            </aside>
          </div>
        </section>

        <section id="context-view" hidden={activeView !== "context"} aria-label="Application context">
          <div className="context-layout">
            <CvUpload
              cv={cv}
              targetRole={targetRole}
              jobDescription={jobDescription}
              interviewMode={interviewMode}
              onInterviewModeChange={setInterviewMode}
              onCvLoaded={setCv}
              onTargetRoleChange={setTargetRole}
              onJobDescriptionChange={setJobDescription}
            />
            <aside className="context-guide">
              <span className="step-number">01 / PREPARE</span>
              <h2>Better context.<br />More relevant questions.</h2>
              <ol>
                <li><strong>Choose your role</strong><p>Use the job title you are actually applying for.</p></li>
                <li><strong>Add the posting &amp; CV</strong><p>Give the interviewer real requirements and projects to discuss.</p></li>
                <li><strong>Start the conversation</strong><p>Generate an opening based on your context, then answer in your own words.</p></li>
              </ol>
              <button type="button" className="primary-button" onClick={handleGenerateOpening} disabled={openingBusy || sending || ended}>
                {openingBusy ? "Generating…" : "Generate opening & practice"} <span aria-hidden="true">→</span>
              </button>
              <p className="observation-note">Your details are sent to Groq with the interview. Nothing is saved to a server-side profile.</p>
            </aside>
          </div>
        </section>

        <section id="insights-view" hidden={activeView !== "insights"} aria-label="Session insights">
          <SessionReview
            key={session.session_id}
            session={session}
            disabled={(session.turn_index ?? 0) < 2 || sending}
          />
          <DecisionMonitor
            result={latestResult}
            session={session}
            onReset={handleReset}
            audioObservation={audio.observation}
            audioReadiness={audio.readiness}
            visualObservation={vision.observation}
          />
        </section>

        <footer className="dashboard-footer">
          <span>Built for thoughtful practice, not automated hiring decisions.</span>
          <span>Facial cues are advisory, not a measure of mental state.</span>
        </footer>
      </main>
    </div>
  );
}
