import { useRef, useState } from "react";
import { extractCv } from "../services/interviewApi.js";

export default function CvUpload({
  cv,
  targetRole,
  jobDescription,
  interviewMode,
  onInterviewModeChange,
  onCvLoaded,
  onTargetRoleChange,
  onJobDescriptionChange,
}) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const result = await extractCv({ file });
      onCvLoaded(result);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  function handleClear() {
    onCvLoaded(null);
    setError("");
  }

  return (
    <section className="panel cv-upload">
      <header className="panel-heading">
        <p className="eyebrow">Optional context</p>
        <h2>Target role and CV</h2>
        <p className="panel-subtitle">
          The interviewer uses this to ask about your real projects instead of
          generic ones. Text stays in your browser session and is sent with each
          turn; nothing is stored on the server.
        </p>
      </header>

      <label className="field">
        <span>Role you are applying for</span>
        <input
          type="text"
          value={targetRole}
          placeholder="e.g. Flutter Developer or Sales Associate"
          maxLength={200}
          onChange={(event) => onTargetRoleChange(event.target.value)}
        />
      </label>

      <label className="field">
        <span>Job description (recommended)</span>
        <textarea
          rows={5}
          value={jobDescription}
          placeholder="Paste the posting: responsibilities, required skills, tools, and metrics the role cares about."
          maxLength={8000}
          onChange={(event) => onJobDescriptionChange(event.target.value)}
        />
        <small className="cv-status">
          {jobDescription.length} / 8000 characters — the interviewer uses this
          to pick which CV projects to probe.
        </small>
      </label>

      <fieldset className="mode-select">
        <legend>Interviewer style</legend>
        <label>
          <input
            type="radio"
            name="interview-mode"
            value="strict"
            checked={interviewMode === "strict"}
            onChange={() => onInterviewModeChange("strict")}
          />
          <span>
            <strong>Strict</strong> — one question, no commentary
          </span>
        </label>
        <label>
          <input
            type="radio"
            name="interview-mode"
            value="coach"
            checked={interviewMode === "coach"}
            onChange={() => onInterviewModeChange("coach")}
          />
          <span>
            <strong>Coach</strong> — one short observation, then the question
          </span>
        </label>
      </fieldset>
      <p className="cv-status">
        Applies to the next generated opening or reply; switching styles keeps
        the conversation. Local control replies are unchanged.
      </p>


      <div className="cv-actions">
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          onChange={handleFile}
          disabled={busy}
          aria-label="Upload CV"
        />
        {cv && (
          <button type="button" className="secondary" onClick={handleClear}>
            Remove CV
          </button>
        )}
      </div>

      {busy && <p className="cv-status">Extracting text…</p>}
      {error && <p className="vision-error">{error}</p>}
      {cv && (
        <div className="cv-loaded">
          <p className="cv-status">
            Loaded <strong>{cv.filename}</strong> — {cv.characters} characters
          </p>
          <p className="cv-preview">{cv.preview}</p>
        </div>
      )}
    </section>
  );
}
