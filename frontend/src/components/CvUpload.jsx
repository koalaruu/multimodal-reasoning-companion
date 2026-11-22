import { useRef, useState } from "react";
import { extractCv } from "../services/interviewApi.js";

export default function CvUpload({ cv, targetRole, onCvLoaded, onTargetRoleChange }) {
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
          placeholder="e.g. Machine Learning Engineer Intern"
          maxLength={200}
          onChange={(event) => onTargetRoleChange(event.target.value)}
        />
      </label>

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
