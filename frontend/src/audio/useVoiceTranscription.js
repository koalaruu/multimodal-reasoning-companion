import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudio } from "../services/interviewApi.js";

const MAX_RECORDING_MS = 60_000;
const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

function idleState() {
  return { status: "idle", error: null, result: null };
}

function recorderOptions() {
  if (typeof MediaRecorder === "undefined") return null;
  const mimeType =
    typeof MediaRecorder.isTypeSupported === "function"
      ? PREFERRED_MIME_TYPES.find((candidate) =>
          MediaRecorder.isTypeSupported(candidate),
        )
      : null;
  return mimeType ? { mimeType } : {};
}

export default function useVoiceTranscription({ getStream, onTranscript }) {
  const [state, setState] = useState(idleState);
  const sessionRef = useRef(null);
  const requestRef = useRef(null);
  const transcriptHandlerRef = useRef(onTranscript);

  useEffect(() => {
    transcriptHandlerRef.current = onTranscript;
  }, [onTranscript]);

  const transcribeRecording = useCallback(async (blob) => {
    if (blob.size === 0) {
      setState({
        status: "error",
        error: "The recording was empty. Try speaking after recording starts.",
        result: null,
      });
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;
    setState({ status: "transcribing", error: null, result: null });
    try {
      const result = await transcribeAudio({ audio: blob, signal: controller.signal });
      transcriptHandlerRef.current(result.text);
      setState({ status: "ready", error: null, result });
    } catch (error) {
      if (error?.name === "AbortError") return;
      setState({
        status: "error",
        error:
          error instanceof TypeError
            ? "Cannot reach the local transcription service."
            : error.message,
        result: null,
      });
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
  }, []);

  const startRecording = useCallback(() => {
    if (sessionRef.current || state.status === "transcribing") return;
    if (typeof MediaRecorder === "undefined") {
      setState({
        status: "error",
        error: "Voice recording is unavailable in this browser.",
        result: null,
      });
      return;
    }

    const stream = getStream();
    if (!stream?.active) {
      setState({
        status: "error",
        error: "Enable the microphone before recording a voice answer.",
        result: null,
      });
      return;
    }

    let recorder;
    try {
      recorder = new MediaRecorder(stream, recorderOptions());
    } catch {
      setState({
        status: "error",
        error: "The browser could not start an audio recording.",
        result: null,
      });
      return;
    }

    const chunks = [];
    const session = {
      cancelled: false,
      chunks,
      recorder,
      timer: null,
      finish: null,
    };
    session.finish = () => {
      if (session.cancelled || recorder.state === "inactive") return;
      setState({ status: "transcribing", error: null, result: null });
      recorder.stop();
    };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("error", () => {
      session.cancelled = true;
      clearTimeout(session.timer);
      sessionRef.current = null;
      setState({
        status: "error",
        error: "Recording failed before transcription started.",
        result: null,
      });
    });
    recorder.addEventListener("stop", () => {
      clearTimeout(session.timer);
      if (sessionRef.current === session) sessionRef.current = null;
      if (session.cancelled) return;
      const blob = new Blob(chunks, {
        type: recorder.mimeType || "audio/webm",
      });
      void transcribeRecording(blob);
    });

    try {
      recorder.start(250);
      sessionRef.current = session;
      session.timer = setTimeout(session.finish, MAX_RECORDING_MS);
      setState({ status: "recording", error: null, result: null });
    } catch {
      setState({
        status: "error",
        error: "The browser could not start an audio recording.",
        result: null,
      });
    }
  }, [getStream, state.status, transcribeRecording]);

  const stopAndTranscribe = useCallback(() => {
    sessionRef.current?.finish();
  }, []);

  const cancel = useCallback(() => {
    const session = sessionRef.current;
    if (session) {
      session.cancelled = true;
      clearTimeout(session.timer);
      if (session.recorder.state !== "inactive") session.recorder.stop();
      sessionRef.current = null;
    }
    requestRef.current?.abort();
    requestRef.current = null;
    setState(idleState());
  }, []);

  useEffect(() => cancel, [cancel]);

  return {
    cancel,
    error: state.error,
    result: state.result,
    startRecording,
    status: state.status,
    stopAndTranscribe,
  };
}
