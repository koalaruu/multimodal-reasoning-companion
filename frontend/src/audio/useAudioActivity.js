import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { deriveTurnReadiness } from "./turnReadiness.js";

const ENTER_SPEECH_RMS = 0.035;
const EXIT_SPEECH_RMS = 0.02;
const UI_UPDATE_INTERVAL_MS = 100;

function unavailableObservation(permission = "idle", error = null) {
  return {
    availability: "unavailable",
    permission,
    has_speech: false,
    is_speaking: false,
    speech_duration_ms: 0,
    trailing_silence_ms: null,
    level: 0,
    error,
  };
}

export default function useAudioActivity() {
  const [observation, setObservation] = useState(unavailableObservation());
  const resourcesRef = useRef(null);

  const stopResources = useCallback(() => {
    const resources = resourcesRef.current;
    if (!resources) return;

    cancelAnimationFrame(resources.animationFrame);
    resources.source.disconnect();
    resources.analyser.disconnect();
    resources.stream.getTracks().forEach((track) => track.stop());
    void resources.audioContext.close();
    resourcesRef.current = null;
  }, []);

  const disableMicrophone = useCallback(() => {
    stopResources();
    setObservation(unavailableObservation("idle"));
  }, [stopResources]);

  const getStream = useCallback(
    () => resourcesRef.current?.stream ?? null,
    [],
  );

  const enableMicrophone = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setObservation(
        unavailableObservation(
          "unsupported",
          "Microphone activity is unavailable in this browser.",
        ),
      );
      return;
    }

    stopResources();
    setObservation(unavailableObservation("requesting"));

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          autoGainControl: true,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      const audioContext = new AudioContext();
      await audioContext.resume();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 1_024;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);

      const samples = new Float32Array(analyser.fftSize);
      let animationFrame = 0;
      let speaking = false;
      let speechDurationMs = 0;
      let lastSpeechAt = null;
      let previousFrameAt = performance.now();
      let lastUiUpdateAt = 0;

      const sampleAudio = (now) => {
        analyser.getFloatTimeDomainData(samples);
        let sumSquares = 0;
        for (let index = 0; index < samples.length; index += 1) {
          sumSquares += samples[index] * samples[index];
        }
        const rms = Math.sqrt(sumSquares / samples.length);
        const nextSpeaking = speaking
          ? rms >= EXIT_SPEECH_RMS
          : rms >= ENTER_SPEECH_RMS;
        const elapsedMs = Math.min(now - previousFrameAt, 250);

        if (nextSpeaking) {
          speechDurationMs += elapsedMs;
          lastSpeechAt = now;
        }

        const stateChanged = nextSpeaking !== speaking;
        speaking = nextSpeaking;
        previousFrameAt = now;

        if (stateChanged || now - lastUiUpdateAt >= UI_UPDATE_INTERVAL_MS) {
          setObservation({
            availability: "available",
            permission: "granted",
            has_speech: lastSpeechAt !== null,
            is_speaking: speaking,
            speech_duration_ms: Math.round(speechDurationMs),
            trailing_silence_ms:
              lastSpeechAt === null
                ? null
                : speaking
                  ? 0
                  : Math.round(now - lastSpeechAt),
            level: Math.min(1, rms / 0.12),
            error: null,
          });
          lastUiUpdateAt = now;
        }
        animationFrame = requestAnimationFrame(sampleAudio);
        resourcesRef.current.animationFrame = animationFrame;
      };

      resourcesRef.current = {
        analyser,
        animationFrame,
        audioContext,
        source,
        stream,
      };
      setObservation({
        ...unavailableObservation("granted"),
        availability: "available",
      });
      animationFrame = requestAnimationFrame(sampleAudio);
      resourcesRef.current.animationFrame = animationFrame;
    } catch (error) {
      stopResources();
      const denied = error?.name === "NotAllowedError";
      setObservation(
        unavailableObservation(
          denied ? "denied" : "error",
          denied
            ? "Microphone permission was not granted. Text practice still works."
            : "Microphone activity could not be started. Text practice still works.",
        ),
      );
    }
  }, [stopResources]);

  useEffect(() => stopResources, [stopResources]);

  const readiness = useMemo(
    () => deriveTurnReadiness(observation),
    [observation],
  );

  return {
    disableMicrophone,
    enableMicrophone,
    getStream,
    observation,
    readiness,
  };
}
