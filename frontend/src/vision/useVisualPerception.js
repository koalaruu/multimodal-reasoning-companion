import { useCallback, useEffect, useRef, useState } from "react";
import { createLocalFaceLandmarker } from "./faceLandmarkerRuntime.js";
import {
  freshEmotionalState,
  summarizeFaceLandmarkerResult,
  unavailableVisualObservation,
  VISION_MAX_AGE_MS,
} from "./visualObservation.js";
import { analyzeVision } from "../services/interviewApi.js";

const INFERENCE_INTERVAL_MS = 200;
const VISION_UPLOAD_INTERVAL_MS = 1000;
const VISION_MIN_SAMPLES = 3;
const VISION_PREVIOUS_SAMPLES = 4;

const CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: "user",
    width: { ideal: 640 },
    height: { ideal: 480 },
  },
};

function cameraErrorMessage(error) {
  if (error?.name === "NotAllowedError") {
    return "Camera permission was not granted.";
  }
  if (error?.name === "NotFoundError") {
    return "No camera was found.";
  }
  if (error?.name === "NotReadableError") {
    return "The camera is already in use or unavailable.";
  }
  if (error?.name === "CameraUnsupportedError") {
    return "Camera access is not supported in this browser.";
  }
  return "Local face perception could not start. Check the model connection and try again.";
}

export default function useVisualPerception() {
  const [observation, setObservation] = useState(
    unavailableVisualObservation(),
  );
  const [emotionalState, setEmotionalState] = useState(null);
  const videoRef = useRef(null);
  const resourcesRef = useRef(null);
  const operationRef = useRef(0);
  // Facial evidence is timestamped so an old reading cannot outlive the camera.
  const emotionalRef = useRef(null);
  const analysisRef = useRef(null);

  const clearReading = useCallback(() => {
    clearTimeout(emotionalRef.current?.expiry);
    emotionalRef.current = null;
    setEmotionalState(null);
  }, []);

  const clearEmotionalState = useCallback(() => {
    analysisRef.current?.abort();
    analysisRef.current = null;
    if (resourcesRef.current) resourcesRef.current.history = [];
    clearReading();
  }, [clearReading]);

  const releaseResources = useCallback((target = resourcesRef.current) => {
    if (!target) return;

    cancelAnimationFrame(target.animationFrame);
    target.stream?.getTracks().forEach((track) => track.stop());
    try {
      target.landmarker?.close();
    } catch {
      // Camera tracks are already stopped; a runtime close failure is non-fatal.
    }
    if (target.video?.srcObject === target.stream) {
      target.video.srcObject = null;
    }
    if (resourcesRef.current === target) {
      resourcesRef.current = null;
    }
  }, []);

  const disableCamera = useCallback(() => {
    operationRef.current += 1;
    releaseResources();
    clearEmotionalState();
    setObservation(unavailableVisualObservation());
  }, [releaseResources, clearEmotionalState]);

  const enableCamera = useCallback(async () => {
    releaseResources();
    clearEmotionalState();
    const operation = operationRef.current + 1;
    operationRef.current = operation;
    const resources = {
      animationFrame: 0,
      landmarker: null,
      stream: null,
      video: null,
      history: [],
    };
    resourcesRef.current = resources;
    let permission = "requesting";
    setObservation(unavailableVisualObservation(permission));

    const isCurrent = () =>
      operation === operationRef.current && resourcesRef.current === resources;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        const unsupported = new Error("Camera API unavailable");
        unsupported.name = "CameraUnsupportedError";
        throw unsupported;
      }

      resources.stream = await navigator.mediaDevices.getUserMedia(
        CAMERA_CONSTRAINTS,
      );
      if (!isCurrent()) {
        releaseResources(resources);
        return;
      }
      for (const track of resources.stream.getVideoTracks()) {
        track.addEventListener("ended", () => {
          if (isCurrent()) disableCamera();
        }, { once: true });
      }

      permission = "granted";
      resources.video = videoRef.current;
      if (!resources.video) {
        throw new Error("Camera preview unavailable");
      }
      resources.video.srcObject = resources.stream;
      await resources.video.play();
      if (!isCurrent()) return;
      setObservation({
        ...unavailableVisualObservation(permission),
        cue: "loading_model",
      });

      resources.landmarker = await createLocalFaceLandmarker();
      if (!isCurrent()) {
        releaseResources(resources);
        return;
      }

      setObservation(
        summarizeFaceLandmarkerResult(
          { faceLandmarks: [], faceBlendshapes: [] },
          {},
        ),
      );

      let lastInferenceAt = -INFERENCE_INTERVAL_MS;
      let lastVideoTime = -1;
      let lastUploadAt = -VISION_UPLOAD_INTERVAL_MS;
      const render = (timestamp) => {
        if (!isCurrent()) {
          return;
        }

        const video = resources.video;
        if (
          video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          video.currentTime !== lastVideoTime &&
          timestamp - lastInferenceAt >= INFERENCE_INTERVAL_MS
        ) {
          try {
            lastInferenceAt = timestamp;
            lastVideoTime = video.currentTime;
            const startedAt = performance.now();
            const result = resources.landmarker.detectForVideo(video, timestamp);
            const obs = summarizeFaceLandmarkerResult(result, {
              inferenceMs: performance.now() - startedAt,
              observedAtMs: Date.now(),
            });
            setObservation(obs);

            if (obs.face_count !== 1) {
              if (analysisRef.current || emotionalRef.current || resources.history.length) {
                clearEmotionalState();
              }
            } else if (
              timestamp - lastUploadAt >= VISION_UPLOAD_INTERVAL_MS &&
              !analysisRef.current
            ) {
              lastUploadAt = timestamp;
              const controller = new AbortController();
              analysisRef.current = controller;
              analyzeVision({
                observation: obs,
                previousObservations: resources.history,
                signal: controller.signal,
              })
                .then(({ emotional_state: value, sample_count: count }) => {
                  // Stop or loss of a single face invalidates the whole window.
                  if (!isCurrent() || controller.signal.aborted) return;
                  const age = Date.now() - obs.observed_at_ms;
                  if (!value.face_available || age > VISION_MAX_AGE_MS) {
                    clearEmotionalState();
                    return;
                  }
                  resources.history.push(obs);
                  if (resources.history.length > VISION_PREVIOUS_SAMPLES) resources.history.shift();
                  if (count < VISION_MIN_SAMPLES) {
                    clearReading();
                    return;
                  }
                  clearTimeout(emotionalRef.current?.expiry);
                  emotionalRef.current = {
                    value,
                    atMs: obs.observed_at_ms,
                    expiry: setTimeout(clearReading, VISION_MAX_AGE_MS - age + 1),
                  };
                  setEmotionalState(value);
                })
                .catch(() => {
                  if (isCurrent() && !controller.signal.aborted) clearEmotionalState();
                })
                .finally(() => {
                  if (analysisRef.current === controller) {
                    analysisRef.current = null;
                  }
                });
            }
          } catch {
            operationRef.current += 1;
            releaseResources(resources);
            clearEmotionalState();
            setObservation(
              unavailableVisualObservation(
                "granted",
                "Local face perception stopped after an inference error.",
              ),
            );
            return;
          }
        }

        resources.animationFrame = requestAnimationFrame(render);
      };

      resources.animationFrame = requestAnimationFrame(render);
    } catch (error) {
      if (operation !== operationRef.current) return;
      releaseResources(resources);
      clearEmotionalState();
      setObservation(
        unavailableVisualObservation(
          error?.name === "NotAllowedError" ? "denied" : permission,
          cameraErrorMessage(error),
        ),
      );
    }
  }, [releaseResources, clearEmotionalState, clearReading, disableCamera]);

  // Returns facial evidence only while the camera runs and the reading is recent.
  const currentEmotionalState = useCallback(() => {
    if (!resourcesRef.current) return null;
    const value = freshEmotionalState(emotionalRef.current);
    if (!value && emotionalRef.current) clearReading();
    return value;
  }, [clearReading]);

  useEffect(
    () => () => {
      operationRef.current += 1;
      analysisRef.current?.abort();
      analysisRef.current = null;
      clearTimeout(emotionalRef.current?.expiry);
      releaseResources();
    },
    [releaseResources],
  );

  return {
    observation,
    emotionalState,
    currentEmotionalState,
    videoRef,
    enableCamera,
    disableCamera,
  };
}
