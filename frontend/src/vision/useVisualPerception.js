import { useCallback, useEffect, useRef, useState } from "react";
import { createLocalFaceLandmarker } from "./faceLandmarkerRuntime.js";
import {
  summarizeFaceLandmarkerResult,
  unavailableVisualObservation,
} from "./visualObservation.js";

const INFERENCE_INTERVAL_MS = 200;
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
  const videoRef = useRef(null);
  const resourcesRef = useRef(null);
  const operationRef = useRef(0);

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
    setObservation(unavailableVisualObservation());
  }, [releaseResources]);

  const enableCamera = useCallback(async () => {
    releaseResources();
    const operation = operationRef.current + 1;
    operationRef.current = operation;
    const resources = {
      animationFrame: 0,
      landmarker: null,
      stream: null,
      video: null,
    };
    resourcesRef.current = resources;
    let permission = "requesting";
    setObservation(unavailableVisualObservation(permission));

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        const unsupported = new Error("Camera API unavailable");
        unsupported.name = "CameraUnsupportedError";
        throw unsupported;
      }

      resources.stream = await navigator.mediaDevices.getUserMedia(
        CAMERA_CONSTRAINTS,
      );
      if (operation !== operationRef.current) {
        releaseResources(resources);
        return;
      }

      permission = "granted";
      resources.video = videoRef.current;
      if (!resources.video) {
        throw new Error("Camera preview unavailable");
      }
      resources.video.srcObject = resources.stream;
      await resources.video.play();
      setObservation({
        ...unavailableVisualObservation(permission),
        cue: "loading_model",
      });

      resources.landmarker = await createLocalFaceLandmarker();
      if (operation !== operationRef.current) {
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
      const render = (timestamp) => {
        if (
          operation !== operationRef.current ||
          resourcesRef.current !== resources
        ) {
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
            setObservation(
              summarizeFaceLandmarkerResult(result, {
                inferenceMs: performance.now() - startedAt,
                observedAtMs: Date.now(),
              }),
            );
          } catch {
            operationRef.current += 1;
            releaseResources(resources);
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
      setObservation(
        unavailableVisualObservation(
          error?.name === "NotAllowedError" ? "denied" : permission,
          cameraErrorMessage(error),
        ),
      );
    }
  }, [releaseResources]);

  useEffect(
    () => () => {
      operationRef.current += 1;
      releaseResources();
    },
    [releaseResources],
  );

  return { observation, videoRef, enableCamera, disableCamera };
}
