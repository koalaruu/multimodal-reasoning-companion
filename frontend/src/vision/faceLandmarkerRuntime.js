import wasmBinaryPath from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import wasmLoaderPath from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import {
  FACE_LANDMARKER_MAX_FACES,
  FACE_LANDMARKER_MODEL_URL,
} from "./visionRuntimeConfig.js";

export { FACE_LANDMARKER_MODEL_URL } from "./visionRuntimeConfig.js";

export async function createLocalFaceLandmarker({ runningMode = "VIDEO" } = {}) {
  const { FaceLandmarker } = await import("@mediapipe/tasks-vision");

  return FaceLandmarker.createFromOptions(
    { wasmBinaryPath, wasmLoaderPath },
    {
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL },
      runningMode,
      numFaces: FACE_LANDMARKER_MAX_FACES,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    },
  );
}
