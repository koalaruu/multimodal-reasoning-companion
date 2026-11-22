import wasmBinaryPath from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import wasmLoaderPath from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";

export const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export async function createLocalFaceLandmarker() {
  const { FaceLandmarker } = await import("@mediapipe/tasks-vision");

  return FaceLandmarker.createFromOptions(
    { wasmBinaryPath, wasmLoaderPath },
    {
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL },
      runningMode: "VIDEO",
      numFaces: 2,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: false,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    },
  );
}
