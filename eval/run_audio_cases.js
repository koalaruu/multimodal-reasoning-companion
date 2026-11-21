import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { buildAudioReport, loadAudioDataset } from "./audio_evaluation.js";

const directory = path.dirname(fileURLToPath(import.meta.url));
const casesPath = path.join(directory, "audio_cases.json");
const resultsPath = path.join(directory, "results", "audio-latest.json");

export async function runAudioEvaluation() {
  const dataset = await loadAudioDataset(casesPath);
  const report = buildAudioReport(dataset);

  console.table(
    report.threshold_results.map((result) => ({
      threshold_ms: result.threshold_ms,
      calibration_false_ready_rate:
        result.calibration.false_ready_rate,
      calibration_mean_delay_ms:
        result.calibration.mean_detection_delay_ms,
      held_out_false_ready_rate: result.held_out.false_ready_rate,
      held_out_mean_delay_ms: result.held_out.mean_detection_delay_ms,
    })),
  );
  console.log(`Selected threshold: ${report.selected_threshold_ms} ms`);
  console.table(
    report.selected_held_out_cases.map((testCase) => ({
      id: testCase.id,
      false_ready: testCase.false_ready,
      detection_delay_ms: testCase.detection_delay_ms,
      passed: testCase.passed,
    })),
  );
  console.log(
    `Acceptance: ${report.acceptance.passed ? "PASS" : "FAIL"}`,
  );

  await mkdir(path.dirname(resultsPath), { recursive: true });
  await writeFile(resultsPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  return report;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const report = await runAudioEvaluation();
  process.exitCode = report.acceptance.passed ? 0 : 1;
}
