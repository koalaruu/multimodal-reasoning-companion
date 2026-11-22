import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { createServer } from "vite";
import { validateVisionDataset } from "../src/vision/visionEvaluation.js";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(frontendRoot, "..");
const defaults = {
  cases: path.join(repositoryRoot, "eval", "vision_cases.json"),
  output: path.join(repositoryRoot, "eval", "results", "vision-latest.json"),
};

function parseArguments(argv) {
  const args = {
    cases: defaults.cases,
    output: defaults.output,
    validateOnly: false,
    enforceTarget: false,
    headed: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--validate-only") args.validateOnly = true;
    else if (argument === "--enforce-target") args.enforceTarget = true;
    else if (argument === "--headed") args.headed = true;
    else if (argument === "--cases") args.cases = path.resolve(argv[++index]);
    else if (argument === "--output") args.output = path.resolve(argv[++index]);
    else throw new Error(`unknown argument: ${argument}`);
  }
  return args;
}

function printReport(report) {
  for (const outcome of report.cases) {
    if (outcome.status === "completed") {
      console.log(
        `${outcome.exact_match ? "PASS" : "FAIL"} ${outcome.split}/${outcome.id} ` +
          `faces=${outcome.detected_face_count}/${outcome.expected_face_count} ` +
          `p50=${outcome.median_inference_ms}ms p95=${outcome.p95_inference_ms}ms`,
      );
    } else {
      console.log(`FAIL ${outcome.split}/${outcome.id} failure=${outcome.failure}`);
    }
  }
  for (const split of ["calibration", "held_out"]) {
    const summary = report.summary[split];
    console.log(
      `${split}: success=${summary.runtime_successful}/${summary.total} ` +
        `face_count_accuracy=${summary.face_count_accuracy} ` +
        `false_positive_rate=${summary.no_face_false_positive_rate} ` +
        `p95=${summary.p95_inference_ms}ms`,
    );
  }
  console.log(`Target status: ${report.acceptance.passed ? "PASS" : "NOT MET"}`);
}

async function launchChrome(headed) {
  const executablePath = process.env.CHROME_PATH || process.env.PUPPETEER_EXECUTABLE_PATH;
  const options = { headless: !headed };
  if (executablePath) options.executablePath = executablePath;
  else options.channel = process.env.CHROME_CHANNEL || "chrome";
  try {
    return await puppeteer.launch(options);
  } catch (error) {
    throw new Error(
      "Chrome could not start. Install Google Chrome or set CHROME_PATH to its executable.",
      { cause: error },
    );
  }
}

async function executeBenchmark(dataset, headed) {
  const server = await createServer({
    root: frontendRoot,
    logLevel: "error",
    server: { host: "127.0.0.1", port: 0, strictPort: false },
  });
  let browser = null;
  try {
    await server.listen();
    const baseUrl = server.resolvedUrls?.local?.[0];
    if (!baseUrl) throw new Error("Vite did not expose a local benchmark URL.");

    browser = await launchChrome(headed);
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });
    await page.goto(baseUrl, { waitUntil: "networkidle0" });
    return await page.evaluate(async (manifest) => {
      const { runVisionBenchmark } = await import("/src/vision/visionBenchmark.js");
      return runVisionBenchmark(manifest);
    }, dataset);
  } finally {
    await browser?.close();
    await server.close();
  }
}

async function main() {
  const args = parseArguments(process.argv.slice(2));
  const dataset = validateVisionDataset(
    JSON.parse(await readFile(args.cases, "utf8")),
  );
  if (args.validateOnly) {
    console.log(`Valid vision dataset: ${dataset.assets.length} assets, ${dataset.cases.length} cases`);
    return 0;
  }

  const report = await executeBenchmark(dataset, args.headed);
  await mkdir(path.dirname(args.output), { recursive: true });
  await writeFile(args.output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  printReport(report);
  return args.enforceTarget && !report.acceptance.passed ? 1 : 0;
}

main()
  .then((exitCode) => {
    process.exitCode = exitCode;
  })
  .catch((error) => {
    console.error(`Vision evaluation error: ${error.message}`);
    process.exitCode = 1;
  });
