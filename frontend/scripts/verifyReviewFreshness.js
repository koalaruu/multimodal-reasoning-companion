// Controlled API responses exercise review snapshots, not Groq's prose quality.
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { withBrowserPage } from "./browserHarness.js";

const apiBase = "http://127.0.0.1:8000";
process.env.VITE_API_BASE_URL = apiBase;
const fixtureDir = await mkdtemp(path.join(tmpdir(), "companion-review-cv-"));
const cvPath = path.join(fixtureDir, "synthetic-cv.txt");
const cvText = "Synthetic CV: built a Flutter expense tracker and used widget tests.";

try {
  await writeFile(cvPath, cvText);
  await withBrowserPage(async ({ page, baseUrl }) => {
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.setRequestInterception(true);
    const respond = (request, body, status = 200) => request.respond({
      status,
      contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      body: JSON.stringify(body),
    });
    page.on("request", (request) => {
      if (!request.url().startsWith(apiBase + "/")) return void request.continue();
      if (request.method() === "OPTIONS") {
        return void request.respond({ status: 204, headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        } });
      }
      if (request.url().endsWith("/v1/cv/extract")) {
        return void respond(request, {
          filename: "synthetic-cv.txt", text: cvText,
          characters: cvText.length, preview: cvText,
        });
      }
      if (!request.url().endsWith("/v1/interview/turn") &&
          !request.url().endsWith("/v1/interview/review")) void request.abort();
      // Review and turn requests stay pending until each scenario resolves them.
    });
    const navigate = (view) => page.click(`button[aria-controls="${view}-view"]`);
    const reviewButton = ".session-review > button";
    const holdRequest = async (endpoint, action) => {
      const pending = page.waitForRequest((request) =>
        request.url() === `${apiBase}/v1/interview/${endpoint}` && request.method() === "POST");
      await action();
      return pending;
    };
    const holdTurn = async (answer) => {
      await navigate("practice");
      await page.type("#candidate-answer", answer);
      return holdRequest("turn", () => page.click('.answer-form button[type="submit"]'));
    };
    const finishTurn = async (request, status = 200) => {
      const { state, candidate_input } = JSON.parse(request.postData());
      const reply = "How did you check your work?";
      await respond(request, {
        reply, reply_source: "local", signals: {}, content_evidence: {},
        decision: { action: "converse", reason: "groq_led_turn", followup_topic: "none" },
        updated_state: {
          ...state, turn_index: (state.turn_index ?? 0) + 1,
          cumulative_evidence: {
            input_type: "answer", relevance: "relevant", project_described: true,
            evaluation_method_mentioned: true, evaluation_result_mentioned: true,
          },
          conversation: [...state.conversation,
            { role: "candidate", content: candidate_input },
            { role: "interviewer", content: reply }],
        },
      }, status);
      await page.waitForFunction(() => !document.querySelector("#candidate-answer").disabled);
    };
    const holdReview = async () => {
      await navigate("insights");
      return holdRequest("review", () => page.click(reviewButton));
    };
    const finishReview = async (request, marker, status = 200) => {
      const { state } = JSON.parse(request.postData());
      await respond(request, {
        turns: state.turn_index,
        covered: ["a specific work example", "how the work was checked", "a concrete outcome"],
        missing: [], suggestions: [marker], questions_to_ask: [], review_source: "local",
      }, status);
      await page.waitForFunction(() => !document.querySelector(".session-review > button").disabled);
    };
    const expectSnapshot = async (state, turns, marker) => {
      await page.waitForSelector(`.review-freshness.${state}`);
      const label = await page.$eval(".review-freshness", (element) => element.textContent);
      assert.ok(label.includes(`${turns} completed turns`), label);
      assert.ok((await page.$eval(".review-body", (element) => element.textContent)).includes(marker));
    };

    await page.goto(baseUrl, { waitUntil: "networkidle0" });
    await finishTurn(await holdTurn("I built a Flutter expense tracker."));
    await finishTurn(await holdTurn("I used widget tests and improved load time."));
    await finishReview(await holdReview(), "Advice for snapshot two");
    await expectSnapshot("current", 2, "Advice for snapshot two");

    // A pending answer immediately makes the prior review outdated.
    const thirdTurn = await holdTurn("I also checked integration tests.");
    await navigate("insights");
    await expectSnapshot("outdated", 2, "Advice for snapshot two");
    assert.equal(await page.$eval(reviewButton, (button) => button.disabled), true);
    await finishTurn(thirdTurn);
    await expectSnapshot("outdated", 2, "Advice for snapshot two");
    assert.equal(await page.$eval(reviewButton, (button) => button.textContent.trim()), "Update review");

    // A response requested at turn 3 must not become current after turn 4 finishes.
    const delayedReview = await holdReview();
    await finishTurn(await holdTurn("I measured the result on a slower device."));
    await navigate("insights");
    await finishReview(delayedReview, "Delayed advice for snapshot three");
    await expectSnapshot("outdated", 3, "Delayed advice for snapshot three");
    await finishReview(await holdReview(), "Advice for snapshot four");
    await expectSnapshot("current", 4, "Advice for snapshot four");

    // Role/JD edits invalidate at the same turn count.
    await navigate("context");
    await page.type('.field input[type="text"]', "Flutter Developer");
    await page.type(".field textarea", "Write widget tests and improve load time.");
    await navigate("insights");
    await expectSnapshot("outdated", 4, "Advice for snapshot four");
    const roleReview = await holdReview();
    const roleState = JSON.parse(roleReview.postData()).state;
    assert.equal(roleState.target_role, "Flutter Developer");
    assert.equal(roleState.job_description, "Write widget tests and improve load time.");
    await finishReview(roleReview, "Advice for role context");
    await expectSnapshot("current", 4, "Advice for role context");

    // Mode changes and unsubmitted drafts are not review-input changes.
    await navigate("context");
    await page.click('input[value="coach"]');
    await navigate("practice");
    await page.type("#candidate-answer", "Unsubmitted draft");
    await navigate("insights");
    await expectSnapshot("current", 4, "Advice for role context");

    // CV changes invalidate too; refresh failure keeps the old result labelled outdated.
    await navigate("context");
    const input = await page.$('input[type="file"]');
    await input.uploadFile(cvPath);
    await page.waitForSelector(".cv-loaded");
    await navigate("insights");
    await expectSnapshot("outdated", 4, "Advice for role context");
    await finishReview(await holdReview(), "Unused failed response", 503);
    await page.waitForSelector(".session-review .vision-error");
    await expectSnapshot("outdated", 4, "Advice for role context");
    await finishReview(await holdReview(), "Advice for uploaded CV");
    await expectSnapshot("current", 4, "Advice for uploaded CV");
    await navigate("context");
    await page.click(".cv-actions button");
    await navigate("insights");
    await expectSnapshot("outdated", 4, "Advice for uploaded CV");
    const clearedReview = await holdReview();
    assert.equal(JSON.parse(clearedReview.postData()).state.cv_text, null);
    await finishReview(clearedReview, "Advice after removing CV");
    await expectSnapshot("current", 4, "Advice after removing CV");

    // A failed answer did not change completed answers: the matching review stays valid.
    const failedTurn = await holdTurn(" (continued)");
    await navigate("insights");
    await expectSnapshot("outdated", 4, "Advice after removing CV");
    await finishTurn(failedTurn, 503);
    await expectSnapshot("current", 4, "Advice after removing CV");

    // Reset unmounts the snapshot and cancels the pending client request.
    const abandonedReview = await holdReview();
    const cancellation = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        page.off("requestfailed", onFailure);
        reject(new Error("Reset did not cancel the old review request"));
      }, 10_000);
      function onFailure(request) {
        if (request !== abandonedReview) return;
        clearTimeout(timer);
        page.off("requestfailed", onFailure);
        resolve(request.failure()?.errorText);
      }
      page.on("requestfailed", onFailure);
    });
    await page.click(".sidebar-session button");
    assert.match(await cancellation, /ERR_ABORTED/);
    await navigate("insights");
    assert.equal(await page.$(".review-body"), null);
    assert.equal(await page.$eval(reviewButton, (button) => button.disabled), true);
    assert.deepEqual(errors, []);
    console.log("PASS review freshness: pending/completed turns, late response, context/CV changes, failed refresh, non-input edits, and reset cancellation.");
  });
} finally {
  await rm(fixtureDir, { recursive: true, force: true });
}
