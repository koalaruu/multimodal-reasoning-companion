// Isolated transport/retry regression; API responses are controlled, not real Groq calls.
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { withBrowserPage } from "./browserHarness.js";

const apiBase = "http://127.0.0.1:8000";
process.env.VITE_API_BASE_URL = apiBase;
const fixtureDir = await mkdtemp(path.join(tmpdir(), "companion-message-cv-"));
const cvPath = path.join(fixtureDir, "synthetic-cv.txt");
const cvText = "Synthetic CV for message recovery verification.";

try {
  await writeFile(cvPath, cvText);
  await withBrowserPage(async ({ page, baseUrl }) => {
    const errors = [];
    const turns = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.setRequestInterception(true);
    const respond = (request, body, status = 200) => request.respond({
      status, contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" }, body: JSON.stringify(body),
    });
    page.on("request", (request) => {
      if (!request.url().startsWith(apiBase + "/")) return void request.continue();
      if (request.method() === "OPTIONS") return void request.respond({ status: 204, headers: {
        "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
      } });
      if (request.url().endsWith("/v1/cv/extract")) return void respond(request, {
        filename: "synthetic-cv.txt", text: cvText, characters: cvText.length, preview: cvText,
      });
      if (request.url().endsWith("/v1/interview/turn")) turns.push(request);
      else if (!request.url().endsWith("/v1/interview/review")) void request.abort();
      // Hold turn/review responses until the scenario resolves them.
    });
    const navigate = (view) => page.click(`button[aria-controls="${view}-view"]`);
    const sendButton = '.answer-form button[type="submit"]';
    const hold = async (endpoint, selector) => {
      const pending = page.waitForRequest((request) =>
        request.url() === `${apiBase}/v1/interview/${endpoint}` && request.method() === "POST");
      await page.click(selector);
      return pending;
    };
    const replaceInput = async (selector, value = "") => {
      await page.click(selector);
      await page.keyboard.down("Control");
      await page.keyboard.press("a");
      await page.keyboard.up("Control");
      await page.keyboard.press("Backspace");
      if (value) await page.type(selector, value);
    };
    const candidateTexts = () => page.$$eval(".message.candidate p", (nodes) =>
      nodes.map((node) => node.textContent));
    const waitForFailure = async (text) => {
      await page.waitForSelector(".message.candidate.failed .message-delivery");
      await page.waitForFunction(() => {
        const label = document.querySelector(".message-delivery").getBoundingClientRect();
        const list = document.querySelector(".message-list").getBoundingClientRect();
        return label.top >= list.top && label.bottom <= list.bottom;
      });
      await page.waitForFunction(() => !document.querySelector("#candidate-answer").disabled);
      assert.deepEqual(await candidateTexts(), [text]);
      assert.equal(await page.$eval("#candidate-answer", (input) => input.value), text);
      assert.equal(await page.$eval(sendButton, (button) => button.textContent.trim()), "Retry answer");
      assert.equal(await page.$(".message.candidate.sent"), null);
    };
    const finishTurn = async (request, reply, fallback = false) => {
      const { state, candidate_input } = JSON.parse(request.postData());
      await respond(request, {
        reply, reply_source: fallback ? "local_fallback" : "local",
        fallback_reason: fallback ? "sdk_error" : null,
        signals: {}, content_evidence: {},
        decision: { action: "ask_followup", reason: "evaluation_evidence_missing", followup_topic: "evaluation_method" },
        updated_state: { ...state, turn_index: (state.turn_index ?? 0) + 1,
          conversation: [...state.conversation,
            { role: "candidate", content: candidate_input },
            { role: "interviewer", content: reply }],
        },
      });
      await page.waitForFunction(() => !document.querySelector("#candidate-answer").disabled);
    };

    await page.setViewport({ width: 1365, height: 768 });
    await page.goto(baseUrl, { waitUntil: "networkidle0" });
    const initialMessage = await page.$eval(".message.assistant p", (node) => node.textContent);
    await navigate("context");
    await page.type('.field input[type="text"]', "Flutter Developer");
    await page.type(".field textarea", "Build Flutter screens.");
    await (await page.$('input[type="file"]')).uploadFile(cvPath);
    await page.waitForSelector(".cv-loaded");
    await navigate("practice");
    const original = "Original answer that was not confirmed.";
    await page.type("#candidate-answer", original);
    const first = await hold("turn", sendButton);
    const initialState = JSON.parse(first.postData()).state;
    assert.equal(initialState.cv_text, cvText);
    assert.deepEqual(initialState.conversation, [{ role: "interviewer", content: initialMessage }]);
    assert.equal(await page.$eval(sendButton, (button) => button.disabled), true);
    await page.click(sendButton);
    assert.equal(turns.length, 1);
    await first.abort("failed");
    await waitForFailure(original);
    assert.equal(turns.length, 1); // No automatic retry.

    // Editing or clearing context after failure must affect the retry, not revive its old snapshot.
    await navigate("context");
    await page.click(".cv-actions button");
    await replaceInput('.field input[type="text"]', "Support Engineer");
    await replaceInput(".field textarea", "Troubleshoot APIs and explain customer checks.");
    await navigate("practice");
    const revised = "Revised answer with a clearer work example.";
    await replaceInput("#candidate-answer", revised);
    const retry = await hold("turn", sendButton);
    const retryBody = JSON.parse(retry.postData());
    assert.deepEqual(await candidateTexts(), [revised]);
    assert.equal(await page.$(".message.candidate.failed"), null);
    assert.equal(retryBody.candidate_input, revised);
    assert.equal(retryBody.state.session_id, initialState.session_id);
    assert.equal(retryBody.state.turn_index ?? 0, 0);
    assert.deepEqual(retryBody.state.conversation, initialState.conversation);
    assert.equal(retryBody.state.cv_text, null);
    assert.equal(retryBody.state.cv_filename, null);
    assert.equal(retryBody.state.target_role, "Support Engineer");
    assert.equal(retryBody.state.job_description, "Troubleshoot APIs and explain customer checks.");
    await respond(retry, {}, 503);
    await waitForFailure(revised);

    // A successful HTTP turn with a labelled local fallback is still a confirmed delivery.
    const successfulRetry = await hold("turn", sendButton);
    assert.deepEqual(JSON.parse(successfulRetry.postData()).state.conversation, initialState.conversation);
    const firstReply = "How did you check your work?";
    await finishTurn(successfulRetry, firstReply, true);
    assert.deepEqual(await candidateTexts(), [revised]);
    assert.equal(await page.$$eval(".message.candidate.sent", (nodes) => nodes.length), 1);
    assert.equal(await page.$(".message.candidate.failed"), null);
    assert.equal(await page.$eval("#candidate-answer", (input) => input.value), "");
    assert.equal(await page.$eval(sendButton, (button) => button.textContent.trim()), "Send answer");
    assert.deepEqual(await page.$$eval(".message.assistant p", (nodes) => nodes.map((node) => node.textContent)),
      [initialMessage, firstReply]);

    const nextAnswer = "A second confirmed answer.";
    await page.type("#candidate-answer", nextAnswer);
    const next = await hold("turn", sendButton);
    const nextState = JSON.parse(next.postData()).state;
    assert.equal(nextState.turn_index, 1);
    assert.deepEqual(nextState.conversation.filter((message) => message.role === "candidate")
      .map((message) => message.content), [revised]);
    await finishTurn(next, "Which result did you observe?");
    assert.deepEqual(await candidateTexts(), [revised, nextAnswer]);

    // Review sees only accepted turns, never failed attempts or duplicate candidate entries.
    await navigate("insights");
    const reviewRequest = await hold("review", ".session-review > button");
    const reviewedState = JSON.parse(reviewRequest.postData()).state;
    assert.equal(reviewedState.turn_index, 2);
    assert.deepEqual(reviewedState.conversation.filter((message) => message.role === "candidate")
      .map((message) => message.content), [revised, nextAnswer]);
    await respond(reviewRequest, {
      turns: 2, covered: [], missing: [], suggestions: [], questions_to_ask: [], review_source: "local",
    });
    await page.waitForSelector(".review-freshness.current");

    // A malformed success body must not add an assistant or advance the confirmed session.
    await navigate("practice");
    const unconfirmed = "Unconfirmed third answer.";
    await page.type("#candidate-answer", unconfirmed);
    const malformed = await hold("turn", sendButton);
    await respond(malformed, { reply: "Invalid payload", decision: null });
    await page.waitForSelector(".message.candidate.failed .message-delivery");
    await page.waitForFunction(() => !document.querySelector("#candidate-answer").disabled);
    assert.deepEqual(await candidateTexts(), [revised, nextAnswer, unconfirmed]);
    assert.equal(await page.$$eval(".message.candidate.failed", (nodes) => nodes.length), 1);
    assert.equal(await page.$$eval(".message.assistant", (nodes) => nodes.length), 3);
    await navigate("insights");
    await page.waitForSelector(".review-freshness.current");
    assert.ok((await page.$eval(".review-freshness", (node) => node.textContent)).includes("2 completed turns"));

    await page.click(".sidebar-session button");
    assert.equal(await page.$(".message.candidate"), null);
    assert.equal(await page.$eval("#candidate-answer", (input) => input.value), "");
    assert.equal(await page.$eval(sendButton, (button) => button.textContent.trim()), "Send answer");
    assert.equal(await page.$eval(".brief-actions > .primary-button", (button) => button.disabled), false);
    assert.deepEqual(errors, []);
    console.log("PASS message recovery: network/HTTP/malformed failures, edited retries without duplicates, current context, confirmed history, and reset.");
  });
} finally {
  await rm(fixtureDir, { recursive: true, force: true });
}
