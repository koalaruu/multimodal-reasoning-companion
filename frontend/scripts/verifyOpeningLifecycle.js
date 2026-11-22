// Isolated UI regression: controlled API responses, no FastAPI or Groq key needed.
import assert from "node:assert/strict";
import { withBrowserPage } from "./browserHarness.js";

const apiBase = "http://127.0.0.1:8000";
process.env.VITE_API_BASE_URL = apiBase;
await withBrowserPage(async ({ page, baseUrl }) => {
  const openingRequests = [];
  const turnRequests = [];
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    if (!request.url().startsWith(apiBase + "/")) return void request.continue();
    if (request.method() === "OPTIONS") {
      return void request.respond({
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
        },
      });
    }
    // Hold API requests until the scenario explicitly resolves them.
    if (request.url().endsWith("/v1/interview/opening")) openingRequests.push(request);
    else if (request.url().endsWith("/v1/interview/turn")) turnRequests.push(request);
    else void request.abort();
  });

  const respond = (request, body, status = 200) => request.respond({
    status,
    contentType: "application/json",
    headers: { "Access-Control-Allow-Origin": "*" },
    body: JSON.stringify(body),
  });
  const holdRequest = async (endpoint, selector) => {
    const pending = page.waitForRequest((request) =>
      request.url() === `${apiBase}/v1/interview/${endpoint}` && request.method() === "POST");
    await page.click(selector);
    return pending;
  };
  const openingButton = ".brief-actions > .primary-button";
  const contextOpeningButton = ".context-guide > .primary-button";
  const sendButton = ".answer-form button[type=submit]";
  const openingFlags = () => page.$$eval(
    ".brief-actions > .primary-button, .context-guide > .primary-button",
    (buttons) => buttons.map((button) => button.disabled),
  );
  const messages = () => page.$$eval(".message p", (nodes) => nodes.map((node) => node.textContent));
  const waitForIdle = () => page.waitForFunction(() =>
    !document.querySelector("#candidate-answer").disabled);

  await page.goto(baseUrl, { waitUntil: "networkidle0" });
  const initialMessages = await messages();
  const draft = "I built a Flutter application.";
  await page.type("#candidate-answer", draft);

  // Pending and failed opening: no answer/reset can race it, draft survives.
  const failedOpening = await holdRequest("opening", openingButton);
  await page.waitForFunction(() => document.querySelector("#candidate-answer").disabled);
  assert.deepEqual(await openingFlags(), [true, true]);
  assert.equal(await page.$eval(sendButton, (button) => button.disabled), true);
  assert.equal(await page.$eval(".sidebar-session button", (button) => button.disabled), true);
  await page.click('button[aria-controls="insights-view"]');
  assert.equal(await page.$eval(".monitor-heading button", (button) => button.disabled), true);
  await page.click('button[aria-controls="practice-view"]');
  await page.click(sendButton);
  assert.equal(turnRequests.length, 0);
  await respond(failedOpening, {}, 503);
  await waitForIdle();
  assert.deepEqual(await messages(), initialMessages);
  assert.equal(await page.$eval("#candidate-answer", (input) => input.value), draft);
  assert.deepEqual(await openingFlags(), [false, false]);

  // Regeneration is allowed before the first answer and replaces only the opening.
  for (const reply of ["Which Flutter project would you like to discuss?", "How did you approach your Flutter project?"]) {
    const opening = await holdRequest("opening", openingButton);
    await respond(opening, { reply, reply_source: "local" });
    await waitForIdle();
    assert.deepEqual(await messages(), [reply]);
  }
  const openingText = (await messages())[0];

  // The first pending answer locks both opening controls and both reset controls.
  const firstTurn = await holdRequest("turn", sendButton);
  const firstRequest = JSON.parse(firstTurn.postData());
  assert.deepEqual(firstRequest.state.conversation, [{ role: "interviewer", content: openingText }]);
  await page.waitForFunction(() => document.querySelector("#practice-opening-note"));
  assert.deepEqual(await openingFlags(), [true, true]);
  await page.click('button[aria-controls="context-view"]');
  assert.equal(await page.$eval(contextOpeningButton, (button) => button.disabled), true);
  await page.click('button[aria-controls="insights-view"]');
  assert.equal(await page.$eval(".monitor-heading button", (button) => button.disabled), true);
  await page.click('button[aria-controls="practice-view"]');
  const reply = "How did you check the result?";
  await respond(firstTurn, {
    reply,
    reply_source: "local",
    decision: { action: "ask_followup", reason: "evaluation_evidence_missing", followup_topic: "evaluation_method" },
    signals: {},
    content_evidence: {},
    updated_state: {
      ...firstRequest.state,
      turn_index: 1,
      conversation: [
        ...firstRequest.state.conversation,
        { role: "candidate", content: firstRequest.candidate_input },
        { role: "interviewer", content: reply },
      ],
    },
  });
  await waitForIdle();
  await page.click(openingButton);
  assert.equal(openingRequests.length, 3);
  assert.deepEqual(await openingFlags(), [true, true]);
  assert.deepEqual(await messages(), [openingText, draft, reply]);

  // Only an explicit new session re-enables opening generation.
  await page.click(".sidebar-session button");
  assert.deepEqual(await messages(), initialMessages);
  assert.deepEqual(await openingFlags(), [false, false]);
  await page.type("#candidate-answer", draft);
  const failedTurn = await holdRequest("turn", sendButton);
  const freshRequest = JSON.parse(failedTurn.postData());
  assert.notEqual(freshRequest.state.session_id, firstRequest.state.session_id);
  assert.deepEqual(freshRequest.state.conversation, [{ role: "interviewer", content: initialMessages[0] }]);
  await respond(failedTurn, {}, 503);
  await waitForIdle();
  // Even a failed answer is visible in history; never replace its opening underneath it.
  assert.deepEqual(await openingFlags(), [true, true]);
  assert.deepEqual(await messages(), [...initialMessages, draft]);
  await page.click(".sidebar-session button");
  assert.deepEqual(await openingFlags(), [false, false]);
  assert.deepEqual(await messages(), initialMessages);
  assert.deepEqual(pageErrors, []);
  console.log("PASS opening lifecycle: pending/failure, pre-answer regeneration, active-session lock, and explicit reset.");
});
