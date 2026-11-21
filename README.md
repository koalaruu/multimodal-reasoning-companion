# Multimodal Reasoning Companion

A CPU-first adaptive interview-practice assistant built as an AI systems portfolio project. The project separates observable input processing, internal state, dialogue policy, and response generation so each decision can be tested without relying on one large model.

The current milestone is a **React interview interface with local Faster-Whisper speech-to-text, audio turn-taking cues, and explainable answer-quality evidence**, backed by a stateless FastAPI service with optional Gemini reply generation. Dialogue policy and state updates remain deterministic and fully testable offline; vision is not implemented.

## Research question

> Compared with a transcript-only assistant, can speech-activity and optional visual observations later improve when the assistant waits, offers a hint, or asks a follow-up without making unsupported claims about the user's mental state?

The text baseline establishes the control condition for that future comparison.

## Architecture

```mermaid
flowchart LR
    M[Opt-in microphone] --> N[Browser voice<br/>activity detection]
    N --> L
    M --> O[Explicit MediaRecorder<br/>voice answer]
    O --> P[FastAPI bounded<br/>audio endpoint]
    P --> Q[Faster-Whisper<br/>CPU int8]
    Q --> R[Editable transcript]
    R --> L
    L[React interview UI] --> K[FastAPI text transport]
    K --> A[Candidate text]
    A --> B[Perception<br/>explicit requests]
    A --> C[Perception<br/>answer-quality evidence]
    B --> D[Deterministic<br/>dialogue policy]
    C --> D
    E[Session state] --> D
    D --> F[Immutable state<br/>transition]
    D --> G{Eligible action and<br/>Gemini configured?}
    G -->|No| H[Local reply]
    G -->|Yes| I[Gemini structured<br/>reply only]
    I -->|Failure or invalid| H
    F --> J[Turn result]
    H --> J
    I --> J
    J --> K
    K --> L
```

| Layer | Responsibility | Must not do |
|---|---|---|
| `perception` | Detect explicit controls and answer evidence; transcribe explicit audio clips locally | Infer emotion, score the candidate, or choose the final action |
| `cognitive` | Apply immutable session-state transitions | Extract evidence, generate dialogue, or call a model |
| `reasoning` | Apply one documented policy, render local replies, and optionally request one structured Gemini reply | Allow Gemini to change an action, reason, or state |
| `service` | Orchestrate one text turn and expose reply source/fallback metadata | Persist transcripts or duplicate policy rules |
| `api` | Validate text requests and bounded audio streams, then delegate to the correct local component | Store sessions or media, duplicate policy, or expose internal errors |
| `frontend` | Keep the session in memory, derive voice-activity cues, and record only explicit voice answers | Auto-submit transcripts, persist media, or infer psychological state |
| `eval` | Exercise deterministic domain and timing behavior | Start Gemini or claim human validity from fixtures |

Control and answer precedence is:

```text
ended session > stop > thinking time > hint > missing input
> irrelevant/unclear answer > class-level follow-up
> missing project > missing evaluation > continue
```

## Current behavior

The baseline supports seven actions:

- `offer_hint`
- `wait`
- `ask_followup`
- `end_session`
- `continue_interview`
- `request_input`
- `request_clarification`

Every turn returns explicit signals; input type; answer relevance; project, evaluation-method, and evaluation-result evidence; the selected action and reason; updated session state; reply; `reply_source`; and optional `fallback_reason`. `SessionState` contains counters and control state only; it does not retain candidate text.

The evidence extractor uses documented, precompiled lexical patterns rather than claiming general language understanding. It identifies concrete project descriptions, evaluation terminology, numeric results, and explicit controls. Irrelevant or unclear answers receive `request_clarification`. `wait`, `end_session`, and `request_input` are always rendered locally. `offer_hint`, `ask_followup`, `continue_interview`, and `request_clarification` are eligible for optional Gemini wording.

## Quickstart

Use official CPython 3.11 or newer and Node.js 22 or newer. Install and verify the backend from the repository root:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install -r backend/requirements-stt.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m eval.run_cases
.venv\Scripts\python -m eval.answer_quality_evaluation --enforce-target
```

Start FastAPI in the first terminal:

```powershell
.venv\Scripts\python -m uvicorn backend.app.main:app --reload
```

Start React in a second terminal:

```powershell
cd frontend
npm install
npm test
npm run dev
```

Open `http://127.0.0.1:5173`. To use a different backend URL, copy `frontend/.env.example` to `frontend/.env` and change `VITE_API_BASE_URL`.

### Local speech-to-text

`backend/requirements-stt.txt` installs Faster-Whisper 1.2.1 and pins PyAV below 19 because that Faster-Whisper release still calls the pre-19 `av.open` API. Defaults are CPU `int8`, English `base.en`, and a 60-second decoded-audio limit:

```powershell
$env:WHISPER_MODEL = "base.en"
$env:WHISPER_DEVICE = "cpu"
$env:WHISPER_COMPUTE_TYPE = "int8"
$env:WHISPER_LANGUAGE = "en"
$env:WHISPER_MAX_AUDIO_SECONDS = "60"
```

The first transcription downloads model weights from Hugging Face and loads them lazily. Later requests use the local cache. Model weights may persist in the Hugging Face cache; candidate audio does not. The project reads process environment variables and does not silently load `.env`.

### Optional Gemini replies

The project uses the official `google-genai` SDK and defaults to the current stable low-latency model `gemini-3.8-flash`. Copy `.env.example` as a reference, then set variables in the process environment; the project does not silently load `.env` files.

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GEMINI_MODEL = "gemini-3.8-flash"
$env:GEMINI_TIMEOUT_SECONDS = "8"
```

The FastAPI composition root reads these variables when the server starts and constructs `InterviewService` once. No key keeps the API in local response mode.

An absent or blank key keeps local mode. Invalid timeout configuration fails at construction only when a key is present. Timeout, SDK error, malformed structured output, or invalid reply uses the exact local response and returns `reply_source="local_fallback"` with a machine-readable reason. There are no application-level retries.

Run backend commands from the repository root and npm commands from `frontend/`.

## Text API

The API is stateless. Clients send the current `SessionState` with every turn and receive the updated state in the response.

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | Returns `{\"status\":\"ok\"}` without invoking the model |
| `POST` | `/v1/interview/turn` | Runs one deterministic interview turn |
| `POST` | `/v1/audio/transcriptions` | Transcribes a bounded raw WebM, Ogg, MP4, FLAC, or WAV body locally |
| `GET` | `/docs` | Interactive OpenAPI documentation |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/interview/turn \
  -H \"Content-Type: application/json\" \
  -d '{
    \"candidate_input\": \"I built a classifier for support tickets.\",
    \"state\": {\"session_id\": \"demo-session\"}
  }'
```

The response includes observable evidence, the local policy decision, updated state, reply, `reply_source`, and optional `fallback_reason`. Schema errors return HTTP `422`. Gemini failures still return HTTP `200` with the exact local reply and `reply_source=\"local_fallback\"`.

Browser access is restricted to the local Vite origins `http://localhost:5173` and `http://127.0.0.1:5173`.

## Text interview UI

The React client keeps session state and conversation history in memory only. It provides:

- a text interview flow across multiple turns;
- an evidence and deterministic policy decision monitor;
- visible `local`, `gemini`, and `local_fallback` reply sources;
- explicit voice recording that appends a local transcript to the editable answer;
- a reset action that discards the in-memory transcript and creates a new session;
- an explicit API connectivity error without exposing backend internals.

The UI does not infer or display stress, mood, focus, personality, or hiring suitability.

## Audio turn-taking and local speech-to-text

Audio remains optional and starts only after **Enable microphone**. The Web Audio API continuously derives observable voice-activity timing in the browser: microphone availability, current speech-like activity, cumulative speech duration, and trailing silence. It does not retain those samples.

The readiness policy distinguishes `speaking`, `listening`, `thinking_pause`, and `ready`. A pause shorter than 1.6 seconds remains `thinking_pause`. This timing cue never submits text or changes a dialogue action.

Speech-to-text is a separate, explicit path:

1. **Start voice answer** starts `MediaRecorder` on the already-consented microphone stream.
2. **Stop and transcribe** sends the clip only to `127.0.0.1` FastAPI.
3. FastAPI accepts WebM, Ogg, MP4, FLAC, or WAV; rejects bodies above 10 MiB; and keeps accepted bytes in an in-memory stream.
4. Faster-Whisper decodes the clip, rejects decoded audio above 60 seconds, and transcribes with CPU `int8` by default.
5. The transcript is appended to **Your answer** for review and editing. Only **Send answer** starts an interview turn.

The backend serializes model inference to avoid concurrent CPU and memory spikes. Errors such as unsupported media, silence, missing runtime, or model failure are returned as safe status messages; text input remains available.

## Evaluation

### Dialogue smoke evaluation

`eval/cases.json` contains five initial interview scenarios:

1. explicit hint request;
2. explicit request for thinking time;
3. project description without evaluation evidence;
4. macro-F1 with class-imbalance rationale but no class-level result;
5. explicit request to end the session.

The runner creates a fresh session for each case and compares `action`, `reason`, and `followup_topic`. Human-facing `reply_criteria` remain in the fixture for later structured-output evaluation.

A successful run prints:

```text
PASS request_hint
PASS request_thinking_time
PASS project_without_evaluation
PASS metric_with_rationale
PASS request_end_session
Summary: 5 passed, 0 failed
```

It writes `eval/results/latest.json` with this shape:

```json
{
  "schema_version": "eval-v1",
  "summary": {
    "total": 5,
    "passed": 5,
    "failed": 0
  },
  "cases": [
    {
      "id": "request_hint",
      "expected": {
        "action": "offer_hint",
        "reason": "explicit_hint_request",
        "followup_topic": "none"
      },
      "actual": {
        "action": "offer_hint",
        "reason": "explicit_hint_request",
        "followup_topic": "none"
      },
      "reply": "Start by naming the metric you used and why it matched the problem.",
      "passed": true
    }
  ]
}
```

The five cases are a smoke evaluation, not evidence of robustness, scientific validity, multilingual performance, or generalization.

### Answer-quality benchmark

`eval/answer_quality_cases.json` defines six calibration and six held-out cases. The contract labels observable answer properties (`input_type`, `relevance`, project evidence, evaluation-method evidence, and evaluation-result evidence) plus the expected policy action. It does not score intelligence, personality, employability, or speaking style.

The fixture covers missing input, irrelevant and vague answers, paraphrased evidence, a prompt-injection attempt, and explicit hint/thinking/stop controls. Case IDs are unique across splits. The held-out cases were fixed before this evaluator and were not changed while implementing it.

```powershell
python -m eval.answer_quality_evaluation --enforce-target
```

The explainable deterministic evaluator produces:

| Split | Exact evidence + action | Expected action |
|---|---:|---:|
| Calibration | 6/6 (100%) | 6/6 (100%) |
| Held-out | 6/6 (100%) | 6/6 (100%) |

The acceptance target is at least 80% held-out exact match and 100% held-out critical-action accuracy. CI now enforces that target. The runner writes `eval/results/answer-quality-latest.json`; reports contain case IDs, labels, decisions, and aggregate metrics but omit candidate text.

Twelve authored cases are only deterministic regression fixtures. The perfect result is not evidence of human validity, multilingual performance, broad semantic understanding, or generalization. A larger independently labeled dataset is required before making those claims.

### Audio turn evaluation

`eval/audio_cases.json` contains separate synthetic `calibration` and `held_out` timing traces sampled at 100 ms. Calibration selects the threshold with the fewest false-ready cases, then the lowest mean end-detection delay. Held-out traces are never used for threshold selection.

```powershell
node --test eval/audio_evaluation.test.js
node eval/run_audio_cases.js
```

| Threshold | Calibration false-ready rate | Calibration mean delay | Held-out false-ready rate | Held-out mean delay |
|---:|---:|---:|---:|---:|
| 800 ms | 66.67% | 800 ms | 50% | 1,000 ms |
| 1,200 ms | 33.33% | 1,200 ms | 50% | 1,400 ms |
| **1,600 ms** | **0%** | **1,600 ms** | **25%** | **1,800 ms** |
| 2,000 ms | 0% | 2,000 ms | 0% | 2,200 ms |

The deterministic selection rule chooses 1,600 ms. On held-out traces it detects every completed turn, handles unavailable/no-speech cases correctly, and exposes one known failure: a candidate who resumes after a 1,700 ms thinking pause is marked ready 100 ms too early. The acceptance gate allows at most a 25% false-ready rate and 2,200 ms mean delay, so the current pilot passes without hiding that failed case.

The runner writes `eval/results/audio-latest.json`. These traces are synthetic engineering fixtures, not evidence of human conversational validity or cross-speaker generalization. A consented multi-speaker pilot is required before making portfolio performance claims.

### Local STT smoke

The real adapter and HTTP endpoint were manually exercised on Faster-Whisper's public 11-second `jfk.flac` fixture with `tiny.en`, CPU `int8`, and Python 3.14. Both produced the expected JFK sentence.

| Smoke path | Audio duration | Reported processing time |
|---|---:|---:|
| Cold adapter, including initial model acquisition/loading | 11,000 ms | 13,751 ms |
| HTTP endpoint with cached model | 11,000 ms | 1,663 ms |

This single English sample is an integration smoke check, not a WER or latency benchmark; the default `base.en` model and other hardware will differ.

## Privacy and scientific boundaries

- Candidate text is processed in memory and is not stored in `SessionState` or default logs.
- Gemini receives candidate text only when an API key is configured, the generator is explicitly injected, and the selected action is eligible. Review Google's data handling terms before using real interview content.
- Logs contain reply source and fallback reason only, not candidate text, prompts, response bodies, API keys, audio, or exception messages.
- Voice-activity samples remain in browser memory and are never recorded by the timing path.
- Voice-answer recording requires a separate explicit click. The clip is sent only to local FastAPI, bounded by size and decoded duration, processed in memory, and not persisted.
- Microphone denial or transcription failure leaves the complete text workflow available.
- Missing audio is represented as `unavailable`, never as silence or a neutral psychological state.
- Text heuristics, audio energy, transcripts, and future facial blendshapes do **not** measure stress, focus, mood, honesty, personality, competence, or hiring suitability.
- Vision has not been implemented. It remains an optional experiment that must demonstrate measurable value before inclusion.

Gemini generates wording only. Local policy retains ownership of `action`, `reason`, `followup_topic`, and all state transitions; control actions remain local even when Gemini is configured.
