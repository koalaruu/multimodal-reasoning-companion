# Technical Brief — Adaptive Practice Partner

**Multimodal turn-taking for AI interview practice**
KAIT International Lab Internship 2026 — Theme 3: Multimodal Reasoning AI Systems

---

## 1. Research question

> Compared with an assistant that reads only the transcript, do speech-activity
> and facial-muscle signals help a practice interviewer choose when to wait,
> offer a hint, or ask a follow-up?

The target is not "detect emotion". It is a narrower, testable claim: sensor
evidence should change the **turn-taking action**, and it should stay silent
when the evidence is weak.

## 2. Architecture

```text
Browser                          Backend (local)              Cloud
─────────                        ───────────────              ─────
MediaRecorder ─────────────────▶ Faster-Whisper small   (or Groq Whisper)
                                 
Web Audio VAD ── turn_readiness ─┐
                                 ├─▶ dialogue_policy.decide()
MediaPipe Face ── 52 blendshape ─┘        │
      │                                   ├─▶ deterministic action
      └─▶ FACS AU proxy ──▶ emotion       │
                                          └─▶ Groq renders wording
CV + JD ─────────────────────────────────────▶ Groq context
```

**What the model may decide:** wording, and what to ask next.
**What it may not decide:** ending the session, or overriding an explicit
request.

### Layer responsibilities

| Layer | Technology | Runs | Why there |
|---|---|---|---|
| Turn detection | Web Audio VAD | browser | no network, no audio leaves device |
| Speech-to-text | Faster-Whisper `small` (default) or Groq Whisper | local / cloud | privacy default, accuracy option |
| Face landmarks | MediaPipe Face Landmarker | browser | frames never uploaded |
| AU → emotion | FACS rules (Ekman & Friesen, 1978) | backend | auditable, no training data |
| Turn policy | deterministic Python | backend | replayable, defensible |
| Conversation | Groq `gpt-oss-20b` | cloud | wording + next-question choice |

## 3. Design decisions and their reasons

**Policy stays deterministic for turn-taking.** A model that both chooses the
action and writes the words cannot be audited. Keeping the action in Python
means every decision has a reason code (`visual_load_high`,
`audio_turn_incomplete`, …) that can be replayed from a log.

**Only one facial signal changes behaviour.** High-arousal AU combinations
(`angry`, `fearful`, `disgusted`) at confidence ≥ 0.35 trigger `give_space`,
capped at once per session. Neutral, happy, low-confidence, and no-face readings
change nothing. This is deliberately narrow: broad FER claims are not
defensible (Cabitza et al., 2022; Cross et al., 2023; Feldman Barrett et al.,
2019).

**The model never sees raw audio or video.** Only transcript text, the
blendshape-derived emotion, CV text, and job description reach Groq.

**Only the candidate can end the session.** Enforced in policy, not prompt.

## 4. Results

### 4.1 Multimodal A/B — does a modality change the action?

Ten fixed scenarios, identical text input, three conditions.

| Condition | Audio | Vision |
|---|---|---|
| `text` | withheld | withheld |
| `text_audio` | available | withheld |
| `text_audio_visual` | available | available |

| Scenario | text | text+audio | text+audio+visual |
|---|---|---|---|
| Complete answer, calm face | `continue_interview` | `continue_interview` | `continue_interview` |
| Complete answer, tense face | `continue_interview` | `continue_interview` | **`give_space`** |
| Thin answer, mid-speech pause | `request_clarification` | **`wait`** | `wait` |
| Thin answer, turn ready | `request_clarification` | `request_clarification` | `request_clarification` |
| Vague answer, still speaking | `request_clarification` | **`wait`** | `wait` |
| Project only, tense face | `ask_followup` | `ask_followup` | **`give_space`** |
| Project only, calm face | `ask_followup` | `ask_followup` | `ask_followup` |
| Nothing spoken | `request_input` | **`wait`** | `wait` |
| Pause but answer complete | `continue_interview` | **`wait`** | `wait` |
| Low-confidence face | `continue_interview` | `continue_interview` | `continue_interview` |

**Audio changed the action in 4 of 10 scenarios. Vision changed it in 2 of 10.
Both left 4 untouched.**

The unchanged rows matter as much as the changed ones: a calm or low-confidence
face must not manufacture a decision.

### 4.2 Audio turn-end threshold

| Threshold | Calibration false-ready | Held-out false-ready | Held-out mean delay |
|---|---|---|---|
| 800 ms | 0.67 | 0.50 | 1000 ms |
| 1200 ms | 0.33 | 0.50 | 1400 ms |
| **1600 ms** | **0.00** | **0.25** | **1800 ms** |
| 2000 ms | 0.00 | 0.00 | 2200 ms |

Selected 1600 ms: it is the smallest threshold with zero false-ready on the
calibration split, and it stays inside the acceptance budget on held-out
(false-ready ≤ 0.25, mean delay ≤ 2200 ms). **One held-out case fails** —
a 1700 ms pause is read as a finished turn.

### 4.3 Face pipeline

| Metric | Calibration | Held-out |
|---|---|---|
| Runtime success | 2/2 | 4/4 |
| Face-count accuracy | 1.00 | 1.00 |
| No-face false positives | 0 | 0 |
| Stability | 1.00 | 1.00 |
| p50 inference | 10.6 ms | — |
| p95 inference | 53.4 ms | — |

Detection is reliable; **emotion labelling is not validated here.** The
blendshape→AU→emotion mapping was checked against 7 synthetic idealized
profiles (κ = 1.0), which proves internal consistency only.

### 4.4 Speech-to-text

| Split | Cases | WER | Mean RTF |
|---|---|---|---|
| Calibration | 1 | 0.00 | 1.93 |
| Held-out | 2 | 0.00 | 0.12 |

Three clips, English. **No Indonesian benchmark exists yet** — this is the
largest unmeasured claim in the project.

Public Indonesian speech on Wikimedia Commons is one-word pronunciation
recordings, which cannot support a word-error-rate claim. Rather than report a
number that looks like a benchmark but measures nothing, the project ships a
builder for a local benchmark instead:

```powershell
# put mono WAVs + references.json + plan.json in eval/recordings/, then:
python -m eval.build_local_stt_dataset
python -m eval.run_stt_cases --cases eval/stt_cases_local.json
```

The builder refuses to write a dataset that cannot support a claim: it requires
at least two held-out cases from at least two distinct speakers, and it rejects
any speaker appearing in both calibration and held-out, because that leaks.
Recordings stay gitignored; only the generated dataset with verified SHA-256
hashes is committed.

## 5. What this proves and what it does not

**Proves:** the policy is genuinely multimodal. Sensor evidence changes the
selected action, the direction matches the design, and weak evidence changes
nothing.

**Does not prove:** that the chosen actions were correct for a real candidate.
That requires human raters scoring the same clips — two raters marking each
decision point as acceptable or not, then Cohen's κ against the system.

**Does not prove:** bilingual STT quality, or that FACS proxies track felt
emotion rather than posed expression.

## 6. Limitations, stated plainly

1. Scenario inputs are synthetic and written by the author. Real candidates
   pause, restart, and contradict themselves in ways these ten rows do not.
2. FER accuracy degrades sharply outside frontal, well-lit conditions
   (Cabitza et al., 2022; Cross et al., 2023).
3. Inferring emotion from facial movement is theoretically contested
   (Feldman Barrett et al., 2019). The project uses it as advisory tone input,
   never as a diagnosis.
4. MediaPipe blendshapes are not FACS action units; the mapping is a documented
   approximation, with AU23 synthesized from `mouthClose` + `lipPress`.
5. STT is unmeasured in Indonesian.
6. No human raters yet.

## 7. Reproducing

```powershell
# Backend tests
.venv/Scripts/python.exe -m pytest -q          # 162 passed

# Multimodal A/B
.venv/Scripts/python.exe -m eval.run_multimodal_ab

# Audio turn-end threshold
node eval/run_audio_cases.js

# Face pipeline (from frontend/, needs a headless Chrome)
cd frontend && npm run eval:vision

# Speech-to-text
.venv/Scripts/python.exe -m eval.run_stt_cases
```

Every run writes a JSON report to `eval/results/`.

## 8. Next work, in priority order

1. **Human raters** on the ten decision points — converts "policy is
   multimodal" into "policy is right N% of the time".
2. **Indonesian STT benchmark** — the builder exists; it needs real
   recordings from at least two speakers.
3. **Recorded sessions from volunteers** — replaces synthetic scenarios.
4. **Video demo + this brief** as the application package.

## References

- Ekman, P. & Friesen, W.V. (1978). *Facial Action Coding System*.
- Cabitza, F., Campagner, A. & Mattioli, M. (2022). The unbearable (technical)
  unreliability of automated facial emotion recognition. *Big Data & Society*.
  doi:10.1177/20539517221129549
- Cross, M.P., Acevedo, A.M. & Hunter, J.F. (2023). A Critique of Automated
  Approaches to Code Facial Expressions. *Affective Science*.
  doi:10.1007/s42761-023-00195-0
- Feldman Barrett, L. et al. (2019). Emotional expressions reconsidered.
  *Psychological Science in the Public Interest*. doi:10.1177/1529100619832930
- Attrah, S. (2025). Emotion estimation from video footage with LSTM.
  *Frontiers in Neurorobotics*. doi:10.3389/fnbot.2025.1678984
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
