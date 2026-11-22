"""Explainable answer-quality evidence extracted from candidate text."""

from __future__ import annotations

import re
from collections.abc import Sequence

from backend.app.schemas import (
    AnswerRelevance,
    ContentEvidence,
    ExplicitSignals,
    InputType,
)


def _compile_terms(terms: Sequence[str]) -> re.Pattern[str]:
    alternatives = "|".join(
        rf"(?<!\w){re.escape(term)}(?!\w)" for term in terms
    )
    return re.compile(alternatives)


_PROJECT_ACTION = _compile_terms(
    (
        "built",
        "created",
        "designed",
        "developed",
        "implemented",
        "trained",
        "launched",
        "ran",
        "handled",
        "resolved",
        "managed",
        "membuat",
        "membangun",
        "merancang",
        "mengembangkan",
        "menerapkan",
        "melatih",
        "meluncurkan",
        "menjalankan",
        "menangani",
        "menyelesaikan",
        "mengelola",
    )
)
_PROJECT_ARTIFACT = _compile_terms(
    (
        "application",
        "app",
        "assistant",
        "classifier",
        "model",
        "pipeline",
        "system",
        "website",
        "campaign",
        "complaint",
        "prototype",
        "workflow",
        "aplikasi",
        "situs web",
        "kampanye",
        "keluhan",
        "prototipe",
        "alur kerja",
        "sistem",
    )
)
_PROJECT_CONTEXT = _compile_terms(("capstone", "portfolio", "project", "proyek"))
_PROJECT_OWNER = _compile_terms(("my", "our", "saya", "kami"))
_EVALUATION_METHOD = _compile_terms(
    (
        "accuracy",
        "auc",
        "cross-validation",
        "cross validation",
        "evaluated",
        "evaluation",
        "f1",
        "held-out",
        "held out",
        "measured",
        "precision",
        "recall",
        "test set",
        "unseen examples",
        "validation set",
        "tested",
        "testing",
        "unit test",
        "unit tests",
        "widget test",
        "widget tests",
        "integration test",
        "integration tests",
        "end-to-end test",
        "end-to-end tests",
        "usability test",
        "usability tests",
        "customer feedback",
        "customer checks",
        "customer survey",
        "customer interviews",
        "user feedback",
        "user testing",
        "a/b test",
        "a/b testing",
        "diuji",
        "menguji",
        "pengujian",
        "uji unit",
        "uji widget",
        "uji integrasi",
        "uji kegunaan",
        "uji pengguna",
        "umpan balik pelanggan",
        "umpan balik pengguna",
        "survei pelanggan",
        "wawancara pelanggan",
        "dievaluasi",
        "evaluasi",
        "diukur",
        "akurasi",
    )
)
_MACRO_F1 = _compile_terms(("macro-f1", "macro f1", "macro f-score"))
_CLASS_IMBALANCE = _compile_terms(
    ("class imbalance", "imbalanced", "minority class")
)
_CLASS_LEVEL = _compile_terms(
    ("class-level", "hardest class", "per-class", "per class")
)
_AI_TOPIC = _compile_terms(
    (
        "ai",
        "artificial intelligence",
        "capstone",
        "classifier",
        "machine learning",
        "model",
        "neural network",
        "project",
    )
)
# Quantities need a percentage, a score cue, or a timed improvement. Bare
# decimals/integers can be versions, ticket identifiers, or years, not outcomes.
_NUMBER = r"\d+(?:[.,]\d+)?"
_SCORE = r"(?:0[.,]\d+|1(?:[.,]0+)?)"
_TIME_UNIT = r"(?:ms|milliseconds?|s|seconds?|detik|milidetik)"
_TIME_METRIC = (
    r"(?:latency|(?:screen |page |app )?load(?:ing)? time|response time|latensi|"
    r"waktu (?:muat|pemuatan|respons|respon|tunggu))"
)
_TIME_CHANGE = (
    r"(?:reduced|decreased|improved|cut|lowered|dropped|fell|"
    r"menurunkan|mengurangi|berkurang|turun|dipangkas|dikurangi|diturunkan)"
)
_TIMED_OUTCOME = (
    rf"(?:from|dari)\s+{_NUMBER}\s*{_TIME_UNIT}\s+"
    rf"(?:to|menjadi|ke)\s+{_NUMBER}\s*{_TIME_UNIT}(?!\w)"
    rf"|(?:by|to|sebesar|menjadi)\s+{_NUMBER}\s*{_TIME_UNIT}(?!\w)"
)
_EVALUATION_RESULT = re.compile(
    rf"(?<![\w.,]){_NUMBER}\s*(?:%|percent|persen)(?!\w)"
    rf"|(?<!\w)(?:macro-f1|macro f1|macro f-score|f1|accuracy|auc|"
    rf"precision|recall|akurasi)\s*(?:(?:score|was|is|of|skor|sebesar|"
    rf"adalah)\s+|[=:]\s*){{0,2}}{_SCORE}(?!\w|[.,]\d)"
    rf"|(?<!\w)(?:achieved|obtained|reached|scored|mencapai|memperoleh)"
    rf"\s+(?:(?:a\s+)?(?:score|skor)(?:\s+(?:of|sebesar))?\s+)?"
    rf"{_SCORE}(?!\w|[.,]\d)"
    rf"|(?<!\w)(?:{_TIME_CHANGE}\s+{_TIME_METRIC}|"
    rf"{_TIME_METRIC}\s+{_TIME_CHANGE})\s+(?:{_TIMED_OUTCOME})"
)

# Bare acknowledgements. These carry no project evidence, so repeating the
# same substantive question at them is what makes the interviewer feel stuck.
_MINIMAL_ANSWERS = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "sure",
        "no",
        "nope",
        "maybe",
        "idk",
        "dunno",
        "not sure",
        "i don't know",
        "i dont know",
        "saya tidak tahu",
        "saya kurang tahu",
        "kurang tahu",
        "tidak",
        "iya",
        "ya",
        "bisa",
    }
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def _classify_input(text: str, signals: ExplicitSignals) -> InputType:
    if not text.strip():
        return InputType.MISSING
    if signals.hint_requested or signals.thinking_time_requested or signals.stop_requested:
        return InputType.CONTROL
    return InputType.ANSWER


def extract_answer_quality(
    text: str,
    signals: ExplicitSignals,
) -> ContentEvidence:
    """Extract lexical cues, not semantic competence or verified achievements."""

    input_type = _classify_input(text, signals)
    if input_type != InputType.ANSWER:
        return ContentEvidence(
            input_type=input_type,
            relevance=AnswerRelevance.NOT_APPLICABLE,
        )

    normalized = _normalize(text)
    has_artifact = _PROJECT_ARTIFACT.search(normalized) is not None
    project_described = has_artifact and any(
        pattern.search(normalized) is not None
        for pattern in (_PROJECT_ACTION, _PROJECT_CONTEXT, _PROJECT_OWNER)
    )
    evaluation_method_mentioned = (
        _EVALUATION_METHOD.search(normalized) is not None
    )
    evaluation_result_mentioned = (
        _EVALUATION_RESULT.search(normalized) is not None
    )
    macro_f1_mentioned = _MACRO_F1.search(normalized) is not None
    class_imbalance_mentioned = _CLASS_IMBALANCE.search(normalized) is not None
    class_level_performance_mentioned = _CLASS_LEVEL.search(normalized) is not None

    if project_described or evaluation_method_mentioned or evaluation_result_mentioned:
        relevance = AnswerRelevance.RELEVANT
    elif _AI_TOPIC.search(normalized) is not None:
        relevance = AnswerRelevance.UNCLEAR
    else:
        relevance = AnswerRelevance.IRRELEVANT

    stripped = normalized.strip(" .!?,")
    minimal_answer = stripped in _MINIMAL_ANSWERS

    return ContentEvidence(
        input_type=input_type,
        relevance=relevance,
        project_described=project_described,
        evaluation_method_mentioned=evaluation_method_mentioned,
        evaluation_result_mentioned=evaluation_result_mentioned,
        macro_f1_mentioned=macro_f1_mentioned,
        class_imbalance_mentioned=class_imbalance_mentioned,
        class_level_performance_mentioned=class_level_performance_mentioned,
        minimal_answer=minimal_answer,
    )
