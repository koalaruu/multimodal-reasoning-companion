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
    ("built", "created", "designed", "developed", "implemented", "trained")
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
    )
)
_PROJECT_CONTEXT = _compile_terms(("capstone", "portfolio", "project"))
_PROJECT_OWNER = _compile_terms(("my", "our"))
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
_EVALUATION_RESULT = re.compile(
    r"(?<!\w)\d+(?:\.\d+)?%(?!\w)"
    r"|(?<!\w)0\.\d+(?!\w)"
    r"|(?<!\w)(?:achieved|obtained|reached|scored)\s+"
    r"(?:[a-z-]+\s+){0,2}\d+(?:\.\d+)?%?(?!\w)"
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
    """Extract observable content evidence without judging the candidate."""

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

    return ContentEvidence(
        input_type=input_type,
        relevance=relevance,
        project_described=project_described,
        evaluation_method_mentioned=evaluation_method_mentioned,
        evaluation_result_mentioned=evaluation_result_mentioned,
        macro_f1_mentioned=macro_f1_mentioned,
        class_imbalance_mentioned=class_imbalance_mentioned,
        class_level_performance_mentioned=class_level_performance_mentioned,
    )
