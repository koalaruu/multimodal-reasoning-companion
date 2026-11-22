"""Behavioral regressions for role-neutral lexical answer evidence."""

from __future__ import annotations

import pytest

from backend.app.perception.answer_quality import extract_answer_quality
from backend.app.schemas import AnswerRelevance, ExplicitSignals, InputType


@pytest.mark.parametrize(
    "text",
    [
        (
            "I built a Flutter application for delivery tracking. I checked it "
            "with widget and integration tests and reduced latency from "
            "800 ms to 300 ms."
        ),
        (
            "Saya membangun aplikasi Flutter untuk pelacakan pengiriman. "
            "Saya menguji dengan uji widget dan uji integrasi, lalu "
            "menurunkan latensi dari 800 ms menjadi 300 ms."
        ),
        (
            "I developed an application and used unit tests. Load time "
            "dropped from 3 seconds to 1 second."
        ),
        (
            "Saya membuat aplikasi dan melakukan pengujian. Waktu muat "
            "berkurang dari 3 detik menjadi 1 detik."
        ),
        (
            "I managed a sales campaign, checked customer feedback, and "
            "increased conversions by 15%."
        ),
        (
            "Saya mengelola kampanye penjualan, melakukan survei pelanggan, "
            "dan meningkatkan konversi sebesar 15 persen."
        ),
        (
            "I resolved a customer complaint, used customer checks, and "
            "reduced response time by 30 seconds."
        ),
        (
            "Saya menangani keluhan pelanggan dan melakukan wawancara "
            "pelanggan. Waktu respons turun sebesar 30 detik."
        ),
        (
            "I designed a UX prototype and used usability tests. "
            "Task completion improved by 22 percent."
        ),
        (
            "Saya merancang prototipe UX dengan uji kegunaan dan umpan "
            "balik pengguna. Penyelesaian tugas meningkat 22,5%."
        ),
    ],
)
def test_non_ml_work_has_universal_evidence(text: str) -> None:
    evidence = extract_answer_quality(text, ExplicitSignals())

    assert evidence.input_type == InputType.ANSWER
    assert evidence.relevance == AnswerRelevance.RELEVANT
    assert evidence.project_described is True
    assert evidence.evaluation_method_mentioned is True
    assert evidence.evaluation_result_mentioned is True
    assert evidence.macro_f1_mentioned is False
    assert evidence.class_imbalance_mentioned is False
    assert evidence.class_level_performance_mentioned is False


@pytest.mark.parametrize(
    "text",
    [
        (
            "I built a support-ticket classifier. I evaluated it on a "
            "held-out test set using macro-F1 and obtained 0.81."
        ),
        (
            "For my capstone, I trained a ticket-routing model. Performance "
            "was measured on unseen examples; it reached 84% correct predictions."
        ),
        "I developed a classifier; accuracy was 0.92.",
        "Saya melatih model klasifikasi; akurasi sebesar 0,92.",
    ],
)
def test_ml_work_retains_universal_evidence(text: str) -> None:
    evidence = extract_answer_quality(text, ExplicitSignals())

    assert evidence.project_described is True
    assert evidence.evaluation_method_mentioned is True
    assert evidence.evaluation_result_mentioned is True


def test_ml_detail_flags_remain_observable_not_required() -> None:
    evidence = extract_answer_quality(
        "I built a classifier and evaluated macro-F1 because of class "
        "imbalance. Per-class recall was 0.75.",
        ExplicitSignals(),
    )

    assert evidence.macro_f1_mentioned is True
    assert evidence.class_imbalance_mentioned is True
    assert evidence.class_level_performance_mentioned is True
    assert evidence.evaluation_result_mentioned is True


@pytest.mark.parametrize(
    "text",
    [
        "Flutter, Dart, Figma, Salesforce, Zendesk.",
        "I used Flutter and Figma.",
        "Saya menggunakan Flutter dan Figma.",
        "I built with Flutter.",
        "An application, a campaign, and a prototype.",
    ],
)
def test_tools_and_unanchored_artifacts_are_not_work_examples(text: str) -> None:
    evidence = extract_answer_quality(text, ExplicitSignals())

    assert evidence.project_described is False
    assert evidence.evaluation_method_mentioned is False
    assert evidence.evaluation_result_mentioned is False


@pytest.mark.parametrize(
    "detail",
    [
        "The project started in 2024.",
        "Ticket 800 was linked to complaint 300.",
        "The application identifier is app_0.81.",
        "The application version is 0.81.",
        "The application reached version 0.81.",
        "We reached milestone 2024.",
        "Accuracy uses dataset 2024.",
        "The application scored ticket 300.",
        "Latency changed from 800 to 300.",
        "The load time requirement is 300 ms.",
        "Accuracy was 0.81.2 in the release name.",
    ],
)
def test_identifiers_years_and_requirements_are_not_outcomes(detail: str) -> None:
    evidence = extract_answer_quality(
        f"I built an application and used integration tests. {detail}",
        ExplicitSignals(),
    )

    assert evidence.project_described is True
    assert evidence.evaluation_method_mentioned is True
    assert evidence.evaluation_result_mentioned is False


def test_work_and_checks_do_not_invent_an_outcome() -> None:
    evidence = extract_answer_quality(
        "Saya membangun aplikasi Flutter dengan uji widget dan uji integrasi.",
        ExplicitSignals(),
    )

    assert evidence.project_described is True
    assert evidence.evaluation_method_mentioned is True
    assert evidence.evaluation_result_mentioned is False


def test_evaluation_alone_does_not_invent_a_work_example() -> None:
    evidence = extract_answer_quality(
        "I used integration tests and reduced latency from 800 ms to 300 ms.",
        ExplicitSignals(),
    )

    assert evidence.project_described is False
    assert evidence.evaluation_method_mentioned is True
    assert evidence.evaluation_result_mentioned is True


@pytest.mark.parametrize(
    ("text", "signals", "input_type"),
    [
        ("   ", ExplicitSignals(), InputType.MISSING),
        (
            "I built an application, tested it, and reduced latency to 300 ms.",
            ExplicitSignals(hint_requested=True),
            InputType.CONTROL,
        ),
    ],
)
def test_non_answer_inputs_do_not_produce_content_evidence(
    text: str, signals: ExplicitSignals, input_type: InputType
) -> None:
    evidence = extract_answer_quality(text, signals)

    assert evidence.input_type == input_type
    assert evidence.relevance == AnswerRelevance.NOT_APPLICABLE
    assert evidence.project_described is False
    assert evidence.evaluation_method_mentioned is False
    assert evidence.evaluation_result_mentioned is False
