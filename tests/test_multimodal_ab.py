"""Regression tests for the three-condition multimodal A/B evaluation."""

from __future__ import annotations

from eval.run_multimodal_ab import CONDITIONS, _SCENARIOS, build_report


class TestReportShape:
    def test_every_scenario_runs_in_every_condition(self) -> None:
        report = build_report()

        assert len(report.results) == len(_SCENARIOS) * len(CONDITIONS)
        assert {r.condition for r in report.results} == set(CONDITIONS)

    def test_summary_counts_match_action_changes(self) -> None:
        report = build_report()

        assert report.summary["scenarios"] == len(_SCENARIOS)
        assert report.summary["audio_changed_actions"] == sum(
            1 for c in report.action_changes if c["audio_changed"]
        )
        assert report.summary["visual_changed_actions"] == sum(
            1 for c in report.action_changes if c["visual_changed"]
        )


class TestModalitiesChangeActions:
    def _change(self, report, scenario_id: str) -> dict:
        return next(
            c for c in report.action_changes if c["id"] == scenario_id
        )

    def test_audio_waits_during_thinking_pause(self) -> None:
        change = self._change(build_report(), "thin_answer_mid_speech")

        assert change["text"] == "request_clarification"
        assert change["text_audio"] == "wait"
        assert change["audio_changed"] is True

    def test_audio_waits_while_speaking(self) -> None:
        change = self._change(build_report(), "vague_answer_still_speaking")

        assert change["text_audio"] == "wait"

    def test_audio_waits_when_nothing_spoken(self) -> None:
        change = self._change(build_report(), "no_speech_detected")

        assert change["text"] == "request_input"
        assert change["text_audio"] == "wait"

    def test_visual_gives_space_on_high_load_face(self) -> None:
        change = self._change(build_report(), "complete_answer_tense_face")

        assert change["text_audio"] == "continue_interview"
        assert change["text_audio_visual"] == "give_space"
        assert change["visual_changed"] is True

    def test_visual_gives_space_before_probing(self) -> None:
        change = self._change(build_report(), "project_only_tense")

        assert change["text_audio"] == "ask_followup"
        assert change["text_audio_visual"] == "give_space"


class TestModalitiesDoNotOverreach:
    def _change(self, report, scenario_id: str) -> dict:
        return next(
            c for c in report.action_changes if c["id"] == scenario_id
        )

    def test_low_confidence_face_changes_nothing(self) -> None:
        change = self._change(build_report(), "answer_low_confidence_face")

        assert change["text_audio"] == change["text_audio_visual"]
        assert change["visual_changed"] is False

    def test_calm_face_changes_nothing(self) -> None:
        change = self._change(build_report(), "complete_answer_calm")

        assert change["visual_changed"] is False

    def test_ready_turn_does_not_block_followup(self) -> None:
        change = self._change(build_report(), "thin_answer_turn_ready")

        assert change["text"] == change["text_audio"]

    def test_text_only_never_sees_sensors(self) -> None:
        report = build_report()
        text_only = [r for r in report.results if r.condition == "text"]

        assert all(
            r.action not in {"give_space"} for r in text_only
        )
        assert all(
            r.reason != "audio_turn_incomplete" for r in text_only
        )
