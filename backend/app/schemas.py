"""Version-one domain contracts for the text interview baseline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    """Strict base model shared by domain contracts."""

    model_config = ConfigDict(extra="forbid")


class DialogueAction(str, Enum):
    OFFER_HINT = "offer_hint"
    WAIT = "wait"
    ASK_FOLLOWUP = "ask_followup"
    END_SESSION = "end_session"
    CONTINUE_INTERVIEW = "continue_interview"
    REQUEST_INPUT = "request_input"
    REQUEST_CLARIFICATION = "request_clarification"



class EmotionLabel(str, Enum):
    """Proxy emotion labels derived from FACS AU combinations.

    Mapping follows Ekman & Friesen (1978) documented AU-emotion
    combinations. See README references for citation and limitations.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"
    DISGUSTED = "disgusted"
    FEARFUL = "fearful"
    UNCERTAIN = "uncertain"

class DecisionReason(str, Enum):
    SESSION_ALREADY_ENDED = "session_already_ended"
    EXPLICIT_STOP_REQUEST = "explicit_stop_request"
    EXPLICIT_THINKING_TIME_REQUEST = "explicit_thinking_time_request"
    EXPLICIT_HINT_REQUEST = "explicit_hint_request"
    INPUT_MISSING = "input_missing"
    ANSWER_IRRELEVANT = "answer_irrelevant"
    ANSWER_UNCLEAR = "answer_unclear"
    PROJECT_EVIDENCE_MISSING = "project_evidence_missing"
    CLASS_LEVEL_PERFORMANCE_MISSING = "class_level_performance_missing"
    EVALUATION_EVIDENCE_MISSING = "evaluation_evidence_missing"
    ANSWER_COMPLETE = "answer_complete"


class FollowupTopic(str, Enum):
    NONE = "none"
    EVALUATION_METHOD = "evaluation_method"
    CLASS_LEVEL_PERFORMANCE = "class_level_performance"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"


class ReplySource(str, Enum):
    LOCAL = "local"
    GROQ = "groq"
    LOCAL_FALLBACK = "local_fallback"


class ReplyFallbackReason(str, Enum):
    TIMEOUT = "timeout"
    SDK_ERROR = "sdk_error"
    MALFORMED_OUTPUT = "malformed_output"
    INVALID_REPLY = "invalid_reply"


class ExplicitSignals(DomainModel):
    hint_requested: bool = False
    thinking_time_requested: bool = False
    stop_requested: bool = False


class InputType(str, Enum):
    ANSWER = "answer"
    CONTROL = "control"
    MISSING = "missing"


class AnswerRelevance(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "not_applicable"


class ContentEvidence(DomainModel):
    input_type: InputType
    relevance: AnswerRelevance
    project_described: bool = False
    evaluation_method_mentioned: bool = False
    evaluation_result_mentioned: bool = False
    macro_f1_mentioned: bool = False
    class_imbalance_mentioned: bool = False
    class_level_performance_mentioned: bool = False

    @model_validator(mode="after")
    def validate_relevance(self) -> "ContentEvidence":
        answer_has_label = self.relevance != AnswerRelevance.NOT_APPLICABLE
        if (self.input_type == InputType.ANSWER) != answer_has_label:
            raise ValueError(
                "only answer input may have an applicable relevance label"
            )
        return self


class BlendshapeReading(DomainModel):
    """A single MediaPipe blendshape value."""

    name: str
    score: float = Field(ge=0.0, le=1.0)


class VisionObservation(DomainModel):
    """Raw perceptual input from MediaPipe Face Landmarker.

    face_available=false means no face was detected; downstream
    consumers must treat emotional_state as unavailable, not neutral.
    """

    face_available: bool
    face_count: int = Field(default=0, ge=0)
    inference_ms: float | None = None
    observed_at_ms: int | None = None
    blendshapes: list[BlendshapeReading] = Field(default_factory=list)


class EmotionalState(DomainModel):
    """Proxy emotional state derived from FACS AU mapping.

    All scores are in [0.0, 1.0] and represent the strength of
    AU-combination evidence, NOT a direct measurement of internal
    mental state. See README for scientific limitations.
    """

    dominant_emotion: EmotionLabel = EmotionLabel.NEUTRAL
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    emotion_scores: dict[str, float] = Field(default_factory=dict)
    activated_aus: list[str] = Field(default_factory=list)
    face_available: bool = False


class EmotionalStateHistory(DomainModel):
    """Moving-average emotional state across recent observations."""

    current: EmotionalState
    window_size: int = Field(default=5, ge=1, le=20)
    samples: int = Field(default=0, ge=0)


class SessionState(DomainModel):
    session_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    turn_index: int = Field(default=0, ge=0)
    hints_offered: int = Field(default=0, ge=0)
    followups_asked: int = Field(default=0, ge=0)
    last_action: DialogueAction | None = None
    emotional_state: EmotionalState | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value


class ActionDecision(DomainModel):
    action: DialogueAction
    reason: DecisionReason
    followup_topic: FollowupTopic = FollowupTopic.NONE

    @model_validator(mode="after")
    def validate_followup_topic(self) -> "ActionDecision":
        asks_followup = self.action == DialogueAction.ASK_FOLLOWUP
        has_topic = self.followup_topic != FollowupTopic.NONE
        if asks_followup and not has_topic:
            raise ValueError("ask_followup requires a followup topic")
        if not asks_followup and has_topic:
            raise ValueError("only ask_followup may have a followup topic")
        return self


class TurnResult(DomainModel):
    signals: ExplicitSignals
    content_evidence: ContentEvidence
    decision: ActionDecision
    updated_state: SessionState
    reply: str = Field(min_length=1)
    reply_source: ReplySource = ReplySource.LOCAL
    fallback_reason: ReplyFallbackReason | None = None

    @model_validator(mode="after")
    def validate_fallback_metadata(self) -> "TurnResult":
        used_fallback = self.reply_source == ReplySource.LOCAL_FALLBACK
        has_reason = self.fallback_reason is not None
        if used_fallback and not has_reason:
            raise ValueError("local_fallback requires a fallback reason")
        if not used_fallback and has_reason:
            raise ValueError("fallback reason requires local_fallback source")
        return self
