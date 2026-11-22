"""Local response text for the offline baseline."""

from __future__ import annotations

from backend.app.schemas import (
    ActionDecision,
    DecisionReason,
    DialogueAction,
    FollowupTopic,
)

_REPLIES = {
    (DialogueAction.OFFER_HINT, FollowupTopic.NONE): (
        "Start with one thing your work needed to achieve and how you checked it."
    ),
    (DialogueAction.WAIT, FollowupTopic.NONE): "Of course—take your time.",
    (DialogueAction.ASK_FOLLOWUP, FollowupTopic.EVALUATION_METHOD): (
        "How did you check whether your approach worked?"
    ),
    (DialogueAction.ASK_FOLLOWUP, FollowupTopic.CLASS_LEVEL_PERFORMANCE): (
        "What concrete outcome did your work achieve?"
    ),
    (DialogueAction.END_SESSION, FollowupTopic.NONE): (
        "Understood. We'll end the practice here."
    ),
    (DialogueAction.REQUEST_INPUT, FollowupTopic.NONE): (
        "Please share your answer when you're ready."
    ),
    (DialogueAction.REQUEST_CLARIFICATION, FollowupTopic.NONE): (
        "Please describe a concrete work or project example relevant to your "
        "target role and how you checked your approach."
    ),
    (DialogueAction.CONTINUE_INTERVIEW, FollowupTopic.NONE): (
        "Thank you. That covers your example, how you checked it, and its outcome."
    ),
    (DialogueAction.GIVE_SPACE, FollowupTopic.NONE): (
        "No rush. Take the time you need, and continue when you're ready."
    ),
    (DialogueAction.ANSWER_META_QUESTION, FollowupTopic.NONE): (
        "This is a structured interview drill, not an open chat. We explore "
        "work or project examples relevant to your target role, how you checked "
        "your approach, and the outcomes. Answer in your own "
        "words, ask for a hint with \"could I have a hint\", or stop with "
        "\"I'd like to stop the practice\"."
    ),
}


def generate_local_reply(decision: ActionDecision) -> str:
    """Generate wording without changing the policy decision."""

    if decision.reason == DecisionReason.CLARIFICATION_LIMIT_REACHED:
        return (
            "Let's take a different angle. Pick one work or project example, "
            "name one thing it had to achieve, and say how you checked it did."
        )
    if decision.reason == DecisionReason.PROJECT_EXAMPLE_NEEDED:
        return (
            "Which concrete work or project example do those checks and "
            "outcomes relate to?"
        )
    if decision.reason == DecisionReason.MINIMAL_ANSWER_NEEDS_CHOICE:
        return (
            "That's a short answer, so let me make it concrete. Pick one: "
            "your goal, the approach you took, or how you checked your work."
        )
    if decision.reason == DecisionReason.TOPIC_ADVANCED_WITHOUT_DETAIL:
        return (
            "Let's move on and come back to that later. What would you change "
            "about how you checked your approach if you did the work again?"
        )
    return _REPLIES[(decision.action, decision.followup_topic)]
