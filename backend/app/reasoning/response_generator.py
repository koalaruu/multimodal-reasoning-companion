"""Local response text for the offline baseline."""

from __future__ import annotations

from backend.app.schemas import ActionDecision, DialogueAction, FollowupTopic

_REPLIES = {
    (DialogueAction.OFFER_HINT, FollowupTopic.NONE): (
        "Start by naming the metric you used and why it matched the problem."
    ),
    (DialogueAction.WAIT, FollowupTopic.NONE): "Of course—take your time.",
    (DialogueAction.ASK_FOLLOWUP, FollowupTopic.EVALUATION_METHOD): (
        "How did you evaluate the classifier?"
    ),
    (DialogueAction.ASK_FOLLOWUP, FollowupTopic.CLASS_LEVEL_PERFORMANCE): (
        "Which class was hardest to predict, and what did its results show?"
    ),
    (DialogueAction.END_SESSION, FollowupTopic.NONE): (
        "Understood. We'll end the practice here."
    ),
    (DialogueAction.REQUEST_INPUT, FollowupTopic.NONE): (
        "Please share your answer when you're ready."
    ),
    (DialogueAction.CONTINUE_INTERVIEW, FollowupTopic.NONE): (
        "Thank you. That covers the project and its evaluation."
    ),
}


def generate_local_reply(decision: ActionDecision) -> str:
    """Generate wording without changing the policy decision."""

    return _REPLIES[(decision.action, decision.followup_topic)]
