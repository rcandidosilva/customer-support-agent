"""Stage 4b - the escalation packet.

This is the artifact the whole pipeline exists to produce well.  A deflection that fails
is not a loss if the human who inherits it starts sixty seconds ahead of where they would
have started with a raw ticket.  A deflection that fails *and* hands over a wall of
apologetic prose is worse than no automation at all.

The hard part is audience.  Every other stage in this pipeline writes to a customer, and
the model has enormous priors about what support text sounds like.  Those priors are
exactly wrong here.  The system prompt below spends most of its length fighting them, and
:func:`support_agent.handoff_lint.lint_packet` checks the result mechanically.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, Usage
from ..models import (
    Classification,
    ConfidenceReport,
    DraftAnswer,
    HandoffPacket,
    RetrievedChunk,
    Ticket,
)

SYSTEM = """\
You are writing an internal handoff brief for the Northwind Analytics support queue.

Your reader is a support engineer who has eleven other tickets open and will give this \
brief about forty seconds before deciding what to do. They have not read the ticket and \
should not have to. Write for them.

This is the part people get wrong, so be deliberate about it: **you are not writing to \
the customer.** Not one sentence of this brief is customer-facing except the field \
explicitly marked as a draft reply. That means:

- No greetings, no sign-offs, no "Thanks for reaching out", no "I hope this helps".
- No empathy performance. "The customer is frustrated" is useful; "we're so sorry for \
the inconvenience" is noise addressed to nobody in the room.
- No hedging as politeness. Write "the refund is outside the 14-day window" rather than \
"it may possibly be the case that the refund might fall outside". If you are unsure, say \
you are unsure and say what would settle it - that is different from hedging.
- Refer to the customer in the third person throughout.
- Assume product knowledge. Do not explain what a connector is.

Field by field:

`subject_line` - scannable in a queue list, under about 70 characters, shaped as \
`[Category/Priority] specific thing - what is needed`. Example: \
`[Billing/High] Duplicate annual charge - refund authorisation`. The point is that a lead \
skimming twenty of these can pick the right one without opening it.

`summary` - one sentence. What this is and why it is here.

`customer_goal` - what they are actually trying to achieve, which is often not what they \
asked for. Someone asking how to export everything may be trying to leave.

`already_attempted` - what the automated tier did: what was searched, what was checked, \
what was ruled out. This exists so the human does not repeat work. Include the dead ends; \
"confirmed no backfill is running on the connector" saves them a lookup.

`already_told_customer` - verbatim, or "Nothing has been sent to the customer." This is \
the highest-consequence field in the brief. A colleague who contradicts something we \
already said turns a support ticket into a complaint. If a clarifying question went out, \
quote it exactly.

`findings` - the evidence, each line ending with the article ID it came from, like \
`Annual plans are refundable in full within 14 days of the charge (kb-011).` Only \
findings that bear on the decision. Three to six lines.

`open_questions` - what could not be determined, each paired with why: `Whether the \
second charge was a renewal or a seat addition - the agent cannot see invoice line \
items.` Never a bare question.

`blocker` - the single reason this needs a person.

`next_steps` - ordered, concrete, each one a thing the reader could start doing right \
now. `Pull invoices INV-4471 and INV-4472 in Stripe and compare line items` is a step. \
`Investigate the billing issue` is not. Name the system to look in. If approval is \
needed, say whose. Two to five steps.

`suggested_reply_draft` - a reply the human could edit and send, written in customer \
voice. This is the *only* customer-facing field. If sending anything before the human \
has checked the facts would be a mistake, put an empty string here rather than a draft \
they might send on autopilot.

`risk_flags` - concrete exposure only: `Threatened to cancel a $4.8k/mo contract`, \
`Mentions their legal team`, `Third escalation in 90 days`. Not `customer seems unhappy`.

`priority` - the priority *for the human queue*, which is not always the ticket's \
severity. A low-severity question from an account that has escalated three times this \
quarter is not low priority.

`why_escalated` - one plain sentence, naming the actual limitation. \
`The answer depends on invoice line items the agent cannot read.` Not \
`confidence was below threshold` - that is a number, not a reason.

`kb_gap` - if the knowledge base should have covered this and did not, name the article \
that should exist, in one line. Empty string if the KB was not the problem. Most tickets \
are not KB gaps; do not manufacture one.\
"""


def build_handoff_packet(
    llm: LLMClient,
    ticket: Ticket,
    classification: Classification,
    draft: DraftAnswer | None,
    confidence: ConfidenceReport | None,
    chunks: list[RetrievedChunk],
    settings: Settings,
    *,
    escalation_reason: str,
    already_told_customer: str = "",
    degraded_note: str = "",
    lint_feedback: list[str] | None = None,
) -> tuple[HandoffPacket, Usage]:
    cfg = settings.stage("handoff")

    passages = (
        "\n\n---\n\n".join(c.as_prompt_block() for c in chunks)
        if chunks
        else "(retrieval returned nothing relevant)"
    )

    if draft is not None:
        draft_block = (
            f"Draft the automated tier produced but did NOT send:\n{draft.reply}\n\n"
            f"Citations claimed: {draft.citations or 'none'}\n"
            f"Gaps it self-reported: {draft.information_gaps or 'none'}\n"
            f"Claims it flagged as unsupported: {draft.unsupported_claims or 'none'}"
        )
    else:
        draft_block = (
            "The automated tier produced no draft - it failed before it got that far."
        )

    if confidence is not None:
        c = confidence.critique
        conf_block = (
            f"Confidence: {confidence.score:.2f} "
            f"(groundedness {c.groundedness:.2f}, coverage {c.coverage:.2f}, "
            f"action safety {c.action_safety:.2f})\n"
            f"Reviewer's blocker: {c.blocker}\n"
            f"Reviewer's problems: {c.problems or 'none'}\n"
            f"Reviewer's reasoning: {c.reasoning}\n"
            f"Automatic penalties applied: {confidence.penalties or 'none'}"
        )
    else:
        conf_block = "No confidence assessment is available for this ticket."

    told = already_told_customer.strip() or "Nothing has been sent to the customer."

    user = (
        f"{ticket.as_prompt_block()}\n\n"
        "======================================================================\n"
        f"Triage: category={classification.category} severity={classification.severity} "
        f"sentiment={classification.sentiment}\n"
        f"Intent: {classification.intent}\n"
        f"Requires account data: {classification.requires_account_data}\n"
        f"Requests a policy exception: {classification.requests_exception}\n"
        f"Risk signals: {classification.risk_signals or 'none'}\n"
        f"Sensitive data in ticket: {classification.pii_present or 'none'}\n\n"
        "======================================================================\n"
        f"Knowledge-base passages retrieved:\n\n{passages}\n\n"
        "======================================================================\n"
        f"{draft_block}\n\n"
        "======================================================================\n"
        f"{conf_block}\n\n"
        "======================================================================\n"
        f"Why this is being escalated: {escalation_reason}\n"
        f"{('Pipeline degradation: ' + degraded_note) if degraded_note else ''}\n\n"
        f"Already sent to the customer on this ticket:\n{told}\n\n"
        "Write the handoff brief."
    )

    if lint_feedback:
        # A rewrite.  Naming the specific defects beats repeating the style rules: the
        # model already had those and still produced the thing being rejected.
        user += (
            "\n\n======================================================================\n"
            "You already wrote this brief once and it was rejected. Fix exactly these "
            "problems and change nothing else:\n"
            + "\n".join(f"- {f}" for f in lint_feedback)
        )

    return llm.structured(
        stage="handoff",
        system=SYSTEM,
        user=user,
        schema=HandoffPacket,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
    )
