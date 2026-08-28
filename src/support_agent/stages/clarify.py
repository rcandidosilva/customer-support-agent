"""Stage 4a - the middle rung: ask, do not guess, do not escalate.

Reached only when the reviewer said the missing piece is something the customer can
actually supply.  The constraint that matters here is *one* question.  A support bot that
interrogates is worse than one that hands off, because it spends the customer's patience
and still ends up handing off.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, Usage
from ..models import ClarifyingQuestion, CritiqueReport, DraftAnswer, Ticket

SYSTEM = """\
You are a tier-1 support agent at Northwind Analytics. You cannot answer this ticket \
confidently yet, and one specific fact from the customer would fix that.

Write a short reply that asks for it. Rules:

- Ask for exactly one thing. If two facts would help, ask for the one that unblocks the \
most, or ask for both in a single sentence if they are genuinely the same lookup \
("the invoice number and date from the charge").
- Give the customer something first. If part of the answer is already certain, say that \
part before asking. A bare question reads like a stall.
- Tell them where to find what you are asking for, if it is not obvious.
- Three to five sentences. No apology paragraph, no "I'd be happy to help".
- Never say you are checking with a colleague, escalating, or investigating. You are not.
- Do not mention confidence, internal tooling, or that you are automated.

`expected_information` records what this question is meant to establish, for the audit \
trail and for whoever picks the ticket up if the answer does not help.\
"""


def ask_clarifying_question(
    llm: LLMClient,
    ticket: Ticket,
    draft: DraftAnswer,
    report: CritiqueReport,
    settings: Settings,
) -> tuple[ClarifyingQuestion, Usage]:
    cfg = settings.stage("clarify")
    user = (
        f"{ticket.as_prompt_block()}\n\n"
        "======================================================================\n\n"
        f"What we were about to say (do NOT send this as-is):\n{draft.reply}\n\n"
        f"Why it is not safe to send: {report.reasoning}\n"
        f"Problems found: {report.problems or 'none listed'}\n"
        f"What we are missing: {draft.information_gaps or report.problems}\n"
        f"The drafter's suggested question: "
        f"{draft.best_clarifying_question or '(none offered)'}\n\n"
        "Write the clarifying reply."
    )
    return llm.structured(
        stage="clarify",
        system=SYSTEM,
        user=user,
        schema=ClarifyingQuestion,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
    )
