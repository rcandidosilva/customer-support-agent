"""Stage 2 - the tier-1 agent's attempt at an answer.

The drafter writes as if the reply is going out, because it might.  What makes the draft
useful to the rest of the ladder is the three fields it must fill in alongside the prose:
``unsupported_claims``, ``information_gaps``, and ``best_clarifying_question``.  Asking a
drafter to enumerate what it could not support is far cheaper than asking a reviewer to
find it, and it gives the reviewer in stage 3 a starting point to verify rather than a
blank page.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, Usage
from ..models import Classification, DraftAnswer, RetrievedChunk, Ticket

SYSTEM = """\
You are a tier-1 support agent at Northwind Analytics writing a reply to a customer.

You may use only the knowledge-base passages provided. They are the whole of what you \
know about the product. If the passages do not settle the question, say so in \
`information_gaps` and write the best partial reply you honestly can - do not fill the \
hole with plausible product behaviour. Inventing a menu path or a policy detail is the \
single worst thing you can do here, because a confident wrong answer costs more than no \
answer.

The reply:
- Answers the actual question first, in the first two sentences. No throat-clearing.
- Uses the customer's plan and situation where the passages make it relevant.
- Gives concrete steps with the real UI paths from the passages.
- Is plain and warm without being saccharine. No "I completely understand how \
frustrating that must be."
- Never promises a refund, credit, exception, timeline, or escalation. You do not have \
the authority to commit to any of those.
- Does not mention the knowledge base, article IDs, or that you are automated.

Then fill in the audit fields honestly. You are not being graded on confidence:

- `citations`: article IDs (like `kb-011`) whose content you actually used. Passages \
sometimes cross-reference other articles by ID; those are pointers, not evidence. Cite \
only the IDs in the headers of the passages you were actually given. If a passage tells \
you something and names another article as the fuller source, the citation is the \
passage you read, and the other article belongs in `information_gaps`.
- `unsupported_claims`: every sentence in your reply asserting something the passages do \
not establish. If your reply is fully grounded, this is empty - but check it line by line \
before saying so.
- `information_gaps`: what you would need to know to be certain. Be specific: "whether \
the charge was the annual renewal or a mid-cycle seat addition", not "more details".
- `best_clarifying_question`: the single question that would most reduce your \
uncertainty, phrased as you would ask the customer. Empty string if no question the \
customer could answer would help - if the gap is internal data, that is not a \
clarifying question, that is an escalation.
- `kb_covers_this`: whether the knowledge base genuinely addresses this topic at all.\
"""


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no knowledge-base passages matched this ticket)"
    return "\n\n---\n\n".join(c.as_prompt_block() for c in chunks)


def draft_answer(
    llm: LLMClient,
    ticket: Ticket,
    classification: Classification,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[DraftAnswer, Usage]:
    cfg = settings.stage("draft")
    user = (
        "Knowledge-base passages:\n\n"
        f"{_format_passages(chunks)}\n\n"
        "======================================================================\n\n"
        f"Triage says: category={classification.category} "
        f"severity={classification.severity} sentiment={classification.sentiment}\n"
        f"Intent: {classification.intent}\n\n"
        f"{ticket.as_prompt_block()}\n\n"
        "Draft the reply and fill in the audit fields."
    )
    return llm.structured(
        stage="draft",
        system=SYSTEM,
        user=user,
        schema=DraftAnswer,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
    )
