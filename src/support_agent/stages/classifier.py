"""Stage 1 - triage.

Runs before retrieval because two of its outputs steer everything downstream:
``search_terms`` becomes the retrieval query (customers do not write in help-centre
vocabulary), and ``requires_account_data`` / ``requests_exception`` are read by the
policy layer as hard escalation triggers before any confidence score exists.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, Usage
from ..models import Classification, Ticket

SYSTEM = """\
You are the triage stage of an automated support pipeline for Northwind Analytics, a \
B2B data-analytics SaaS. You do not answer tickets. You label them, accurately and \
without hedging, so later stages can route them.

Two fields decide whether a human ever sees this ticket, so be exacting about them:

- `requires_account_data` is true when a correct answer depends on facts about *this \
specific account* that a help-centre article cannot contain: what a particular invoice \
charged, whether a named sync actually ran, what a token's scopes are, why this \
workspace was suspended. It is false when the question is about how the product works in \
general, even if the customer phrased it about themselves. "Why was I charged $480?" is \
true. "How does proration work?" is false.

- `requests_exception` is true when the customer wants something documented policy does \
not give them: a refund outside the window, a rate limit above their plan, a backdated \
invoice, an MFA reset without verification. It is false when they want something they \
are simply entitled to.

`search_terms`: 4-8 terms in the vocabulary of a help-centre article, not the \
customer's. A customer writing "my charts are all showing yesterday's numbers" should \
produce terms like "sync latency", "freshness", "backfill", "sync interval".

`risk_signals`: short phrases naming concrete exposure - "threatens to cancel", \
"mentions legal counsel", "reports possible unauthorized access", "references public \
review site". Empty list when there is none. Do not invent risk from ordinary irritation.

`pii_present`: name the *kinds* of sensitive data pasted into the ticket ("full card \
number", "API token", "customer email addresses"), never the values themselves.

`severity` reflects business impact and urgency together, not tone. A polite report of a \
total outage is `urgent`. A furious complaint about a UI preference is `low`.\
"""


def classify(
    llm: LLMClient, ticket: Ticket, settings: Settings
) -> tuple[Classification, Usage]:
    cfg = settings.stage("classify")
    return llm.structured(
        stage="classify",
        system=SYSTEM,
        user=f"Classify this ticket.\n\n{ticket.as_prompt_block()}",
        schema=Classification,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
    )
