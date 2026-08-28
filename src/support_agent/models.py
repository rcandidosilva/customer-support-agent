"""Typed contracts for every stage of the deflection ladder.

Two families of models live here:

* **LLM-facing schemas** (``Classification``, ``DraftAnswer``, ``CritiqueReport``,
  ``ClarifyingQuestion``, ``HandoffPacket``) are sent to the Messages API as structured
  output schemas.  They deliberately avoid optional fields: every field is required and
  "nothing to report" is expressed as an empty string or empty list.  Optionality in a
  structured-output schema invites the model to omit the field entirely, and the whole
  point of this pipeline is that a stage's output shape is never in question.

* **Pipeline models** (``Ticket``, ``RetrievedChunk``, ``ConfidenceReport``, ``Decision``,
  ``Resolution``, ``StageTrace``) are ours alone and may use defaults freely.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Shared vocabularies
# --------------------------------------------------------------------------------------

Category = Literal[
    "account",       # login, SSO, MFA, membership
    "billing",       # invoices, refunds, payment failures
    "integrations",  # connectors, syncs, freshness
    "api",           # tokens, endpoints, rate limits
    "security",      # retention, DSAR, compliance, incidents
    "other",
]

Severity = Literal["low", "normal", "high", "urgent"]

Sentiment = Literal["neutral", "confused", "frustrated", "angry"]

Plan = Literal["starter", "team", "business", "enterprise", "unknown"]

#: Why the ladder stopped short of sending an answer.  This is the single most useful
#: field in the handoff packet: it tells the human what *kind* of help is needed before
#: they have read a word of the ticket.
Blocker = Literal[
    "needs_account_data",       # answer depends on data the agent cannot see
    "needs_policy_exception",   # the documented answer is "no" and the customer wants "yes"
    "needs_human_judgment",     # commercial, relational, or legal call
    "knowledge_gap",            # our KB genuinely does not cover this
    "out_of_scope",             # not something support owns at all
    "risk_flag",                # security, legal, churn, or press exposure
    "low_confidence",           # nothing specific; the agent simply is not sure
]

Route = Literal["send", "clarify", "escalate"]


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------


class CustomerContext(BaseModel):
    """What the CRM knows before the agent reads a word of the ticket."""

    account_id: str = ""
    company: str = ""
    plan: Plan = "unknown"
    seats: int = 0
    mrr_usd: float = 0.0
    tenure_months: int = 0
    open_tickets: int = 0
    prior_escalations_90d: int = 0
    csat_last: int = 0  # 0 = unknown, else 1-5
    notes: str = ""


class Ticket(BaseModel):
    """An inbound support ticket."""

    id: str
    subject: str
    body: str
    channel: Literal["email", "chat", "web_form", "in_app"] = "email"
    customer: CustomerContext = Field(default_factory=CustomerContext)
    #: Prior turns in this same ticket, oldest first.  Populated when the ladder loops
    #: back after a clarifying question.
    history: list[str] = Field(default_factory=list)

    def as_prompt_block(self) -> str:
        c = self.customer
        lines = [
            f"Ticket ID: {self.id}",
            f"Channel: {self.channel}",
            f"Subject: {self.subject}",
            "",
            "Customer record:",
            f"  company={c.company or 'unknown'} plan={c.plan} seats={c.seats}"
            f" mrr_usd={c.mrr_usd:.0f} tenure_months={c.tenure_months}",
            f"  open_tickets={c.open_tickets}"
            f" prior_escalations_90d={c.prior_escalations_90d}"
            f" csat_last={c.csat_last or 'unknown'}",
        ]
        if c.notes:
            lines.append(f"  notes: {c.notes}")
        lines += ["", "Message:", self.body.strip()]
        if self.history:
            lines += ["", "Earlier in this conversation:"]
            lines += [f"  - {h}" for h in self.history]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Stage 1 - classification
# --------------------------------------------------------------------------------------


class Classification(BaseModel):
    """Cheap triage pass.  Runs before retrieval so it can steer it."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity
    sentiment: Sentiment
    #: A one-line restatement of the ask, used as the retrieval query and as the
    #: subject line seed for a handoff.
    intent: str = Field(description="One sentence: what the customer actually wants.")
    #: Keywords to feed the retriever, in the vocabulary of a help-centre article rather
    #: than the customer's own words.
    search_terms: list[str]
    #: True when answering correctly requires reading this specific account's data
    #: (invoice amounts, sync queue state, token contents).  A hard escalation trigger.
    requires_account_data: bool
    #: True when the customer is asking for something outside documented policy
    #: (a refund past the window, a rate-limit bump, a backdated invoice).
    requests_exception: bool
    #: Churn, legal, security, or public-complaint language.
    risk_signals: list[str]
    #: Personal or sensitive data the customer pasted in, by kind, not by value.
    pii_present: list[str]


# --------------------------------------------------------------------------------------
# Stage 2 - retrieval + tier-1 draft
# --------------------------------------------------------------------------------------


class KBArticle(BaseModel):
    id: str
    title: str
    category: str
    tags: list[str] = Field(default_factory=list)
    body: str
    path: str = ""


class RetrievedChunk(BaseModel):
    """One scored passage from the knowledge base."""

    article_id: str
    title: str
    score: float
    text: str

    def as_prompt_block(self) -> str:
        return f"[{self.article_id}] {self.title}\n{self.text.strip()}"


class DraftAnswer(BaseModel):
    """The tier-1 agent's attempt, plus its own account of what it could not support."""

    model_config = ConfigDict(extra="forbid")

    #: The customer-facing reply.  Written as if it will be sent, because it might be.
    reply: str
    #: Article IDs actually used.  Validated against the real index downstream - a
    #: citation the retriever never returned is treated as a fabrication.
    citations: list[str]
    #: Things the reply asserts that the retrieved passages do not establish.  Asking
    #: the drafter to self-report these is cheap and correlates well with real errors.
    unsupported_claims: list[str]
    #: What the agent would need in order to be sure.  Feeds both the clarifying
    #: question and the handoff packet's open questions.
    information_gaps: list[str]
    #: The single question that would most reduce uncertainty, or "" if none would.
    best_clarifying_question: str
    #: The drafter's own read on whether the KB covers this at all.
    kb_covers_this: bool


# --------------------------------------------------------------------------------------
# Stage 3 - critique + confidence
# --------------------------------------------------------------------------------------


class CritiqueReport(BaseModel):
    """An independent reviewer's judgement of the draft.

    Deliberately a *separate* call from the drafter.  A model asked to rate its own
    answer anchors on having written it; a reviewer given the same evidence and the
    finished draft does not.
    """

    model_config = ConfigDict(extra="forbid")

    #: Is every claim in the reply traceable to the retrieved passages? 0.0-1.0
    groundedness: float = Field(ge=0.0, le=1.0)
    #: Does the reply address the whole ask, not just the easy part? 0.0-1.0
    coverage: float = Field(ge=0.0, le=1.0)
    #: How bad is it if this reply is wrong and we send it anyway? 0.0-1.0, where 1.0
    #: is "harmless" and 0.0 is "the customer loses money or data".
    action_safety: float = Field(ge=0.0, le=1.0)
    #: Specific problems, quoted from the draft where possible.
    problems: list[str]
    #: True when the missing piece is something the *customer* can supply - which is
    #: what separates "ask a clarifying question" from "escalate".
    blocked_on_customer_input: bool
    #: The reviewer's read on the primary blocker, used when routing to a human.
    blocker: Blocker
    reasoning: str


class ConfidenceReport(BaseModel):
    """Model judgement fused with mechanical checks the model cannot talk its way past."""

    score: float
    critique: CritiqueReport
    #: Deterministic signals computed from the retrieval index, not from any model.
    retrieval_top_score: float = 0.0
    retrieval_margin: float = 0.0
    invalid_citations: list[str] = Field(default_factory=list)
    uncited_reply: bool = False
    components: dict[str, float] = Field(default_factory=dict)
    penalties: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Stage 4 - clarification
# --------------------------------------------------------------------------------------


class ClarifyingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    #: What we expect to learn, recorded so the next pass round the ladder (and any
    #: human who inherits the ticket) knows why we stalled the customer.
    expected_information: list[str]


# --------------------------------------------------------------------------------------
# Stage 5 - the handoff packet
# --------------------------------------------------------------------------------------


class HandoffPacket(BaseModel):
    """A brief written for the human who picks this ticket up next.

    This is not a customer reply and must not read like one.  The reader is a colleague
    with sixty seconds and eleven other tickets: they need to know what this is, what has
    already been said to the customer, what is blocking it, and what to do first.
    """

    model_config = ConfigDict(extra="forbid")

    #: Scannable in a queue list: "[Billing/High] Duplicate annual charge - refund auth".
    subject_line: str
    #: One sentence a manager could read aloud in a standup.
    summary: str
    #: The ask in the customer's own framing, so the human does not re-derive it.
    customer_goal: str
    #: Everything the automated tier already did.  Prevents the human repeating steps
    #: and, more importantly, prevents them contradicting us.
    already_attempted: list[str]
    #: Verbatim of anything the customer has already been sent on this ticket.
    already_told_customer: str
    #: Evidence, each line tied to a KB article id.
    findings: list[str]
    #: What the agent could not determine, and why it could not.
    open_questions: list[str]
    blocker: Blocker
    #: Concrete and ordered.  "Check X in Y, then do Z" - not "investigate further".
    next_steps: list[str]
    #: A reply the human may edit and send.  Explicitly unverified.
    suggested_reply_draft: str
    #: Churn risk, legal exposure, security implications, executive visibility.
    risk_flags: list[str]
    priority: Severity
    #: Plain sentence on why the automated tier did not just answer this.
    why_escalated: str
    #: The feedback loop: if this was a knowledge gap, what should be written down?
    #: Empty string when the KB was not the problem.
    kb_gap: str


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


class Decision(BaseModel):
    route: Route
    reason: str
    #: True when a policy rule forced this route regardless of the confidence score.
    forced_by_policy: bool = False
    rule: str = ""


class StageTrace(BaseModel):
    """One row of the audit trail.  Every stage appends exactly one."""

    stage: str
    ok: bool = True
    detail: str = ""
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    degraded: bool = False


class Resolution(BaseModel):
    """The full result of running one ticket through the ladder."""

    ticket_id: str
    route: Route
    decision: Decision
    #: Present for ``send`` and ``clarify``.
    customer_reply: str = ""
    #: Present for ``escalate``.
    packet: HandoffPacket | None = None
    classification: Classification | None = None
    draft: DraftAnswer | None = None
    confidence: ConfidenceReport | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    trace: list[StageTrace] = Field(default_factory=list)
    #: True when at least one stage failed and the ladder fell back rather than raising.
    degraded: bool = False
    started_at: float = Field(default_factory=time.time)

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.input_tokens for s in self.trace),
            sum(s.output_tokens for s in self.trace),
        )
