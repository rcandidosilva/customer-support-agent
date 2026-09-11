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

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

#: ``review`` is not a terminal route: it means the run is paused on a human and will
#: resolve to one of the other three once a verdict arrives (or the deadline passes).
Route = Literal["send", "clarify", "escalate", "review"]


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
# Human review
# --------------------------------------------------------------------------------------

#: What a reviewer is allowed to say.  The *route* each action produces depends on where
#: the run paused, not on the action itself - which is what lets the same vocabulary serve
#: a reviewer looking at an escalation brief and (later) one looking at a draft reply.
#:
#: ``approve``       - release what the agent produced, unchanged.
#: ``edit_and_send`` - send ``edited_reply`` to the customer instead.
#: ``ask_customer``  - send ``edited_reply`` to the customer as a clarifying question.
#: ``escalate``      - override to the human queue, saying why in ``rationale``.
ReviewAction = Literal["approve", "edit_and_send", "ask_customer", "escalate"]

#: Where a run stopped.  The two sites ask genuinely different questions, and the same
#: action means different things at each:
#:
#: ``draft``     - "is this reply safe to send?"  ``approve`` releases the agent's draft.
#: ``escalation`` - "is this brief ready for the queue?"  ``approve`` queues it.
#:
#: Carried on the request rather than inferred, so the record of a finished review says
#: what was actually being asked.
ReviewSite = Literal["draft", "escalation"]


class ReviewRequest(BaseModel):
    """What a paused run hands the reviewer.

    Deliberately thin: the artifact under review is already on the :class:`Resolution`
    (the packet, the draft, the confidence breakdown), so duplicating it here would give
    two copies that can disagree.  This carries only what is true of the *pause* itself.

    The instance persisted with the interrupt is authoritative.  The node rebuilds an
    equivalent request when it re-executes on resume - LangGraph re-runs a node's body up
    to its ``interrupt()`` - so ``requested_at`` and ``expires_at`` computed in the node
    are not trustworthy, and only the checkpointed copy is used to judge the deadline.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    thread_id: str
    #: Which question is being asked.  Determines what the actions mean and where each
    #: one sends the run.
    site: ReviewSite = "escalation"
    #: Why the run stopped here, in the words already recorded on the escalation.
    reason: str
    #: Actions valid at *this* pause site.  A verdict outside this set is refused.
    allowed_actions: list[ReviewAction]
    requested_at: float = Field(default_factory=time.time)
    #: Wall clock past which no verdict is accepted and the run finalises as an
    #: escalation.  A pause nobody answers must not become a send.
    expires_at: float = 0.0

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) > self.expires_at


class ReviewVerdict(BaseModel):
    """One human decision, validated the way every other boundary here is.

    The resume value of a LangGraph ``interrupt()`` is whatever the caller passes, which
    makes it the least-typed input in the system and the one with the most consequence -
    it can put text in front of a customer.  So it gets the same treatment as a model
    response: a schema, ``extra="forbid"``, and cross-field checks.
    """

    model_config = ConfigDict(extra="forbid")

    action: ReviewAction
    #: Who decided.  This is the audit record; an anonymous approval is not one.
    reviewer: str
    #: The reply the reviewer wrote.  Required by the two actions that reach a customer.
    edited_reply: str = ""
    #: Why they overrode the agent.  Required when escalating against its judgement.
    rationale: str = ""
    decided_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def _complete_for_action(self) -> ReviewVerdict:
        if not self.reviewer.strip():
            raise ValueError("a verdict needs a named reviewer")
        needs_reply = self.action in ("edit_and_send", "ask_customer")
        if needs_reply and not self.edited_reply.strip():
            raise ValueError(f"action {self.action!r} needs a non-empty edited_reply")
        if self.action == "escalate" and not self.rationale.strip():
            raise ValueError("action 'escalate' needs a rationale")
        return self


class ReviewRecord(BaseModel):
    """The audit trail of one review: what was asked, and what came back.

    Present on the :class:`Resolution` whether the verdict was accepted, refused, or
    never arrived - a refused verdict is exactly the event worth being able to find later.
    """

    request: ReviewRequest
    #: ``None`` when the deadline passed or the verdict was refused.
    verdict: ReviewVerdict | None = None
    #: Empty when the verdict was accepted; otherwise why it was not.
    refused: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict is not None and not self.refused


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
    #: Which model answered, as ``provider:model``.  Empty for the rows no model
    #: produced - the deterministic fallbacks, and the failures that never got a
    #: response.  With per-stage routing this is what makes the token columns costable.
    model: str = ""


class Resolution(BaseModel):
    """The full result of running one ticket through the ladder."""

    ticket_id: str
    route: Route
    decision: Decision
    #: Present for ``send`` and ``clarify``.
    customer_reply: str = ""
    #: Present for ``escalate``.
    packet: HandoffPacket | None = None
    #: Present once a run has paused on a human.  While ``route == "review"`` its
    #: ``verdict`` is ``None``: the request is out and nothing has come back.
    review: ReviewRecord | None = None
    classification: Classification | None = None
    draft: DraftAnswer | None = None
    confidence: ConfidenceReport | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    trace: list[StageTrace] = Field(default_factory=list)
    #: True when at least one stage failed and the ladder fell back rather than raising.
    degraded: bool = False
    started_at: float = Field(default_factory=time.time)

    @property
    def pending_review(self) -> bool:
        """True when this resolution is a pause, not an outcome.

        The reason ``run()`` can keep returning a ``Resolution``: a paused run is a
        resolution whose route happens to be non-terminal, so callers that never enable
        review are unaffected and callers that do have one flag to check.
        """
        return self.route == "review"

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (
            sum(s.input_tokens for s in self.trace),
            sum(s.output_tokens for s in self.trace),
        )
