"""Node bodies for the escalation ladder graph.

Each function here is one rung's worth of work.  Nodes never raise: a stage that cannot
get a usable answer out of the model catches :class:`LLMError`, records the failure on the
trace, and sets ``escalation_reason``, which every router downstream reads as "stop
climbing, hand this to a person".  That is what keeps degradation *one-directional* - a
broken pipeline can only ever route more conservatively, never less.

The topology that wires these together lives in :mod:`support_agent.graph`, deliberately
apart from the node bodies so the shape of the ladder can be read on one screen.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from operator import add
from typing import Annotated

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from .config import Settings
from .handoff_lint import has_errors, lint_packet
from .llm import LLMClient, LLMError, Usage
from .models import (
    Classification,
    ConfidenceReport,
    CritiqueReport,
    Decision,
    DraftAnswer,
    HandoffPacket,
    Resolution,
    RetrievedChunk,
    Route,
    StageTrace,
    Ticket,
)
from .policy import check as policy_check
from .retrieval import KnowledgeBase
from .stages.clarify import ask_clarifying_question
from .stages.classifier import classify
from .stages.critic import critique, score_confidence
from .stages.handoff import build_handoff_packet
from .stages.tier1 import draft_answer

AGENT_PREFIX = "AGENT:"


# --------------------------------------------------------------------------------------
# Dependencies and state
# --------------------------------------------------------------------------------------


@dataclass
class LadderDeps:
    """Injected per invocation via LangGraph's runtime context.

    These live outside the state on purpose.  State gets checkpointed and must stay
    serialisable; an HTTP client and a loaded search index are neither.
    """

    llm: LLMClient
    kb: KnowledgeBase
    settings: Settings


class LadderState(BaseModel):
    """Everything the ladder accumulates about one ticket.

    Serialisable throughout, so a run can be checkpointed mid-ladder and resumed - which
    is the point of the customer-reply round trip in the clarify branch.
    """

    ticket: Ticket

    classification: Classification | None = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    draft: DraftAnswer | None = None
    critique: CritiqueReport | None = None
    confidence: ConfidenceReport | None = None

    route: Route = "escalate"
    decision: Decision | None = None
    customer_reply: str = ""
    packet: HandoffPacket | None = None

    #: Non-empty means "a human takes this from here".  Set by the policy gate, by the
    #: confidence gate, and by any stage that failed.  Every router checks it.
    escalation_reason: str = ""
    #: Extra context for the packet when the pipeline itself is what went wrong.
    degraded_note: str = ""
    #: Verbatim of the last thing we sent this customer, carried into the packet.
    already_told: str = ""

    #: Lint findings from the previous handoff attempt, fed back into the next one.
    lint_feedback: list[str] = Field(default_factory=list)
    handoff_attempts: int = 0

    trace: Annotated[list[StageTrace], add] = Field(default_factory=list)
    degraded: bool = False


# --------------------------------------------------------------------------------------
# Trace helpers
# --------------------------------------------------------------------------------------


def _row(
    stage: str,
    usage: Usage | None = None,
    *,
    ok: bool = True,
    detail: str = "",
    degraded: bool = False,
) -> StageTrace:
    return StageTrace(
        stage=stage,
        ok=ok,
        detail=detail,
        duration_ms=round(usage.duration_ms, 2) if usage else 0.0,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        degraded=degraded,
    )


def _failure(stage: str, exc: LLMError) -> StageTrace:
    # `retryable` distinguishes "the API was briefly unavailable" from "the model
    # refused" or "the output did not validate".  The SDK has already retried the
    # transient ones by the time we see them, so this is for the human reading the
    # trace afterwards, not for control flow.
    kind = "transient" if exc.retryable else "permanent"
    return _row(stage, ok=False, detail=f"{exc.detail} [{kind}]", degraded=True)


# --------------------------------------------------------------------------------------
# Conversation helpers
# --------------------------------------------------------------------------------------


def clarify_rounds_used(ticket: Ticket) -> int:
    """How many times we have already stalled this customer with a question."""
    return sum(1 for h in ticket.history if h.strip().startswith(AGENT_PREFIX))


def last_agent_message(ticket: Ticket) -> str:
    for entry in reversed(ticket.history):
        stripped = entry.strip()
        if stripped.startswith(AGENT_PREFIX):
            return stripped[len(AGENT_PREFIX):].strip()
    return ""


# --------------------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------------------


def fallback_classification(ticket: Ticket) -> Classification:
    """A classification built without a model, for when triage itself is unavailable.

    Deliberately pessimistic: it claims nothing it cannot know, and it sets
    ``requires_account_data`` so that any code path which reaches the policy layer with
    it still routes to a human.  The graph short-circuits before that anyway - a degraded
    pipeline should never be a more permissive one.
    """
    return Classification(
        category="other",
        severity="normal",
        sentiment="neutral",
        intent=ticket.subject or ticket.body[:120],
        search_terms=[ticket.subject] if ticket.subject else [],
        requires_account_data=True,
        requests_exception=False,
        risk_signals=["triage unavailable - classification not verified"],
        pii_present=[],
    )


def fallback_packet(state: LadderState, reason: str) -> HandoffPacket:
    """The floor of the ladder: a brief assembled with no model in the loop.

    Worse than a written one - no synthesis, no prioritised next steps - but it preserves
    the properties that make a packet useful: what is known, what was already said to the
    customer, and why it is here.
    """
    classification = state.classification
    category = classification.category if classification else "other"
    severity = classification.severity if classification else "normal"
    goal = classification.intent if classification else (state.ticket.subject or "unclear")

    attempted = ["Automated triage and knowledge-base retrieval ran."]
    if state.draft is not None:
        attempted.append("A draft reply was produced but not verified or sent.")
    if state.confidence is not None:
        attempted.append(f"Confidence assessment scored {state.confidence.score:.2f}.")
    attempted.append("Brief generation was unavailable; this packet is auto-assembled.")

    return HandoffPacket(
        subject_line=f"[{category.title()}/{severity.title()}] "
        f"{(state.ticket.subject or 'Support request')[:60]} - degraded handoff",
        summary=(
            f"{state.ticket.customer.company or 'Customer'} ticket {state.ticket.id} "
            f"could not be handled automatically: {reason}"
        ),
        customer_goal=goal,
        already_attempted=attempted,
        already_told_customer=state.already_told
        or "Nothing has been sent to the customer.",
        findings=(
            [f"Unverified draft the agent produced: {state.draft.reply[:400]}"]
            if state.draft is not None
            else ["No draft was produced."]
        ),
        open_questions=[
            "Everything - the automated pipeline degraded before it reached a "
            "conclusion, so nothing here has been checked."
        ],
        blocker="low_confidence",
        next_steps=[
            f"Read ticket {state.ticket.id} directly; treat this packet as a pointer "
            "only.",
            "Check the pipeline trace for which stage failed before relying on any "
            "field above.",
        ],
        suggested_reply_draft="",
        risk_flags=["Handoff packet was auto-assembled after a pipeline failure."],
        priority=severity,
        why_escalated=reason,
        kb_gap="",
    )


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------


def classify_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    deps = runtime.context
    already_told = last_agent_message(state.ticket)
    try:
        classification, usage = classify(deps.llm, state.ticket, deps.settings)
    except LLMError as exc:
        return {
            "classification": fallback_classification(state.ticket),
            "already_told": already_told,
            "degraded": True,
            # Triage failed, so we never actually classified this ticket.  The policy
            # rules read fields the fallback only guessed at, and a human deserves the
            # real reason rather than a rule name derived from a guess.
            "escalation_reason": (
                "triage stage was unavailable, so the ticket was never classified"
            ),
            "degraded_note": f"classify stage failed: {exc.detail}",
            "decision": Decision(
                route="escalate",
                reason="degraded pipeline: triage unavailable",
                forced_by_policy=True,
                rule="degraded_triage",
            ),
            "trace": [_failure("classify", exc)],
        }
    return {
        "classification": classification,
        "already_told": already_told,
        "trace": [
            _row("classify", usage,
                 detail=f"{classification.category}/{classification.severity}")
        ],
    }


def retrieve_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    """Local and dependency-free, so it runs even on the degraded path.

    Evidence in the packet is worth having whether or not the model stages worked.
    """
    deps = runtime.context
    classification = state.classification
    query = " ".join([classification.intent, *classification.search_terms]) or (
        f"{state.ticket.subject} {state.ticket.body}"
    )
    started = time.perf_counter()
    chunks = deps.kb.search(query, top_k=deps.settings.top_k)
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "retrieved": chunks,
        "trace": [
            StageTrace(
                stage="retrieve",
                detail=f"{len(chunks)} chunk(s), top={chunks[0].score:.3f}"
                if chunks
                else "no matches",
                duration_ms=round(elapsed, 2),
            )
        ],
    }


def policy_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    """Deterministic escalation rules, evaluated before any confidence score exists."""
    deps = runtime.context
    hit = policy_check(state.ticket, state.classification, deps.settings)
    if hit is None:
        return {"trace": [_row("policy", detail="no forced-escalation rule matched")]}
    return {
        "escalation_reason": hit.reason,
        "decision": Decision(
            route="escalate",
            reason=hit.reason,
            forced_by_policy=True,
            rule=hit.rule,
        ),
        "trace": [_row("policy", detail=f"{hit.rule}: {hit.reason}")],
    }


def draft_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    deps = runtime.context
    try:
        draft, usage = draft_answer(
            deps.llm, state.ticket, state.classification, state.retrieved, deps.settings
        )
    except LLMError as exc:
        return {
            "degraded": True,
            "escalation_reason": "the tier-1 agent could not produce a draft",
            "degraded_note": f"draft stage failed: {exc.detail}",
            "decision": Decision(
                route="escalate",
                reason=f"draft stage unavailable ({exc.detail})",
                forced_by_policy=True,
                rule="degraded_draft",
            ),
            "trace": [_failure("draft", exc)],
        }
    return {
        "draft": draft,
        "trace": [_row("draft", usage, detail=f"{len(draft.citations)} citation(s)")],
    }


def critique_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    """Independent review, then fusion with the mechanical evidence checks."""
    deps = runtime.context
    try:
        report, usage = critique(
            deps.llm, state.ticket, state.draft, state.retrieved, deps.settings
        )
    except LLMError as exc:
        # We have an answer that nothing has checked.  That is precisely the situation
        # a human review queue exists for.
        return {
            "degraded": True,
            "escalation_reason": (
                "the draft could not be reviewed, so it was not sent unchecked"
            ),
            "degraded_note": f"critique stage failed: {exc.detail}",
            "decision": Decision(
                route="escalate",
                reason=f"review stage unavailable ({exc.detail})",
                forced_by_policy=True,
                rule="degraded_critique",
            ),
            "trace": [_failure("critique", exc)],
        }

    confidence = score_confidence(
        report, state.draft, state.retrieved, deps.kb, deps.settings
    )
    return {
        "critique": report,
        "confidence": confidence,
        "trace": [
            _row("critique", usage, detail=f"blocker={report.blocker}"),
            _row(
                "score",
                detail=f"score={confidence.score:.2f} "
                f"penalties={len(confidence.penalties)}",
            ),
        ],
    }


def send_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    thresholds = runtime.context.settings.thresholds
    return {
        "route": "send",
        "customer_reply": state.draft.reply,
        "decision": Decision(
            route="send",
            reason=(
                f"confidence {state.confidence.score:.2f} at or above the "
                f"{thresholds.auto_send:.2f} auto-send threshold, with no floor breached"
            ),
        ),
        "trace": [_row("route", detail="send")],
    }


def clarify_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    deps = runtime.context
    thresholds = deps.settings.thresholds
    try:
        question, usage = ask_clarifying_question(
            deps.llm, state.ticket, state.draft, state.critique, deps.settings
        )
    except LLMError as exc:
        # Fall through to escalation rather than send the unverified draft.
        reason = "the clarifying question could not be written"
        return {
            "degraded": True,
            "escalation_reason": reason,
            "degraded_note": f"clarify stage failed: {exc.detail}",
            "decision": Decision(
                route="escalate", reason=reason, rule="degraded_clarify"
            ),
            "trace": [_failure("clarify", exc)],
        }
    return {
        "route": "clarify",
        "customer_reply": question.reply,
        "decision": Decision(
            route="clarify",
            reason=(
                f"confidence {state.confidence.score:.2f} is below the "
                f"{thresholds.auto_send:.2f} send threshold, but the reviewer says the "
                "customer can supply what is missing"
            ),
        ),
        "trace": [
            _row("clarify", usage,
                 detail=f"expects {len(question.expected_information)} fact(s)"),
            _row("route", detail="clarify"),
        ],
    }


def handoff_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    deps = runtime.context
    attempt = state.handoff_attempts + 1
    try:
        packet, usage = build_handoff_packet(
            deps.llm,
            state.ticket,
            state.classification,
            state.draft,
            state.confidence,
            state.retrieved,
            deps.settings,
            escalation_reason=state.escalation_reason,
            already_told_customer=state.already_told,
            degraded_note=state.degraded_note,
            lint_feedback=state.lint_feedback,
        )
    except LLMError as exc:
        return {
            "route": "escalate",
            "packet": fallback_packet(state, state.escalation_reason),
            "handoff_attempts": attempt,
            "degraded": True,
            "trace": [
                _row("handoff", ok=False,
                     detail=f"{exc.detail}; auto-assembled packet", degraded=True)
            ],
        }
    detail = packet.blocker if attempt == 1 else f"{packet.blocker} (rewrite {attempt})"
    return {
        "route": "escalate",
        "packet": packet,
        "handoff_attempts": attempt,
        "trace": [_row("handoff", usage, detail=detail)],
    }


def handoff_lint_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    """Check the brief, and feed real failures back for one rewrite.

    Observing a lint failure and shipping the packet anyway - which is what this pipeline
    did before the graph existed - wastes the check.  Errors get exactly one rewrite;
    warnings are recorded and shipped, because a slightly long subject line is not worth
    a second model call.
    """
    settings = runtime.context.settings
    findings = lint_packet(state.packet)

    if not findings:
        return {"lint_feedback": [], "trace": [_row("handoff_lint", detail="clean")]}

    rendered = [str(f) for f in findings]
    summary = "; ".join(rendered[:4])
    if len(rendered) > 4:
        summary += f" (+{len(rendered) - 4} more)"

    errors = has_errors(findings)
    retrying = errors and state.handoff_attempts < settings.thresholds.max_handoff_attempts
    if retrying:
        summary += " - rewriting"

    return {
        # Only real errors are worth a rewrite; warnings would burn a call on nits.
        "lint_feedback": [str(f) for f in findings if f.severity == "error"],
        "trace": [_row("handoff_lint", ok=not errors, detail=summary)],
    }


def finalise_escalation_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    decision = state.decision or Decision(
        route="escalate", reason=state.escalation_reason, rule="confidence_gate"
    )
    return {"route": "escalate", "decision": decision,
            "trace": [_row("route", detail="escalate")]}


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def to_resolution(state: LadderState, started_at: float) -> Resolution:
    return Resolution(
        ticket_id=state.ticket.id,
        route=state.route,
        decision=state.decision
        or Decision(route=state.route, reason="not yet decided"),
        customer_reply=state.customer_reply,
        packet=state.packet,
        classification=state.classification,
        draft=state.draft,
        confidence=state.confidence,
        retrieved=state.retrieved,
        trace=state.trace,
        degraded=state.degraded or any(s.degraded for s in state.trace),
        started_at=started_at,
    )


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------


def gate_node(state: LadderState, runtime: Runtime[LadderDeps]) -> dict:
    """The confidence gate: the one branch this whole application exists to get right.

    It writes a provisional ``route`` rather than dispatching directly, so the decision
    and the dispatch stay separable - the reason a ticket went one way is recorded even
    when the branch it chose then fails.
    """
    thresholds = runtime.context.settings.thresholds
    confidence = state.confidence
    report = state.critique
    rounds_used = clarify_rounds_used(state.ticket)

    if confidence.score >= thresholds.auto_send:
        return {"route": "send"}

    can_clarify = (
        confidence.score >= thresholds.clarify_floor
        and report.blocked_on_customer_input
        and rounds_used < thresholds.max_clarify_rounds
    )
    if can_clarify:
        return {"route": "clarify"}

    exhausted = rounds_used >= thresholds.max_clarify_rounds
    if exhausted and report.blocked_on_customer_input:
        reason = (
            f"the customer has already been asked {rounds_used} question(s) and the "
            "answer is still not settled"
        )
    elif report.blocked_on_customer_input:
        reason = (
            f"confidence {confidence.score:.2f} is below the "
            f"{thresholds.clarify_floor:.2f} floor for even asking a question"
        )
    else:
        reason = (
            f"confidence {confidence.score:.2f} is below the "
            f"{thresholds.auto_send:.2f} send threshold and the missing piece is not "
            "something the customer can supply"
        )

    return {
        "route": "escalate",
        "escalation_reason": reason,
        "decision": Decision(route="escalate", reason=reason, rule="confidence_gate"),
    }
