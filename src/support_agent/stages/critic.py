"""Stage 3 - independent review, and the fusion that turns it into a routing number.

The reviewer is a *separate* call that never sees itself as the author.  Self-rating is
badly calibrated: a model that has just written an answer will defend it.  A reviewer
handed the same passages and the finished draft will not.

The reviewer's scores are then fused with checks that no amount of eloquence can move:
whether the cited article IDs actually exist in the index, whether retrieval found
anything at all, and how far the top passage beat the runner-up.  A fabricated citation
is not a rounding error on a confidence score - it is a hard cap.
"""

from __future__ import annotations

from ..config import Settings
from ..llm import LLMClient, Usage
from ..models import (
    ConfidenceReport,
    CritiqueReport,
    DraftAnswer,
    RetrievedChunk,
    Ticket,
)
from ..retrieval import KnowledgeBase

SYSTEM = """\
You are reviewing a draft support reply before it is sent to a customer without any \
human seeing it. You did not write it. Your job is to find what is wrong with it.

Assume the draft is wrong until the passages show otherwise. Check every factual claim - \
menu paths, numbers, day counts, plan names, policy boundaries - against the passages. A \
claim that is *probably* true of software like this but is not in the passages is \
ungrounded, and you should say so.

Score three things independently:

- `groundedness` (0-1): the fraction of the reply's factual content that the passages \
actually establish. One fabricated UI path or invented number puts this below 0.5, \
however good the rest is.

- `coverage` (0-1): does the reply answer the whole ask? Customers routinely ask two \
questions in one paragraph. A reply that nails the first and silently drops the second \
scores about 0.5, not 0.9.

- `action_safety` (0-1): how tolerable is it to send this if it turns out to be wrong? \
1.0 is a harmless explanation the customer would simply re-ask about. Below 0.5 means a \
wrong answer costs the customer money, data, or access - telling them to reset something \
that signs out their team, implying a refund they will not get, or giving billing timing \
they will plan around. Judge the *cost of being wrong*, not the likelihood.

`blocked_on_customer_input` is the field that decides whether we stall the customer with \
a question or hand this to a colleague. Set it true only when a specific fact the \
customer themselves can supply would resolve the uncertainty - an invoice number, which \
of two things they meant, what error text they saw. Set it false when the missing piece \
lives in our systems, when it needs a judgement call, or when the reply is simply wrong \
about something. Asking a customer a question we could have answered ourselves is worse \
than escalating.

`blocker` names the single reason this cannot be sent as-is, even if you scored it well - \
pick `low_confidence` only when nothing more specific fits.

`problems`: quote the offending phrase from the draft, then say what is wrong with it. \
Empty list if the draft is genuinely clean; say so plainly when it is.\
"""


def critique(
    llm: LLMClient,
    ticket: Ticket,
    draft: DraftAnswer,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[CritiqueReport, Usage]:
    """Review the draft against the evidence.

    Note what is *not* passed in: the triage classification.  The reviewer sees the
    ticket, the passages, and the draft, and nothing else.  Handing it the pipeline's
    own conclusions - "triage says this is a routine account question" - would give it
    something to agree with, and agreement is the failure mode this stage exists to
    avoid.
    """
    cfg = settings.stage("critique")
    passages = (
        "\n\n---\n\n".join(c.as_prompt_block() for c in chunks)
        if chunks
        else "(no knowledge-base passages matched this ticket)"
    )
    user = (
        "Knowledge-base passages the drafter was given:\n\n"
        f"{passages}\n\n"
        "======================================================================\n\n"
        f"The customer's ticket:\n\n{ticket.as_prompt_block()}\n\n"
        "======================================================================\n\n"
        f"The draft reply:\n\n{draft.reply}\n\n"
        f"Citations claimed: {draft.citations or 'none'}\n"
        f"Gaps the drafter self-reported: {draft.information_gaps or 'none'}\n"
        f"Claims the drafter flagged as unsupported: "
        f"{draft.unsupported_claims or 'none'}\n\n"
        "Review it."
    )
    return llm.structured(
        stage="critique",
        system=SYSTEM,
        user=user,
        schema=CritiqueReport,
        effort=cfg.effort,
        max_tokens=cfg.max_tokens,
    )


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def score_confidence(
    report: CritiqueReport,
    draft: DraftAnswer,
    chunks: list[RetrievedChunk],
    kb: KnowledgeBase,
    settings: Settings,
) -> ConfidenceReport:
    """Fuse reviewer judgement with mechanical evidence checks.

    The weighting is deliberate.  ``groundedness`` dominates because an ungrounded reply
    is the failure mode that actually reaches customers; ``action_safety`` is weighted
    next because it bounds the damage when groundedness is wrong.  Retrieval quality gets
    a small share: good retrieval does not make an answer right, but bad retrieval nearly
    guarantees it is not.
    """
    thresholds = settings.thresholds
    top = chunks[0].score if chunks else 0.0
    second = chunks[1].score if len(chunks) > 1 else 0.0
    margin = max(top - second, 0.0)

    # Mechanical check: did the drafter cite anything that does not exist, or that
    # retrieval never actually surfaced?
    surfaced = {c.article_id for c in chunks}
    invalid = [
        cid for cid in draft.citations if not kb.has_article(cid) or cid not in surfaced
    ]
    uncited = bool(draft.reply.strip()) and not draft.citations

    retrieval_quality = min(top / max(thresholds.min_retrieval_score * 3, 1e-6), 1.0)

    components = {
        "groundedness": report.groundedness,
        "action_safety": report.action_safety,
        "coverage": report.coverage,
        "retrieval": round(retrieval_quality, 4),
    }
    score = (
        0.40 * report.groundedness
        + 0.25 * report.action_safety
        + 0.20 * report.coverage
        + 0.15 * retrieval_quality
    )

    penalties: list[str] = []

    if invalid:
        # A citation the retriever never returned means the drafter is reasoning from
        # something other than the evidence.  Nothing else it says is trustworthy.
        score = min(score, 0.25)
        penalties.append(f"unverifiable citations: {', '.join(invalid)}")

    if draft.unsupported_claims:
        # Self-reported, so cheap to trust and cheap to act on.
        score -= 0.08 * len(draft.unsupported_claims)
        penalties.append(
            f"{len(draft.unsupported_claims)} self-reported unsupported claim(s)"
        )

    if not draft.kb_covers_this or not chunks:
        score = min(score, 0.35)
        penalties.append("knowledge base does not cover this topic")

    if top < thresholds.min_retrieval_score:
        score = min(score, 0.40)
        penalties.append(f"weak retrieval (top score {top:.3f})")

    if uncited:
        score -= 0.10
        penalties.append("reply cites no source")

    # Floors, not weights.  A reply can be well-covered, fluent, and confidently scored
    # and still be one that must never go out unreviewed.
    if report.groundedness < thresholds.min_groundedness:
        score = min(score, thresholds.auto_send - 0.01)
        penalties.append(f"groundedness {report.groundedness:.2f} below floor")
    if report.action_safety < thresholds.min_action_safety:
        score = min(score, thresholds.auto_send - 0.01)
        penalties.append(f"action safety {report.action_safety:.2f} below floor")
    if report.coverage < thresholds.min_coverage:
        score = min(score, thresholds.auto_send - 0.01)
        penalties.append(f"coverage {report.coverage:.2f} below floor")

    if report.blocked_on_customer_input:
        # Not a tuning knob - a consistency rule.  The reviewer has said it needs
        # something from the customer; sending a finished answer in the same breath
        # would be the pipeline contradicting itself.
        score = min(score, thresholds.auto_send - 0.01)
        penalties.append("reviewer says the answer depends on unasked customer input")

    return ConfidenceReport(
        score=round(max(0.0, min(score, 1.0)), 4),
        critique=report,
        retrieval_top_score=top,
        retrieval_margin=round(margin, 4),
        invalid_citations=invalid,
        uncited_reply=uncited,
        components=components,
        penalties=penalties,
    )
