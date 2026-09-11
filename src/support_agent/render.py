"""Turning a resolution into something a person reads.

The packet is a typed object so it can be posted to Zendesk, Linear, or Slack.  This
module renders the one form that matters for a sample: the markdown brief a support
engineer actually opens.  Field order here is reading order, not schema order - subject,
then why it is on their desk, then what has already been said, then the work.
"""

from __future__ import annotations

import time

from .handoff_lint import lint_packet
from .models import HandoffPacket, Resolution

_BLOCKER_LABELS = {
    "needs_account_data": "Needs account data the agent cannot read",
    "needs_policy_exception": "Needs a policy exception or approval",
    "needs_human_judgment": "Needs human judgement",
    "knowledge_gap": "Knowledge-base gap",
    "out_of_scope": "Out of scope for support",
    "risk_flag": "Risk flag",
    "low_confidence": "Low confidence, no specific blocker",
}


def _bullets(items: list[str], empty: str = "_none_") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def render_packet(packet: HandoffPacket, *, ticket_id: str = "") -> str:
    header = packet.subject_line
    if ticket_id:
        header = f"{header}  ·  `{ticket_id}`"

    sections = [
        f"# {header}",
        "",
        f"**Priority:** {packet.priority}  ·  "
        f"**Blocked on:** {_BLOCKER_LABELS.get(packet.blocker, packet.blocker)}",
        "",
        packet.summary,
        "",
        f"> **Why this is not answered already:** {packet.why_escalated}",
        "",
        "## Already said to the customer",
        "",
        packet.already_told_customer.strip() or "Nothing has been sent to the customer.",
        "",
        "## What the customer wants",
        "",
        packet.customer_goal,
        "",
        "## What the agent already did",
        "",
        _bullets(packet.already_attempted),
        "",
        "## Findings",
        "",
        _bullets(packet.findings),
        "",
        "## Open questions",
        "",
        _bullets(packet.open_questions),
        "",
        "## Next steps",
        "",
        "\n".join(f"{i}. {s}" for i, s in enumerate(packet.next_steps, 1))
        or "_none_",
    ]

    if packet.risk_flags:
        sections += ["", "## Risk flags", "", _bullets(packet.risk_flags)]

    if packet.suggested_reply_draft.strip():
        sections += [
            "",
            "## Suggested reply — unverified, check the findings first",
            "",
            "```",
            packet.suggested_reply_draft.strip(),
            "```",
        ]

    if packet.kb_gap.strip():
        sections += ["", "## Knowledge-base gap", "", packet.kb_gap.strip()]

    return "\n".join(sections).rstrip() + "\n"


_SITE_LABELS = {
    "draft": "an unsent reply",
    "escalation": "a finished escalation brief",
}


def _render_review(resolution: Resolution) -> str:
    """The review block: what was asked of a person, and what they said.

    Shown for a pending pause and a finished one alike.  A refusal gets the same
    prominence as a verdict, because "nobody looked at this in time" is the single most
    useful thing the trail can tell an operator.
    """
    record = resolution.review
    request = record.request
    lines = [
        "### Human review",
        "",
        f"- **Looking at:** {_SITE_LABELS.get(request.site, request.site)}",
        f"- **Because:** {request.reason}",
    ]

    if resolution.pending_review:
        waiting = max(request.expires_at - time.time(), 0.0)
        lines += [
            f"- **Waiting on:** `{request.thread_id}` — {waiting / 60:.0f} min left",
            f"- **Can:** {', '.join(f'`{a}`' for a in request.allowed_actions)}",
        ]
        return "\n".join(lines)

    if record.refused:
        lines += [f"- **Not applied:** {record.refused}"]

    if record.verdict is not None:
        verdict = record.verdict
        waited = verdict.decided_at - request.requested_at
        lines += [
            f"- **Decided:** `{verdict.action}` by **{verdict.reviewer}** "
            f"after {waited / 60:.0f} min",
        ]
        if verdict.rationale:
            lines += [f"- **Because:** {verdict.rationale}"]

    return "\n".join(lines)


def render_trace(resolution: Resolution) -> str:
    rows = ["| stage | ok | in | out | ms | detail |", "|---|---|---|---|---|---|"]
    for s in resolution.trace:
        mark = "ok" if s.ok else "FAIL"
        if s.degraded:
            mark += " (degraded)"
        detail = s.detail.replace("|", "\\|")
        rows.append(
            f"| {s.stage} | {mark} | {s.input_tokens} | {s.output_tokens} "
            f"| {s.output_tokens} | {s.duration_ms:.0f} | {detail} |"
        )
    tin, tout = resolution.total_tokens
    rows.append(f"| **total** |  |  | **{tin}** | **{tout}** |  |  |")
    return "\n".join(rows)


def render_resolution(resolution: Resolution, *, show_trace: bool = True) -> str:
    d = resolution.decision
    forced = " (forced by policy)" if d.forced_by_policy else ""
    parts = [
        f"## {resolution.ticket_id} → **{resolution.route.upper()}**{forced}",
        "",
        f"_{d.reason}_" + (f"  ·  rule: `{d.rule}`" if d.rule else ""),
    ]

    if resolution.confidence is not None:
        c = resolution.confidence
        comp = ", ".join(f"{k} {v:.2f}" for k, v in c.components.items())
        parts += ["", f"**Confidence {c.score:.2f}** — {comp}"]
        if c.penalties:
            parts += ["", "Penalties applied:", _bullets(c.penalties)]

    if resolution.retrieved:
        parts += [
            "",
            "**Retrieved:** "
            + ", ".join(
                f"{c.article_id} ({c.score:.2f})" for c in resolution.retrieved
            ),
        ]

    if resolution.review is not None:
        parts += ["", _render_review(resolution)]

    if resolution.route in ("send", "clarify"):
        label = "Reply sent" if resolution.route == "send" else "Clarifying question sent"
        parts += ["", f"### {label}", "", resolution.customer_reply.strip()]

    if resolution.packet is not None:
        parts += ["", "---", "", render_packet(resolution.packet,
                                               ticket_id=resolution.ticket_id)]
        findings = lint_packet(resolution.packet)
        if findings:
            parts += [
                "",
                "### Packet lint",
                "",
                _bullets([str(f) for f in findings]),
            ]

    if show_trace:
        parts += ["", "### Trace", "", render_trace(resolution)]

    return "\n".join(parts).rstrip() + "\n"
