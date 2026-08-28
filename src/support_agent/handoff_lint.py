"""Mechanical quality checks on a handoff packet.

The failure mode this catches is specific and persistent: a model that has spent four
stages writing to customers writes the internal brief to a customer too.  It opens with
"Thanks for reaching out", apologises to nobody, and turns "the refund is outside the
window" into "it may be that this could potentially fall outside".  A human skimming
eleven tickets loses the thread, and the escalation packet stops being worth more than
the raw ticket.

Prompting reduces this a lot.  Checking catches the rest, and - because these are plain
string rules - they can be asserted in tests without an API key or a judge model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import HandoffPacket

#: Phrases that only make sense addressed to a customer.
_CUSTOMER_VOICE = [
    "thanks for reaching out",
    "thank you for reaching out",
    "thanks for contacting",
    "i'm sorry to hear",
    "i am sorry to hear",
    "sorry for the inconvenience",
    "apologies for the inconvenience",
    "we apologize",
    "we apologise",
    "hope this helps",
    "happy to help",
    "i'd be happy to",
    "i would be happy to",
    "let me know if you have any",
    "feel free to reach out",
    "don't hesitate to",
    "best regards",
    "kind regards",
    "warm regards",
]

_GREETING = re.compile(r"^\s*(hi|hello|hey|dear|good (morning|afternoon|evening))\b", re.I)

#: Hedges that add length without adding information in an internal brief.
_HEDGES = [
    "it may possibly",
    "might potentially",
    "could potentially",
    "it seems like it might",
    "it appears that it may",
    "perhaps possibly",
]

#: Next steps that are not steps.
_VAGUE_STEPS = [
    "investigate further",
    "look into it",
    "look into this",
    "follow up as needed",
    "review the issue",
    "review the situation",
    "take appropriate action",
    "handle accordingly",
    "assist the customer",
    "reach out to the customer",  # to say what? name the message.
]

_SUBJECT_SHAPE = re.compile(r"^\s*\[[^\]/]+/[^\]]+\]\s*\S")
_KB_REF = re.compile(r"\bkb-\d{3}\b", re.I)


@dataclass(frozen=True)
class LintFinding:
    field: str
    severity: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.field}: {self.message}"


def _internal_text(packet: HandoffPacket) -> list[tuple[str, str]]:
    """Every field whose audience is the colleague, not the customer.

    ``suggested_reply_draft`` is excluded by design - it is supposed to sound like
    support prose, because it is.
    """
    fields: list[tuple[str, str]] = [
        ("subject_line", packet.subject_line),
        ("summary", packet.summary),
        ("customer_goal", packet.customer_goal),
        ("why_escalated", packet.why_escalated),
        ("kb_gap", packet.kb_gap),
    ]
    for name, items in (
        ("already_attempted", packet.already_attempted),
        ("findings", packet.findings),
        ("open_questions", packet.open_questions),
        ("next_steps", packet.next_steps),
        ("risk_flags", packet.risk_flags),
    ):
        fields += [(f"{name}[{i}]", text) for i, text in enumerate(items)]
    return fields


def lint_packet(packet: HandoffPacket) -> list[LintFinding]:
    """Return everything wrong with this brief, worst first."""
    findings: list[LintFinding] = []

    for field_name, text in _internal_text(packet):
        lowered = text.lower()
        for phrase in _CUSTOMER_VOICE:
            if phrase in lowered:
                findings.append(
                    LintFinding(
                        field_name,
                        "error",
                        f"customer-facing phrasing in an internal brief: {phrase!r}",
                    )
                )
        if _GREETING.match(text):
            findings.append(
                LintFinding(field_name, "error", "brief opens with a greeting")
            )
        for hedge in _HEDGES:
            if hedge in lowered:
                findings.append(
                    LintFinding(field_name, "warning", f"stacked hedge: {hedge!r}")
                )

    # -- structure ---------------------------------------------------------------------

    if not _SUBJECT_SHAPE.match(packet.subject_line):
        findings.append(
            LintFinding(
                "subject_line",
                "warning",
                "expected the scannable '[Category/Priority] thing - need' shape",
            )
        )
    if len(packet.subject_line) > 90:
        findings.append(
            LintFinding(
                "subject_line",
                "warning",
                f"{len(packet.subject_line)} chars; too long to scan in a queue",
            )
        )

    if not packet.next_steps:
        findings.append(
            LintFinding("next_steps", "error", "no next steps; the brief does not act")
        )
    for i, step in enumerate(packet.next_steps):
        lowered = step.lower()
        for vague in _VAGUE_STEPS:
            if vague in lowered:
                findings.append(
                    LintFinding(
                        f"next_steps[{i}]",
                        "error",
                        f"not an actionable step: {vague!r}",
                    )
                )

    if packet.findings and not any(_KB_REF.search(f) for f in packet.findings):
        findings.append(
            LintFinding(
                "findings",
                "warning",
                "no finding is traceable to a KB article id",
            )
        )

    for i, question in enumerate(packet.open_questions):
        # A bare question tells the reader what we do not know but not why we could not
        # find out, which is the part that decides who should pick it up.
        if question.rstrip().endswith("?") and len(question.split()) < 14:
            findings.append(
                LintFinding(
                    f"open_questions[{i}]",
                    "warning",
                    "bare question; pair it with why the agent could not determine it",
                )
            )

    if not packet.already_told_customer.strip():
        findings.append(
            LintFinding(
                "already_told_customer",
                "error",
                "must state what was sent, or explicitly that nothing was",
            )
        )

    order = {"error": 0, "warning": 1}
    findings.sort(key=lambda f: (order[f.severity], f.field))
    return findings


def has_errors(findings: list[LintFinding]) -> bool:
    return any(f.severity == "error" for f in findings)
