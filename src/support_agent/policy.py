"""Deterministic escalation rules, evaluated before any confidence score is read.

Some tickets must reach a person no matter how well the model thinks it answered them.
Encoding that as a threshold is a mistake: thresholds are tuned, and the day someone
raises ``auto_send`` from 0.78 to 0.72 to lift the deflection rate should not be the day
the agent starts unilaterally denying refunds to accounts that mentioned their lawyer.

These rules are also the reason the ladder is not just "score, then branch".  Policy is a
separate, readable, testable layer that the model does not participate in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Settings
from .models import Blocker, Classification, Ticket

#: "$1,200", "1200 USD", "£300", "€75.50"
_MONEY = re.compile(
    r"(?:[$£€]\s?(\d[\d,]*(?:\.\d{1,2})?))|(?:(\d[\d,]*(?:\.\d{1,2})?)\s?(?:usd|eur|gbp|dollars|euros))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyHit:
    rule: str
    reason: str
    blocker: Blocker


def largest_amount(text: str) -> float:
    """Biggest currency figure mentioned anywhere in the ticket.

    Crude on purpose.  It is used only to decide whether a *person* looks at a money
    question, so over-triggering costs an escalation and under-triggering costs a refund
    approved by a language model.  The asymmetry picks the algorithm.
    """
    amounts: list[float] = []
    for match in _MONEY.finditer(text):
        raw = match.group(1) or match.group(2)
        try:
            amounts.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return max(amounts, default=0.0)


def _mentions(text: str, terms: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for term in terms:
        if term in lowered:
            return term.strip()
    return None


def check(
    ticket: Ticket, classification: Classification, settings: Settings
) -> PolicyHit | None:
    """Return the first rule that forces a handoff, or ``None`` to let the ladder run.

    Order matters: the rules are listed most-serious first so the reason attached to the
    escalation is the one a human would consider the headline.
    """
    policy = settings.policy
    haystack = f"{ticket.subject}\n{ticket.body}\n{' '.join(ticket.history)}"

    if term := _mentions(haystack, policy.legal_terms):
        return PolicyHit(
            rule="legal_or_privacy",
            reason=f"ticket references legal or data-protection process ({term!r})",
            blocker="risk_flag",
        )

    if term := _mentions(haystack, policy.security_terms):
        return PolicyHit(
            rule="security_report",
            reason=f"ticket may describe a security issue ({term!r})",
            blocker="risk_flag",
        )

    if classification.pii_present:
        return PolicyHit(
            rule="sensitive_data_in_ticket",
            reason=(
                "customer pasted sensitive data into the ticket "
                f"({', '.join(classification.pii_present)}); a person should handle it"
            ),
            blocker="risk_flag",
        )

    if term := _mentions(haystack, policy.churn_terms):
        return PolicyHit(
            rule="churn_risk",
            reason=f"ticket contains churn language ({term!r})",
            blocker="needs_human_judgment",
        )

    amount = largest_amount(haystack)
    if classification.category == "billing" and amount >= policy.refund_approval_usd:
        return PolicyHit(
            rule="money_above_agent_authority",
            reason=(
                f"billing ticket involving {amount:.0f}, at or above the "
                f"{policy.refund_approval_usd:.0f} approval line"
            ),
            blocker="needs_policy_exception",
        )

    if classification.requests_exception:
        return PolicyHit(
            rule="policy_exception_requested",
            reason="customer is asking for something outside documented policy",
            blocker="needs_policy_exception",
        )

    if classification.requires_account_data:
        return PolicyHit(
            rule="needs_account_data",
            reason="a correct answer requires account data the agent cannot read",
            blocker="needs_account_data",
        )

    if (
        ticket.customer.mrr_usd >= policy.high_touch_mrr_usd
        and classification.severity in ("high", "urgent")
    ):
        return PolicyHit(
            rule="high_touch_account",
            reason=(
                f"{classification.severity} ticket on a "
                f"${ticket.customer.mrr_usd:.0f}/mo account"
            ),
            blocker="needs_human_judgment",
        )

    if ticket.customer.prior_escalations_90d >= 3:
        return PolicyHit(
            rule="repeat_escalation",
            reason=(
                f"{ticket.customer.prior_escalations_90d} escalations in the last 90 "
                "days; the pattern needs a person"
            ),
            blocker="needs_human_judgment",
        )

    if term := _mentions(haystack, policy.exposure_terms):
        return PolicyHit(
            rule="public_exposure",
            reason=f"ticket references public or executive visibility ({term!r})",
            blocker="risk_flag",
        )

    return None
