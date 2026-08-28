"""Run three tickets through the ladder with no API key and no network.

Every model call is answered by :class:`ScriptedLLM` with the object a real stage would
have returned, so what you see below is the genuine routing logic, the genuine confidence
fusion, and the genuine renderer - only the model's words are canned.

    python examples/offline_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent import (  # noqa: E402
    KnowledgeBase,
    Ladder,
    ScriptedLLM,
    Settings,
    Ticket,
    render_resolution,
)
from support_agent.models import (  # noqa: E402
    ClarifyingQuestion,
    Classification,
    CritiqueReport,
    DraftAnswer,
    HandoffPacket,
)


def load(name: str) -> Ticket:
    path = next((ROOT / "examples" / "tickets").glob(f"{name}*.json"))
    return Ticket.model_validate(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------
# Scenario 1 - a routine question the agent answers on its own.
# --------------------------------------------------------------------------------------

CONFIDENT = {
    "classify": Classification(
        category="account",
        severity="normal",
        sentiment="confused",
        intent="Customer is not receiving the password reset email.",
        search_terms=["password reset", "reset link expiry", "sso enforcement",
                      "reset email not received"],
        requires_account_data=False,
        requests_exception=False,
        risk_signals=[],
        pii_present=[],
    ),
    "draft": DraftAnswer(
        reply=(
            "Two things usually explain this. Reset links expire after 60 minutes, and "
            "requesting a new one invalidates the previous link - so if you have "
            "clicked \"Forgot password\" six times, only the most recent email will "
            "work and the earlier ones are dead.\n\n"
            "The other possibility is that your workspace has SSO enforcement turned "
            "on. When it does, password reset is disabled entirely and everyone signs "
            "in through your identity provider instead. That would explain why your "
            "colleague can get in while you are stuck on the reset screen.\n\n"
            "Try the most recent reset email first, and allowlist "
            "no-reply@northwind-analytics.com if nothing has arrived at all. If your "
            "company uses Okta or similar to sign in elsewhere, go through that instead."
        ),
        citations=["kb-001"],
        unsupported_claims=[],
        information_gaps=[],
        best_clarifying_question="",
        kb_covers_this=True,
    ),
    "critique": CritiqueReport(
        groundedness=0.95,
        coverage=0.9,
        action_safety=0.95,
        problems=[],
        blocked_on_customer_input=False,
        blocker="low_confidence",
        reasoning=(
            "The 60-minute expiry, the invalidation of prior links, the allowlist "
            "address, and the SSO-enforcement behaviour are all stated in kb-001 and "
            "kb-002. Nothing in the reply commits us to anything."
        ),
    ),
}

# --------------------------------------------------------------------------------------
# Scenario 2 - the middle rung: one question, then stop.
# --------------------------------------------------------------------------------------

UNCERTAIN = {
    "classify": Classification(
        category="integrations",
        severity="high",
        sentiment="frustrated",
        intent="Customer's exec dashboard is showing stale figures and wants to know why.",
        search_terms=["sync latency", "freshness badge", "backfill", "sync interval",
                      "stale dashboard"],
        requires_account_data=False,
        requests_exception=False,
        risk_signals=[],
        pii_present=[],
    ),
    "draft": DraftAnswer(
        reply=(
            "Dashboards are as fresh as the last completed sync of the tables behind "
            "them, and on the Team plan connectors sync at most every 6 hours."
        ),
        citations=["kb-021", "kb-020"],
        unsupported_claims=[],
        information_gaps=[
            "which connector feeds the exec dashboard",
            "what the freshness badge currently reads",
            "whether a backfill is running",
        ],
        best_clarifying_question=(
            "Which connector feeds that dashboard, and what does its freshness badge say?"
        ),
        kb_covers_this=True,
    ),
    "critique": CritiqueReport(
        groundedness=0.9,
        coverage=0.45,
        action_safety=0.9,
        problems=[
            "\"on the Team plan connectors sync at most every 6 hours\" is correct but "
            "may not be the cause; the reply presents one of three documented "
            "explanations as though it were the answer.",
            "Does not address \"when will it catch up\", which is half the question.",
        ],
        blocked_on_customer_input=True,
        blocker="low_confidence",
        reasoning=(
            "kb-021 lists three causes - plan interval, a running backfill, and a "
            "source-side long transaction - and the freshness badge distinguishes them. "
            "The customer can read that badge in about ten seconds."
        ),
    ),
    "clarify": ClarifyingQuestion(
        reply=(
            "Dashboards show the timestamp of the last completed sync for the tables "
            "behind them, and on the Team plan connectors refresh at most every 6 "
            "hours - so some lag is expected, though not necessarily as much as you are "
            "seeing.\n\n"
            "To tell whether this is the normal interval or something stuck: which "
            "connector feeds that dashboard, and what does its freshness badge say? You "
            "can find both under Data > Connectors > (connector) > Activity, which will "
            "also show whether a backfill is running."
        ),
        expected_information=[
            "which connector feeds the exec dashboard",
            "the current freshness timestamp",
            "whether a backfill is in progress",
        ],
    ),
}

# --------------------------------------------------------------------------------------
# Scenario 3 - the policy gate fires, and the packet gets written.
# --------------------------------------------------------------------------------------

ESCALATED = {
    "classify": Classification(
        category="billing",
        severity="high",
        sentiment="frustrated",
        intent="Customer was charged $960 twice and wants the second charge refunded.",
        search_terms=["duplicate charge", "refund policy", "annual renewal",
                      "proration", "seat addition"],
        requires_account_data=True,
        requests_exception=False,
        risk_signals=["must explain the charge to their CFO"],
        pii_present=[],
    ),
    "handoff": HandoffPacket(
        subject_line="[Billing/High] Duplicate $960 annual charge - refund authorisation",
        summary=(
            "Brightpath Logistics was charged $960 on consecutive days and wants the "
            "second charge refunded today with written confirmation."
        ),
        customer_goal=(
            "The second $960 charge reversed, and something in writing they can show "
            "their CFO."
        ),
        already_attempted=[
            "Retrieved the refund policy and the billing-cycle article.",
            "Confirmed duplicate charges are refundable in full with no approval step "
            "(kb-011).",
            "Could not confirm the two charges are actually duplicates - invoice line "
            "items are not visible to the agent.",
        ],
        already_told_customer="Nothing has been sent to the customer.",
        findings=[
            "Duplicate or double charges are refunded in full, no approval needed "
            "(kb-011).",
            "Adding seats mid-cycle charges a prorated amount immediately, which can "
            "look like a second subscription charge (kb-010).",
            "Refunds go to the original payment method and take 5-10 business days "
            "(kb-011).",
            "Account renews annually in advance; two same-amount charges a day apart is "
            "not a normal renewal pattern.",
        ],
        open_questions=[
            "Whether the second $960 charge is a true duplicate or a mid-cycle seat "
            "addition that happens to match the renewal amount - the agent cannot read "
            "invoice line items.",
            "Whether the customer's finance team has already raised a chargeback, which "
            "would change how this is resolved.",
        ],
        blocker="needs_account_data",
        next_steps=[
            "Pull both charges for acct_2290 in Stripe and compare line items and "
            "invoice IDs.",
            "If both are renewals, refund the later charge - no approval needed under "
            "kb-011.",
            "If the second is a seat addition, reply explaining the proration rather "
            "than refunding, and copy the invoice line items.",
            "Either way, send written confirmation the same day; they have committed to "
            "their CFO.",
        ],
        suggested_reply_draft="",
        risk_flags=[
            "Customer has escalated this internally to their CFO.",
            "$1,600/mo account, 26 months tenure, CSAT 5 - a good account having a bad "
            "week.",
        ],
        priority="high",
        why_escalated=(
            "Confirming this is a duplicate rather than a proration requires invoice "
            "line items the agent cannot read, and the amount is above the agent's "
            "approval line."
        ),
        kb_gap="",
    ),
}


def main() -> int:
    settings = Settings()
    kb = KnowledgeBase.from_dir(settings.kb_dir)

    scenarios = [
        ("A routine question the agent answers itself", "TKT-1001", CONFIDENT),
        ("Not sure, but the customer can settle it", "TKT-1002", UNCERTAIN),
        ("The policy gate fires before a draft is even written", "TKT-1003", ESCALATED),
    ]

    for title, ticket_id, responses in scenarios:
        print("=" * 78)
        print(f"  {title}")
        print("=" * 78)
        print()
        llm = ScriptedLLM(responses=dict(responses))
        result = Ladder(llm, kb, settings).run(load(ticket_id))
        print(render_resolution(result))
        print()

    print("(Every model call above was scripted - no API key, no network.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
