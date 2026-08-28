"""Builders for the offline test suite.

Every test in this project runs without an API key.  The ladder is exercised end to end
by feeding :class:`ScriptedLLM` the objects a real stage would have returned, which means
routing, fusion, degradation, and packet quality are all assertable in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.models import (  # noqa: E402
    ClarifyingQuestion,
    Classification,
    CritiqueReport,
    CustomerContext,
    DraftAnswer,
    HandoffPacket,
    Ticket,
)
from support_agent.retrieval import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def kb(settings: Settings) -> KnowledgeBase:
    return KnowledgeBase.from_dir(settings.kb_dir)


def make_ticket(**overrides) -> Ticket:
    data = {
        "id": "TKT-TEST",
        "subject": "Password reset email never arrives",
        "body": "I clicked forgot password several times and nothing arrives.",
        "channel": "email",
        "customer": CustomerContext(
            account_id="acct_1", company="Testco", plan="team", seats=10,
            mrr_usd=300, tenure_months=12,
        ),
    }
    data.update(overrides)
    return Ticket(**data)


def make_classification(**overrides) -> Classification:
    data = {
        "category": "account",
        "severity": "normal",
        "sentiment": "confused",
        "intent": "Customer cannot receive the password reset email.",
        "search_terms": ["password reset", "reset email", "sso enforcement"],
        "requires_account_data": False,
        "requests_exception": False,
        "risk_signals": [],
        "pii_present": [],
    }
    data.update(overrides)
    return Classification(**data)


def make_draft(**overrides) -> DraftAnswer:
    data = {
        "reply": (
            "Reset links expire after 60 minutes and requesting a new one invalidates "
            "the previous link. If your workspace enforces SSO, password reset is "
            "disabled entirely and you will need to sign in through your identity "
            "provider."
        ),
        "citations": ["kb-001"],
        "unsupported_claims": [],
        "information_gaps": [],
        "best_clarifying_question": "",
        "kb_covers_this": True,
    }
    data.update(overrides)
    return DraftAnswer(**data)


def make_critique(**overrides) -> CritiqueReport:
    data = {
        "groundedness": 0.95,
        "coverage": 0.9,
        "action_safety": 0.95,
        "problems": [],
        "blocked_on_customer_input": False,
        "blocker": "low_confidence",
        "reasoning": "Every claim maps to kb-001.",
    }
    data.update(overrides)
    return CritiqueReport(**data)


def make_clarify(**overrides) -> ClarifyingQuestion:
    data = {
        "reply": (
            "Reset links expire after 60 minutes, so a stale link is the usual cause. "
            "Does your workspace sign in through Okta or another identity provider? "
            "If it does, password reset is switched off and you will need to go "
            "through them instead."
        ),
        "expected_information": ["whether the workspace enforces SSO"],
    }
    data.update(overrides)
    return ClarifyingQuestion(**data)


def make_packet(**overrides) -> HandoffPacket:
    data = {
        "subject_line": "[Billing/High] Duplicate annual charge - refund authorisation",
        "summary": (
            "Brightpath was charged $960 twice in two days and wants the second charge "
            "refunded today."
        ),
        "customer_goal": "A refund of the second $960 charge, with written confirmation.",
        "already_attempted": [
            "Searched the knowledge base for duplicate-charge and refund policy.",
            "Confirmed the documented policy refunds duplicate charges in full.",
        ],
        "already_told_customer": "Nothing has been sent to the customer.",
        "findings": [
            "Duplicate or double charges are refunded in full with no approval (kb-011).",
            "Refunds return to the original payment method in 5-10 business days "
            "(kb-011).",
        ],
        "open_questions": [
            "Whether the second charge is a true duplicate or a seat addition - the "
            "agent cannot read invoice line items.",
        ],
        "blocker": "needs_account_data",
        "next_steps": [
            "Compare the two $960 charges in Stripe and confirm both are renewals.",
            "If duplicated, refund the later charge and reply confirming 5-10 days.",
        ],
        "suggested_reply_draft": "",
        "risk_flags": ["Customer says they must explain this to their CFO."],
        "priority": "high",
        "why_escalated": (
            "The refund is above the agent's approval line and needs invoice line items "
            "the agent cannot read."
        ),
        "kb_gap": "",
    }
    data.update(overrides)
    return HandoffPacket(**data)
