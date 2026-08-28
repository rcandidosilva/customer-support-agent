from __future__ import annotations

import pytest
from conftest import make_classification, make_ticket

from support_agent.config import Settings
from support_agent.policy import check, largest_amount


@pytest.mark.parametrize(
    "text,expected",
    [
        ("we were charged $960 twice", 960.0),
        ("charged $1,240.50 on the 3rd", 1240.5),
        ("that's 300 USD we didn't authorise", 300.0),
        ("£75 and then €120", 120.0),
        ("no money mentioned here at all", 0.0),
    ],
)
def test_largest_amount(text: str, expected: float):
    assert largest_amount(text) == expected


def test_legal_language_escalates(settings: Settings):
    ticket = make_ticket(body="I've forwarded this to our attorney.")
    hit = check(ticket, make_classification(), settings)
    assert hit and hit.rule == "legal_or_privacy"
    assert hit.blocker == "risk_flag"


def test_security_language_escalates(settings: Settings):
    ticket = make_ticket(body="We think an account was compromised last night.")
    hit = check(ticket, make_classification(), settings)
    assert hit and hit.rule == "security_report"


def test_pasted_secrets_escalate(settings: Settings):
    hit = check(
        make_ticket(),
        make_classification(pii_present=["full API token"]),
        settings,
    )
    assert hit and hit.rule == "sensitive_data_in_ticket"


def test_churn_language_escalates(settings: Settings):
    ticket = make_ticket(body="At this rate we will not renew in November.")
    hit = check(ticket, make_classification(), settings)
    assert hit and hit.rule == "churn_risk"
    assert hit.blocker == "needs_human_judgment"


def test_money_above_approval_line_escalates(settings: Settings):
    ticket = make_ticket(body="We were double charged $960 for the renewal.")
    hit = check(ticket, make_classification(category="billing"), settings)
    assert hit and hit.rule == "money_above_agent_authority"


def test_small_billing_amount_does_not_escalate(settings: Settings):
    ticket = make_ticket(body="I was charged $12 more than I expected this month.")
    assert check(ticket, make_classification(category="billing"), settings) is None


def test_money_outside_billing_category_is_not_a_billing_rule(settings: Settings):
    """A $5,000 figure in a capacity-planning question is not a refund request."""
    ticket = make_ticket(body="We process about 5000 USD of orders an hour through this.")
    assert check(ticket, make_classification(category="integrations"), settings) is None


def test_exception_request_escalates(settings: Settings):
    hit = check(make_ticket(), make_classification(requests_exception=True), settings)
    assert hit and hit.rule == "policy_exception_requested"
    assert hit.blocker == "needs_policy_exception"


def test_account_data_requirement_escalates(settings: Settings):
    hit = check(make_ticket(), make_classification(requires_account_data=True), settings)
    assert hit and hit.rule == "needs_account_data"


def test_high_touch_account_escalates_only_when_severe(settings: Settings):
    ticket = make_ticket(customer={"mrr_usd": 9400, "plan": "enterprise"})
    assert check(ticket, make_classification(severity="urgent"), settings) is not None
    assert check(ticket, make_classification(severity="low"), settings) is None


def test_repeat_escalations_escalate(settings: Settings):
    ticket = make_ticket(customer={"prior_escalations_90d": 3})
    hit = check(ticket, make_classification(), settings)
    assert hit and hit.rule == "repeat_escalation"


def test_ordinary_ticket_passes_the_gate(settings: Settings):
    assert check(make_ticket(), make_classification(), settings) is None


def test_rules_are_ordered_most_serious_first(settings: Settings):
    """A ticket that trips several rules reports the one a human would lead with."""
    ticket = make_ticket(
        body="Our lawyer is involved and we will not renew. Refund the $960.",
    )
    hit = check(ticket, make_classification(category="billing"), settings)
    assert hit and hit.rule == "legal_or_privacy"


def test_history_is_searched_too(settings: Settings):
    ticket = make_ticket(
        body="Any update?",
        history=["CUSTOMER: I've asked our attorney to look at this."],
    )
    hit = check(ticket, make_classification(), settings)
    assert hit and hit.rule == "legal_or_privacy"
