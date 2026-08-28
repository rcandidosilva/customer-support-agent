"""The bundled tickets are documentation. They should keep saying what they claim to."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_classification

from support_agent.models import Ticket
from support_agent.policy import check

TICKET_DIR = Path(__file__).resolve().parents[1] / "examples" / "tickets"
EXAMPLES = sorted(TICKET_DIR.glob("*.json"))


def load(name: str) -> Ticket:
    path = next(p for p in EXAMPLES if p.name.startswith(name))
    return Ticket.model_validate(json.loads(path.read_text()))


def test_there_are_examples():
    assert len(EXAMPLES) >= 8


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_every_example_parses(path: Path):
    ticket = Ticket.model_validate(json.loads(path.read_text()))
    assert ticket.id and ticket.subject and ticket.body


def test_duplicate_charge_example_trips_the_money_rule(settings):
    hit = check(load("TKT-1003"), make_classification(category="billing"), settings)
    assert hit and hit.rule == "money_above_agent_authority"


def test_churn_example_trips_the_churn_rule(settings):
    hit = check(load("TKT-1004"), make_classification(), settings)
    assert hit and hit.rule == "churn_risk"


def test_simple_examples_pass_the_policy_gate(settings):
    """These are the ones the ladder is supposed to actually deflect."""
    for name in ("TKT-1001", "TKT-1007"):
        assert check(load(name), make_classification(), settings) is None


def test_the_followup_example_carries_the_earlier_agent_message(settings):
    from support_agent.ladder import clarify_rounds_used, last_agent_message

    ticket = load("TKT-1008")
    assert clarify_rounds_used(ticket) == 1
    assert "freshness badge" in last_agent_message(ticket)


@pytest.mark.parametrize(
    "name,expected_article",
    [
        ("TKT-1001", "kb-001"),
        ("TKT-1002", "kb-021"),
        ("TKT-1003", "kb-011"),
        ("TKT-1006", "kb-031"),
        ("TKT-1007", "kb-030"),
    ],
)
def test_retrieval_reaches_the_right_article_from_raw_ticket_text(
    kb, name: str, expected_article: str
):
    """Retrieval on the raw ticket, before any model has rewritten the query."""
    ticket = load(name)
    hits = kb.search(f"{ticket.subject} {ticket.body}", top_k=5)
    assert expected_article in {h.article_id for h in hits}


def test_the_kb_gap_example_finds_nothing_strong(kb):
    """Embedding dashboards in Salesforce is genuinely not in the knowledge base."""
    ticket = load("TKT-1005")
    hits = kb.search(f"{ticket.subject} {ticket.body}", top_k=5)
    assert not hits or hits[0].score < 0.30
