from __future__ import annotations

from conftest import (
    make_classification,
    make_critique,
    make_draft,
    make_packet,
    make_ticket,
)

from support_agent.ladder import Ladder
from support_agent.llm import ScriptedLLM
from support_agent.render import render_packet, render_resolution, render_trace


def test_packet_renders_in_reading_order():
    md = render_packet(make_packet(), ticket_id="TKT-1003")
    assert md.startswith("# [Billing/High]")
    assert "TKT-1003" in md
    for heading in (
        "## Already said to the customer",
        "## What the customer wants",
        "## What the agent already did",
        "## Findings",
        "## Open questions",
        "## Next steps",
        "## Risk flags",
    ):
        assert heading in md
    # What was already said comes before the work, so a colleague cannot miss it.
    assert md.index("## Already said to the customer") < md.index("## Next steps")


def test_blocker_is_rendered_in_english():
    md = render_packet(make_packet(blocker="needs_account_data"))
    assert "Needs account data the agent cannot read" in md


def test_draft_reply_is_labelled_unverified():
    md = render_packet(make_packet(suggested_reply_draft="We have refunded the charge."))
    assert "unverified" in md.lower()


def test_empty_draft_reply_section_is_omitted():
    assert "Suggested reply" not in render_packet(make_packet(suggested_reply_draft=""))


def test_resolution_renders_a_sent_reply(kb, settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    md = render_resolution(Ladder(llm, kb, settings).run(make_ticket()))
    assert "→ **SEND**" in md
    assert "### Reply sent" in md
    assert "**Confidence" in md
    assert "### Trace" in md


def test_resolution_renders_a_forced_escalation(kb, settings):
    llm = ScriptedLLM(
        responses={"classify": make_classification(), "handoff": make_packet()}
    )
    result = Ladder(llm, kb, settings).run(
        make_ticket(body="Our attorney has been in touch.")
    )
    md = render_resolution(result)
    assert "→ **ESCALATE** (forced by policy)" in md
    assert "`legal_or_privacy`" in md


def test_trace_table_has_a_total_row(kb, settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    table = render_trace(Ladder(llm, kb, settings).run(make_ticket()))
    assert table.splitlines()[0].startswith("| stage |")
    assert "**total**" in table
