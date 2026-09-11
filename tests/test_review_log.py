"""The verdict log: what reviewers decided, beside what the agent had predicted.

The point of these tests is not the file format. It is that the one join worth having -
the agent's score against the human's verdict - is on the same line, and that the rows
which say the *review step* is broken stay distinguishable from the rows which say the
*agent* was wrong.
"""

from __future__ import annotations

import json

from conftest import make_ticket
from langgraph.checkpoint.memory import InMemorySaver
from test_review import ESCALATES, IN_BAND

from support_agent import review_log
from support_agent.config import Settings
from support_agent.ladder import Ladder
from support_agent.llm import ScriptedLLM
from support_agent.models import ReviewVerdict


def logging_ladder(kb, settings: Settings, log, responses, **review) -> Ladder:
    return Ladder(
        ScriptedLLM(responses=dict(responses)),
        kb,
        settings.with_review(**review),
        checkpointer=InMemorySaver(),
        verdict_log=log,
    )


def test_a_verdict_is_logged_beside_the_score_it_judged(kb, settings: Settings, tmp_path):
    """The join that makes the log worth keeping, on one line."""
    log = tmp_path / "verdicts.jsonl"
    ladder = logging_ladder(kb, settings, log, IN_BAND)
    paused = ladder.run(make_ticket())
    ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    entries = list(review_log.read(log))
    assert len(entries) == 1
    entry = entries[0]

    assert entry["ticket_id"] == "TKT-TEST"
    assert entry["site"] == "draft"
    assert entry["action"] == "approve"
    assert entry["reviewer"] == "alice"
    assert entry["route"] == "send"
    assert entry["score"] == round(paused.confidence.score, 4)
    assert entry["components"] and entry["refused"] == ""


def test_a_pause_alone_logs_nothing(kb, settings: Settings, tmp_path):
    """Only a decided review is data; a pending one is just a ticket in a queue."""
    log = tmp_path / "verdicts.jsonl"
    logging_ladder(kb, settings, log, IN_BAND).run(make_ticket())
    assert list(review_log.read(log)) == []


def test_a_run_that_never_paused_logs_nothing(kb, settings: Settings, tmp_path):
    log = tmp_path / "verdicts.jsonl"
    result = Ladder(
        ScriptedLLM(responses=dict(IN_BAND)), kb, settings, verdict_log=log
    ).run(make_ticket())
    assert result.route == "escalate"
    assert list(review_log.read(log)) == []


def test_a_timeout_is_logged_as_a_broken_review_not_a_wrong_agent(
    kb, settings: Settings, tmp_path
):
    """The distinction the log exists to preserve.

    A refused row must never be read as "the human disagreed with the agent" - nobody
    looked. Counting it as disagreement would bias every threshold decision downstream.
    """
    log = tmp_path / "verdicts.jsonl"
    ladder = logging_ladder(kb, settings, log, IN_BAND)
    ladder.run(make_ticket())
    ladder.expire("TKT-TEST")

    entry = next(iter(review_log.read(log)))
    assert entry["action"] is None
    assert "no verdict arrived" in entry["refused"]

    summary = review_log.summarise(list(review_log.read(log)))
    assert summary["refused"] == 1
    assert summary["answered"] == 0
    assert summary["agreement"] == 0.0


def test_entries_accumulate_across_tickets(kb, settings: Settings, tmp_path):
    log = tmp_path / "verdicts.jsonl"
    saver = InMemorySaver()
    for ticket_id, verdict in [
        ("TKT-1", ReviewVerdict(action="approve", reviewer="alice")),
        ("TKT-2", ReviewVerdict(action="edit_and_send", reviewer="bo",
                                edited_reply="Try the SSO path instead.")),
        ("TKT-3", ReviewVerdict(action="escalate", reviewer="bo",
                                rationale="needs invoice line items")),
    ]:
        ladder = Ladder(
            ScriptedLLM(responses=dict(IN_BAND)), kb, settings.with_review(),
            checkpointer=saver, verdict_log=log,
        )
        ladder.run(make_ticket(id=ticket_id))
        ladder.resume(ticket_id, verdict)

    summary = review_log.summarise(list(review_log.read(log)))
    assert summary["answered"] == 3
    assert summary["actions"] == {"approve": 1, "edit_and_send": 1, "escalate": 1}
    assert summary["agreement"] == 1 / 3
    assert summary["reviewers"] == ["alice", "bo"]


def test_the_escalation_pause_is_logged_too(kb, settings: Settings, tmp_path):
    log = tmp_path / "verdicts.jsonl"
    ladder = logging_ladder(
        kb, settings, log, ESCALATES, floor=settings.thresholds.auto_send
    )
    ladder.run(make_ticket())
    ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    entry = next(iter(review_log.read(log)))
    assert entry["site"] == "escalation"
    assert entry["route"] == "escalate"


def test_a_truncated_line_does_not_destroy_the_history(tmp_path):
    """What a crashed writer leaves behind must not make the rest unreadable."""
    log = tmp_path / "verdicts.jsonl"
    log.write_text(
        json.dumps({"ticket_id": "TKT-1", "action": "approve", "reviewer": "a"}) + "\n"
        + '{"ticket_id": "TKT-2", "action": "appro\n'
        + json.dumps({"ticket_id": "TKT-3", "action": "approve", "reviewer": "b"}) + "\n",
        encoding="utf-8",
    )
    assert [e["ticket_id"] for e in review_log.read(log)] == ["TKT-1", "TKT-3"]


def test_reading_a_log_that_does_not_exist_is_empty_not_an_error(tmp_path):
    assert list(review_log.read(tmp_path / "nope.jsonl")) == []


def test_no_log_configured_writes_nothing(kb, settings: Settings, tmp_path):
    ladder = Ladder(
        ScriptedLLM(responses=dict(IN_BAND)), kb, settings.with_review(),
        checkpointer=InMemorySaver(),
    )
    ladder.run(make_ticket())
    ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))
    assert list(tmp_path.iterdir()) == []
