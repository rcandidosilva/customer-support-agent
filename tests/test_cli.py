from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_packet

from support_agent.cli import main


def test_search_finds_articles(capsys):
    assert main(["search", "refund an annual plan"]) == 0
    out = capsys.readouterr().out
    assert "kb-011" in out


def test_search_reports_no_matches(capsys):
    assert main(["search", "zzzz qqqq wwww"]) == 1
    assert "no matches" in capsys.readouterr().out


def test_lint_accepts_a_clean_packet(tmp_path: Path, capsys):
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(make_packet().model_dump(mode="json")))
    assert main(["lint", str(path)]) == 0
    assert "clean" in capsys.readouterr().out


def test_lint_rejects_a_customer_voiced_packet(tmp_path: Path, capsys):
    path = tmp_path / "packet.json"
    packet = make_packet(summary="Thanks for reaching out about your billing issue!")
    path.write_text(json.dumps(packet.model_dump(mode="json")))
    assert main(["lint", str(path)]) == 1
    assert "customer-facing phrasing" in capsys.readouterr().out


def test_lint_unwraps_a_resolution(tmp_path: Path, capsys):
    path = tmp_path / "resolution.json"
    path.write_text(
        json.dumps({"ticket_id": "T", "packet": make_packet().model_dump(mode="json")})
    )
    assert main(["lint", str(path), "--render"]) == 0
    assert "# [Billing/High]" in capsys.readouterr().out


def test_run_reports_a_missing_ticket_file():
    with pytest.raises(SystemExit):
        main(["run", "no-such-file.json"])

# -- the review queue -------------------------------------------------------------------
#
# Driven through `main()` against a real SQLite store. The operator surface is the part
# most likely to break silently: it is the only code path a person uses by hand, and it
# went untested until it crashed in a terminal.


def _seed_queue(tmp_path, ticket_id="TKT-501", **review):
    from conftest import make_ticket
    from test_review import IN_BAND

    from support_agent import KnowledgeBase, Ladder, ScriptedLLM, Settings, sqlite_saver

    db = str(tmp_path / "reviews.db")
    settings = Settings.from_env().with_review(**review)
    kb = KnowledgeBase.from_dir(settings.kb_dir)
    with sqlite_saver(db) as saver:
        Ladder(
            ScriptedLLM(responses=dict(IN_BAND)), kb, settings, checkpointer=saver
        ).run(make_ticket(id=ticket_id))
    return db


@pytest.fixture
def queue(tmp_path: Path, monkeypatch):
    """A store with one banded ticket paused, plus the argv prefix to talk to it."""
    monkeypatch.setenv("SUPPORT_AGENT_REVIEW", "true")
    db = _seed_queue(tmp_path)
    return ["review", "--db", db, "--log", str(tmp_path / "verdicts.jsonl")]


def test_review_lists_the_queue(queue, capsys):
    assert main([*queue, "--list"]) == 0
    out = capsys.readouterr().out
    assert "TKT-501" in out and "draft" in out


def test_review_shows_what_is_being_asked(queue, capsys):
    assert main([*queue, "--show", "TKT-501", "--no-trace"]) == 0
    out = capsys.readouterr().out
    assert "REVIEW" in out
    assert "`approve`" in out, "the reviewer needs to know what they may do"


def test_review_approves_and_sends(queue, capsys):
    assert main([*queue, "--approve", "TKT-501", "--as", "alice", "--no-trace"]) == 0
    out = capsys.readouterr().out
    assert "SEND" in out and "alice" in out

    assert main([*queue, "--list"]) == 0
    assert "empty" in capsys.readouterr().out


def test_review_records_the_verdict_in_the_log(queue, capsys):
    main([*queue, "--approve", "TKT-501", "--as", "alice", "--no-trace"])
    capsys.readouterr()

    assert main([*queue, "--summary"]) == 0
    out = capsys.readouterr().out
    assert "agreement" in out and "100%" in out


def test_review_rejects_an_incomplete_verdict(queue, capsys):
    """`--edit-and-send` with nothing to send is a mistake worth catching at the door."""
    assert main([*queue, "--edit-and-send", "TKT-501", "--as", "bo"]) == 2
    assert "edited_reply" in capsys.readouterr().err


def test_review_reports_an_unknown_ticket(queue, capsys):
    assert main([*queue, "--approve", "TKT-NOPE", "--as", "bo"]) == 1
    assert "no review is pending" in capsys.readouterr().err


def test_review_needs_a_verb(queue, capsys):
    assert main(queue) == 2
    assert "nothing to do" in capsys.readouterr().err


def test_review_escalates_without_credentials(queue, capsys):
    """The bug the CLI found: rejecting a draft writes a brief, which needs the model.

    With no client available it must degrade into the deterministically assembled packet
    the ladder already falls back to - not raise.
    """
    assert main([*queue, "--escalate", "TKT-501", "--as", "bo",
                 "--why", "needs invoice line items", "--no-trace"]) == 0
    out = capsys.readouterr().out
    assert "ESCALATE" in out
    assert "needs invoice line items" in out


def test_review_sweeps_what_is_overdue(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("SUPPORT_AGENT_REVIEW", "true")
    monkeypatch.setenv("SUPPORT_AGENT_REVIEW_SLA", "0")
    from support_agent.config import Settings

    assert Settings.from_env().review.sla_minutes == 0, "the env override must reach us"

    db = _seed_queue(tmp_path, ticket_id="TKT-502", sla_minutes=0)
    argv = ["review", "--db", db, "--log", str(tmp_path / "v.jsonl"), "--sweep"]
    assert main(argv) == 0
    assert "TKT-502" in capsys.readouterr().out
