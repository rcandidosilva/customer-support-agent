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
