"""The store a paused review lives in.

Human review is the first feature here that cannot work without persistence, so the store
gets its own tests: that our models survive a round trip, that LangGraph will still read
them back once its allowlist stops being advisory, and that a pause really does outlive
the process that opened it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from conftest import make_ticket

from support_agent.checkpointing import checkpoint_types, serializer, sqlite_saver
from support_agent.config import Settings
from support_agent.models import ReviewVerdict

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip(
    "langgraph.checkpoint.sqlite",
    reason='needs the optional extra: pip install "support-agent[sqlite]"',
)


# -- the allowlist ----------------------------------------------------------------------


def test_the_allowlist_is_derived_not_typed_out():
    """A hand-maintained list rots, and its failure mode is an unreadable queue."""
    types = checkpoint_types()
    names = {name for _, name in types}

    expected = {"Ticket", "HandoffPacket", "ReviewRequest", "ReviewVerdict", "ReviewRecord"}
    assert expected <= names
    assert {module for module, _ in types} == {
        "support_agent.models",
        "support_agent.nodes",
    }


def test_our_models_survive_a_round_trip():
    serde = serializer()
    ticket = make_ticket(id="TKT-RT")
    restored = serde.loads_typed(serde.dumps_typed(ticket))

    assert restored == ticket
    assert type(restored) is type(ticket)


SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


def test_the_allowlist_covers_the_graph_state_itself(
    kb, settings: Settings, caplog, tmp_path
):
    """`LadderState` lives in `nodes`, not `models`, and is the easiest one to miss.

    Missing it does not degrade gracefully. An *empty* allowlist only warns, but once an
    explicit one exists an omission is a hard block - so a half-complete allowlist is
    worse than none, and that is what this pins.
    """
    assert ("support_agent.nodes", "LadderState") in checkpoint_types()

    db = str(tmp_path / "state.db")
    _open_review_in_a_separate_process(db)

    from support_agent.ladder import Ladder
    from support_agent.llm import ScriptedLLM

    with caplog.at_level("WARNING", logger=SERDE_LOGGER), sqlite_saver(db) as saver:
        resumer = Ladder(ScriptedLLM(), kb, settings.with_review(), checkpointer=saver)
        resumer.pending_review("TKT-DURABLE")
        resumer.resume("TKT-DURABLE", ReviewVerdict(action="approve", reviewer="alice"))

    # Any complaint at all, not one particular wording: "unregistered type" and "Blocked
    # deserialization" are different messages for the same latent breakage, and grepping
    # for only the first is how the LadderState omission survived its first test.
    complaints = [r.getMessage() for r in caplog.records if r.name == SERDE_LOGGER]
    assert not complaints, f"serde complained: {complaints}"


# -- durability -------------------------------------------------------------------------

#: Opens a review and exits, so the resume below happens in a process that has never seen
#: the ticket.  Two ladders sharing an object would prove far less.
_OPENER = """
import sys
sys.path.insert(0, {root!r} + "/src")
sys.path.insert(0, {root!r} + "/tests")
from conftest import (
    make_classification, make_critique, make_draft, make_packet, make_ticket,
)
from support_agent import KnowledgeBase, Ladder, ScriptedLLM, Settings
from support_agent.checkpointing import sqlite_saver

settings = Settings().with_review(sla_minutes=30)
kb = KnowledgeBase.from_dir(settings.kb_dir)
responses = {{
    "classify": make_classification(),
    "draft": make_draft(),
    "critique": make_critique(
        groundedness=0.3, coverage=0.4, action_safety=0.5,
        blocked_on_customer_input=False, problems=["Amount not in evidence."],
    ),
    "handoff": make_packet(),
}}
with sqlite_saver({db!r}) as saver:
    result = Ladder(ScriptedLLM(responses=responses), kb, settings,
                    checkpointer=saver).run(make_ticket(id="TKT-DURABLE"))
    print(result.route)
"""


def _open_review_in_a_separate_process(db: str) -> None:
    opened = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_OPENER).format(root=str(ROOT), db=db)],
        capture_output=True, text=True, timeout=120,
    )
    assert opened.returncode == 0, opened.stderr
    assert opened.stdout.strip() == "review"


def test_a_review_outlives_the_process_that_opened_it(kb, settings: Settings, tmp_path):
    """The claim the whole feature rests on, tested across a real process boundary."""
    db = str(tmp_path / "reviews.db")
    _open_review_in_a_separate_process(db)

    from support_agent.ladder import Ladder
    from support_agent.llm import ScriptedLLM

    with sqlite_saver(db) as saver:
        resumer = Ladder(ScriptedLLM(), kb, settings.with_review(), checkpointer=saver)

        request = resumer.pending_review("TKT-DURABLE")
        assert request is not None, "the pause did not survive the process"
        assert request.ticket_id == "TKT-DURABLE"

        result = resumer.resume(
            "TKT-DURABLE", ReviewVerdict(action="approve", reviewer="alice")
        )

    assert result.route == "escalate"
    assert result.review.accepted
    # The brief was rebuilt from disk, not from anything this process computed.
    assert result.packet is not None
    assert result.packet.subject_line
    assert result.confidence is not None


def test_a_sweeper_can_find_another_process_work(kb, settings: Settings, tmp_path):
    """The sweeper's actual deployment shape: a timer, in a process of its own."""
    db = str(tmp_path / "reviews.db")
    _open_review_in_a_separate_process(db)

    from support_agent.ladder import Ladder
    from support_agent.llm import ScriptedLLM

    with sqlite_saver(db) as saver:
        sweeper = Ladder(ScriptedLLM(), kb, settings.with_review(), checkpointer=saver)
        assert [r.thread_id for r in sweeper.paused_threads()] == ["TKT-DURABLE"]

        # Nothing is overdue yet, so a sweep must leave it alone.
        assert sweeper.expire_overdue() == []
        # Far enough in the future that the 30-minute window has passed.
        released = sweeper.expire_overdue(now=time.time() + 3600)

    assert [r.ticket_id for r in released] == ["TKT-DURABLE"]
    assert released[0].route == "escalate"
    assert "no verdict arrived" in released[0].review.refused
