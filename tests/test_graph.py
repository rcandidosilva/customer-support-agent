"""Behaviour the graph adds over the imperative ladder it replaced.

The routing itself is covered by ``test_ladder.py``, which was written before the port
and passes unchanged. What is tested here is what the port bought: the lint rewrite loop,
the shape of the failure edges, and durability.
"""

from __future__ import annotations

from conftest import (
    make_classification,
    make_critique,
    make_draft,
    make_packet,
    make_ticket,
)

from support_agent.config import Settings
from support_agent.graph import build_ladder_graph
from support_agent.ladder import Ladder
from support_agent.llm import LLMError, ScriptedLLM

# A brief that fails its own lint: customer voice, and a step that is not a step.
BAD_PACKET = {
    "summary": "Thanks for reaching out about your duplicate charge!",
    "next_steps": ["Investigate further."],
}


def escalating_ticket():
    """A ticket the policy gate sends straight to a handoff."""
    return make_ticket(body="Our attorney has been in touch about the duplicate charge.")


# -- the lint rewrite loop ---------------------------------------------------------------


def test_a_failing_brief_is_rewritten_once(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "handoff": [make_packet(**BAD_PACKET), make_packet()],
        }
    )
    result = Ladder(llm, kb, settings).run(escalating_ticket())

    assert llm.stages_called() == ["classify", "handoff", "handoff"]
    # The packet that survives is the clean rewrite, not the first attempt.
    assert result.packet.summary == make_packet().summary
    assert result.route == "escalate"

    lint_rows = [s for s in result.trace if s.stage == "handoff_lint"]
    assert len(lint_rows) == 2
    assert not lint_rows[0].ok and "rewriting" in lint_rows[0].detail
    assert lint_rows[1].detail == "clean"


def test_the_rewrite_is_told_exactly_what_was_wrong(kb, settings: Settings):
    """Naming the defects beats repeating the style rules it already ignored once."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "handoff": [make_packet(**BAD_PACKET), make_packet()],
        }
    )
    Ladder(llm, kb, settings).run(escalating_ticket())

    first, second = (c for c in llm.calls if c.stage == "handoff")
    assert "rejected" not in first.user
    assert "rejected" in second.user
    assert "customer-facing phrasing" in second.user
    assert "not an actionable step" in second.user


def test_the_rewrite_loop_is_bounded(kb, settings: Settings):
    """Two failures is a prompt problem. Ship it with the complaint on the trace."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "handoff": [make_packet(**BAD_PACKET), make_packet(**BAD_PACKET)],
        }
    )
    result = Ladder(llm, kb, settings).run(escalating_ticket())

    assert llm.stages_called().count("handoff") == 2
    assert result.packet is not None  # never lose the ticket
    assert not [s for s in result.trace if s.stage == "handoff_lint"][-1].ok


def test_warnings_alone_do_not_burn_a_second_call(kb, settings: Settings):
    """A slightly unscannable subject line is not worth another model call."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "handoff": make_packet(subject_line="Customer has a billing problem"),
        }
    )
    result = Ladder(llm, kb, settings).run(escalating_ticket())

    assert llm.stages_called().count("handoff") == 1
    lint = [s for s in result.trace if s.stage == "handoff_lint"][-1]
    assert lint.ok  # warnings only
    assert "subject_line" in lint.detail


def test_the_auto_assembled_packet_is_not_rewritten(kb, settings: Settings):
    """It passes lint by construction, and the model is unavailable anyway."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "handoff": LLMError("handoff", "api error 500: internal"),
        }
    )
    result = Ladder(llm, kb, settings).run(escalating_ticket())

    assert llm.stages_called().count("handoff") == 1
    assert result.degraded
    assert "degraded handoff" in result.packet.subject_line


# -- topology ----------------------------------------------------------------------------


def _edges() -> dict[str, set[str]]:
    graph = build_ladder_graph().get_graph()
    edges: dict[str, set[str]] = {}
    for edge in graph.edges:
        edges.setdefault(edge.source, set()).add(edge.target)
    return edges


def test_every_failure_edge_points_at_the_handoff(settings: Settings):
    """The structural guarantee: degradation can only ever route more conservatively.

    Asserted against the compiled graph rather than by reading the source, so adding a
    stage that quietly falls through to `send` fails here.
    """
    edges = _edges()

    for stage in ("retrieve", "policy", "draft", "critique", "clarify"):
        targets = edges[stage]
        assert "handoff" in targets, f"{stage} has no escalation edge"
        assert "send" not in targets, f"{stage} can reach send directly"

    # `send` is reachable from the gate, which is automation deciding it is confident,
    # and from the two review nodes, each of which is a named person deciding. Nothing
    # else may reach it, and no *automated* stage may - both review nodes take that edge
    # only on a validated verdict (see test_review.py).
    assert {s for s, t in edges.items() if "send" in t} == {
        "gate", "human_review", "draft_review",
    }
    # And the lint node is the only thing that can send work back for a rewrite.
    assert "handoff" in edges["handoff_lint"]


def test_review_can_always_fall_back_to_the_queue(settings: Settings):
    """The new node's fail-closed edge, asserted structurally rather than trusted.

    A review step's characteristic failure is silence. If `human_review` could not reach
    `finalise_escalation`, a ticket nobody looked at would have nowhere to go.
    """
    edges = _edges()
    assert "finalise_escalation" in edges["human_review"]
    # Review sits after the lint loop, so a reviewer reads the brief that would actually
    # have been queued rather than a draft of it.
    assert edges["handoff_lint"] == {"handoff", "human_review"}
    # The band's fallback is an ordinary escalation: a draft nobody looked at goes where
    # the gate would have sent it without the band.
    assert "handoff" in edges["draft_review"]
    # And the band is only reachable through the gate - nothing routes into it sideways.
    assert {s for s, t in edges.items() if "draft_review" in t} == {"gate"}


def test_the_graph_renders(kb, settings: Settings):
    mermaid = Ladder(ScriptedLLM(), kb, settings).draw()
    assert "handoff_lint" in mermaid and "gate" in mermaid


# -- durability --------------------------------------------------------------------------


def test_a_run_can_be_checkpointed(kb, settings: Settings):
    """State is serialisable end to end, which is what makes resume possible."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    ladder = Ladder(llm, kb, settings, checkpointer=saver)
    result = ladder.run(make_ticket())
    assert result.route == "send"

    saved = ladder.graph.get_state({"configurable": {"thread_id": "TKT-TEST"}})
    assert saved.values["route"] == "send"
    assert [s.stage for s in saved.values["trace"]][0] == "classify"


def test_threads_default_to_the_ticket_id(kb, settings: Settings):
    """So a ticket coming back after a clarifying question resumes its own history."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    ladder = Ladder(llm, kb, settings, checkpointer=saver)
    ladder.run(make_ticket(id="TKT-9001"))

    saved = ladder.graph.get_state({"configurable": {"thread_id": "TKT-9001"}})
    assert saved.values["ticket"].id == "TKT-9001"


def test_no_checkpointer_needs_no_thread(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    assert Ladder(llm, kb, settings).run(make_ticket()).route == "send"
