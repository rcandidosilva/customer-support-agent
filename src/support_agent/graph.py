"""The shape of the escalation ladder, and nothing else.

Keeping the topology apart from the node bodies in :mod:`support_agent.nodes` is the main
thing the graph buys us as a piece of source code: the entire routing policy of the
application is the forty lines of :func:`build_ladder_graph` below, and every branch in it
is named after the reason it exists.

Two properties are worth reading off the diagram:

* **Every failure edge points the same way.** There is no path from a failed stage to a
  more permissive outcome - only to ``handoff``.
* **The lint loop is a real edge.** A brief that fails its quality check goes back to be
  rewritten once, rather than being shipped with a complaint recorded next to it.
* **Exactly one node can route upward, and only a human can make it.** ``human_review``
  is the sole edge from the escalation path back to ``send`` or ``clarify``, and it takes
  it only on a validated verdict from a named reviewer.  Automation still cannot promote
  itself.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    LadderDeps,
    LadderState,
    clarify_node,
    classify_node,
    critique_node,
    draft_node,
    finalise_escalation_node,
    gate_node,
    handoff_lint_node,
    handoff_node,
    human_review_node,
    policy_node,
    retrieve_node,
    route_from_review,
    send_node,
)

#: Read as: "if a stage set an escalation reason, stop climbing."
_ESCALATED = "handoff"


def _escalated_or(next_node: str):
    """Build the router every stage shares: hand off, or carry on to ``next_node``."""

    def route(state: LadderState) -> str:
        return _ESCALATED if state.escalation_reason else next_node

    route.__name__ = f"escalated_or_{next_node}"
    return route


def _route_from_gate(state: LadderState) -> str:
    return {"send": "send", "clarify": "clarify"}.get(state.route, _ESCALATED)


def _needs_rewrite(state: LadderState, max_attempts: int):
    return bool(state.lint_feedback) and state.handoff_attempts < max_attempts


def build_ladder_graph(
    *,
    max_handoff_attempts: int = 2,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile the ladder.

    ``max_handoff_attempts`` bounds the lint rewrite loop.  It is passed in rather than
    read from the state because the topology has to be fixed at compile time, and the
    graph is compiled once per :class:`~support_agent.ladder.Ladder`.
    """
    graph = StateGraph(LadderState, context_schema=LadderDeps)

    for name, node in (
        ("classify", classify_node),
        ("retrieve", retrieve_node),
        ("policy", policy_node),
        ("draft", draft_node),
        ("critique", critique_node),
        ("gate", gate_node),
        ("send", send_node),
        ("clarify", clarify_node),
        ("handoff", handoff_node),
        ("handoff_lint", handoff_lint_node),
        ("human_review", human_review_node),
        ("finalise_escalation", finalise_escalation_node),
    ):
        graph.add_node(name, node)

    graph.add_edge(START, "classify")
    # Retrieval runs unconditionally - it is local, it cannot fail the way a model call
    # can, and evidence is worth having in the packet even when triage did not work.
    graph.add_edge("classify", "retrieve")

    for source, following in (
        ("retrieve", "policy"),      # triage failed
        ("policy", "draft"),         # a policy rule forced a handoff
        ("draft", "critique"),       # the drafter could not produce anything
        ("critique", "gate"),        # nothing reviewed the draft
    ):
        graph.add_conditional_edges(
            source, _escalated_or(following), {_ESCALATED: _ESCALATED, following: following}
        )

    graph.add_conditional_edges(
        "gate",
        _route_from_gate,
        {"send": "send", "clarify": "clarify", _ESCALATED: _ESCALATED},
    )

    graph.add_edge("send", END)
    # A clarifying question that could not be written falls through to a person rather
    # than releasing the unverified draft.
    graph.add_conditional_edges(
        "clarify", _escalated_or(END), {_ESCALATED: _ESCALATED, END: END}
    )

    graph.add_edge("handoff", "handoff_lint")
    graph.add_conditional_edges(
        "handoff_lint",
        lambda state: "handoff" if _needs_rewrite(state, max_handoff_attempts)
        else "human_review",
        {"handoff": "handoff", "human_review": "human_review"},
    )
    # The only node that can climb back *up* the ladder, and only ever because a named
    # human said so.  With review disabled it is a pass-through to finalise_escalation,
    # and so are a refused verdict and an expired deadline - the three ways this can go
    # wrong all land on the behaviour the ladder had before the node existed.
    graph.add_conditional_edges(
        "human_review",
        route_from_review,
        {
            "send": "send",
            "clarify": "clarify",
            "finalise_escalation": "finalise_escalation",
        },
    )
    graph.add_edge("finalise_escalation", END)

    return graph.compile(checkpointer=checkpointer)
