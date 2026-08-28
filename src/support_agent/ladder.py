"""The escalation ladder.

Four rungs, taken in order, each a strictly weaker claim than the one above it:

1. **Send** - we are confident, the evidence is real, and being wrong is cheap.
2. **Clarify** - we are not confident, but the missing piece is something the customer
   can hand us in one message.
3. **Escalate** - a person needs this, and we hand them a brief worth reading.
4. **Fallback packet** - the model itself is unavailable, and we still hand them
   something, assembled deterministically from whatever we managed to collect.

The rungs are a LangGraph :class:`~langgraph.graph.StateGraph`; its topology lives in
:mod:`support_agent.graph` and its node bodies in :mod:`support_agent.nodes`.  This module
is the facade over it: ``Ladder(llm, kb).run(ticket)`` takes a ticket and returns a
:class:`~support_agent.models.Resolution`, which is all most callers want.

The only way out of :meth:`Ladder.run` without a resolution is a bug.  Every stage is
wrapped: an API outage, a schema violation, and a refusal are the same event to the graph
- this rung is unavailable, take the next one down.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from .config import Settings
from .graph import build_ladder_graph
from .llm import LLMClient
from .models import Resolution, Ticket
from .nodes import (
    AGENT_PREFIX,
    LadderDeps,
    LadderState,
    clarify_rounds_used,
    fallback_classification,
    fallback_packet,
    last_agent_message,
    to_resolution,
)
from .retrieval import KnowledgeBase

__all__ = [
    "AGENT_PREFIX",
    "Ladder",
    "clarify_rounds_used",
    "fallback_classification",
    "fallback_packet",
    "last_agent_message",
    "run_ticket",
]


class Ladder:
    """Runs one ticket through the ladder and returns a :class:`Resolution`."""

    def __init__(
        self,
        llm: LLMClient,
        kb: KnowledgeBase,
        settings: Settings | None = None,
        *,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.llm = llm
        self.kb = kb
        self.settings = settings or Settings()
        self.checkpointer = checkpointer
        # Compiled once, reused for every ticket.  Compilation is not free and the
        # topology never varies per ticket.
        self.graph = build_ladder_graph(
            max_handoff_attempts=self.settings.thresholds.max_handoff_attempts,
            checkpointer=checkpointer,
        )

    def run(self, ticket: Ticket, *, thread_id: str | None = None) -> Resolution:
        """Run the ladder to completion.

        ``thread_id`` is only meaningful when the ladder was built with a checkpointer,
        in which case the run is persisted under that thread and can be resumed.  It
        defaults to the ticket id, so a ticket that comes back round after a clarifying
        question resumes its own history rather than starting a new one.
        """
        started_at = time.time()
        config: dict[str, Any] = {}
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or ticket.id}

        raw = self.graph.invoke(
            LadderState(ticket=ticket),
            context=LadderDeps(llm=self.llm, kb=self.kb, settings=self.settings),
            config=config or None,
        )
        return to_resolution(LadderState.model_validate(raw), started_at)

    def draw(self) -> str:
        """The compiled topology as Mermaid, for documentation and sanity checks."""
        return self.graph.get_graph().draw_mermaid()


def run_ticket(
    ticket: Ticket,
    llm: LLMClient,
    kb: KnowledgeBase,
    settings: Settings | None = None,
) -> Resolution:
    return Ladder(llm, kb, settings).run(ticket)
