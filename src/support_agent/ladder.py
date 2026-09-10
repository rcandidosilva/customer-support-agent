"""The escalation ladder.

Four rungs, taken in order, each a strictly weaker claim than the one above it:

1. **Send** - we are confident, the evidence is real, and being wrong is cheap.
2. **Clarify** - we are not confident, but the missing piece is something the customer
   can hand us in one message.
3. **Escalate** - a person needs this, and we hand them a brief worth reading.
4. **Fallback packet** - the model itself is unavailable, and we still hand them
   something, assembled deterministically from whatever we managed to collect.

Optionally, rung 3 pauses on a named human before the brief reaches the queue
(``Settings.review.enabled``).  That reviewer is the only party in the system who can
move a ticket back *up* the ladder - approving the brief, sending an edited reply, or
turning the escalation into a question.  Everything that can go wrong with the pause -
an unusable verdict, one asking for something this pause does not offer, or none at all
before the deadline - lands on the behaviour the ladder had without it: the brief goes to
the queue.  See :meth:`Ladder.resume`.

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
from langgraph.types import Command
from pydantic import ValidationError

from .config import Settings
from .graph import build_ladder_graph
from .llm import LLMClient
from .models import Resolution, ReviewRequest, ReviewVerdict, Ticket
from .nodes import (
    AGENT_PREFIX,
    REVIEW_EXPIRED,
    LadderDeps,
    LadderState,
    clarify_rounds_used,
    fallback_classification,
    fallback_packet,
    last_agent_message,
    to_paused_resolution,
    to_resolution,
)
from .retrieval import KnowledgeBase

__all__ = [
    "AGENT_PREFIX",
    "Ladder",
    "NoReviewPending",
    "clarify_rounds_used",
    "fallback_classification",
    "fallback_packet",
    "last_agent_message",
    "run_ticket",
]


class NoReviewPending(LookupError):
    """Raised when a thread is asked for a verdict it is not waiting on.

    Distinct from a refused verdict: that is a routing outcome the ladder handles, while
    this is the caller being wrong about which thread is paused.
    """


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
        if self.settings.review.enabled and checkpointer is None:
            # Caught here rather than mid-run on purpose.  A pause is only a pause if it
            # can be resumed, and a run that suspends into a checkpointer that does not
            # exist has silently dropped a customer's ticket.
            raise ValueError(
                "human review needs a checkpointer: build the ladder with "
                "Ladder(..., checkpointer=SqliteSaver(...)) or disable "
                "Settings.review.enabled"
            )
        # Compiled once, reused for every ticket.  Compilation is not free and the
        # topology never varies per ticket.
        self.graph = build_ladder_graph(
            max_handoff_attempts=self.settings.thresholds.max_handoff_attempts,
            checkpointer=checkpointer,
        )

    def run(self, ticket: Ticket, *, thread_id: str | None = None) -> Resolution:
        """Run the ladder to completion, or to a human.

        Still returns a :class:`Resolution` - which is the point.  When human review is
        enabled and this ticket escalates, the run suspends and the resolution comes back
        with ``route == "review"`` and :attr:`Resolution.pending_review` set: a
        non-terminal resolution rather than a different return type.  Callers that leave
        review off can never see one, so their contract is unchanged.

        ``thread_id`` is only meaningful when the ladder was built with a checkpointer,
        in which case the run is persisted under that thread and can be resumed.  It
        defaults to the ticket id, so a ticket that comes back round after a clarifying
        question resumes its own history rather than starting a new one.
        """
        started_at = time.time()
        raw = self.graph.invoke(
            LadderState(ticket=ticket, started_at=started_at),
            context=LadderDeps(llm=self.llm, kb=self.kb, settings=self.settings),
            config=self._config(thread_id or ticket.id),
        )
        return self._resolve(raw, started_at)

    # -- human review ------------------------------------------------------------------

    def pending_review(self, thread_id: str) -> ReviewRequest | None:
        """The request a paused thread is waiting on, or ``None`` if it is not paused.

        Read from the checkpoint, so it survives the process that started the run - which
        is the only reason a review step is usable at all.  It is also the *authoritative*
        copy of the deadline: the review node rebuilds an equivalent request when it
        re-executes on resume, with resume-time clocks.
        """
        snapshot = self.graph.get_state(self._config(thread_id))
        for pending in snapshot.interrupts:
            try:
                return ReviewRequest.model_validate(pending.value)
            except ValidationError:
                continue
        return None

    def paused_threads(self) -> list[ReviewRequest]:
        """Every thread currently waiting on a verdict, discovered from the store.

        Enumerated rather than remembered, so a sweeper does not need to have been the
        process that opened the pause - which is the whole point of persisting it.
        """
        if self.checkpointer is None:
            return []

        pending: list[ReviewRequest] = []
        for thread_id in self._thread_ids():
            request = self.pending_review(thread_id)
            if request is not None:
                pending.append(request)
        return pending

    def expire_overdue(self, *, now: float | None = None) -> list[Resolution]:
        """Release every review whose deadline has passed.

        The deadline on its own only bites when somebody happens to call
        :meth:`resume`; without this it is enforced on contact rather than by the clock,
        and a thread nobody touches waits forever - exactly the failure the deadline
        exists to prevent.  Run it on a timer.

        Returns the resolutions it finalised, so a caller can log or alert on them.
        """
        now = time.time() if now is None else now
        return [
            self.expire(request.thread_id)
            for request in self.paused_threads()
            if request.expired(now)
        ]

    def resume(self, thread_id: str, verdict: ReviewVerdict) -> Resolution:
        """Deliver a verdict and run the ticket to its real outcome.

        The deadline is enforced here rather than in the node, because only the persisted
        request knows when the pause actually began.  A verdict that arrives late is not
        applied: the run finalises as the escalation it already was.
        """
        request = self._require_pending(thread_id)
        envelope: dict[str, Any] = {"request": request.model_dump(mode="json")}
        if request.expired():
            envelope[REVIEW_EXPIRED] = True
        else:
            envelope["verdict"] = verdict.model_dump(mode="json")
        return self._resume_with(thread_id, envelope)

    def expire(self, thread_id: str) -> Resolution:
        """Give up waiting and let the brief go to the queue.

        What a sweeper calls on every thread whose deadline has passed.  Separated from
        :meth:`resume` because "nobody answered" is a different event from "somebody
        answered too late", and both need to be visible in the trace.
        """
        request = self._require_pending(thread_id)
        return self._resume_with(
            thread_id,
            {"request": request.model_dump(mode="json"), REVIEW_EXPIRED: True},
        )

    # -- internals ---------------------------------------------------------------------

    def _thread_ids(self) -> list[str]:
        """Distinct thread ids in the store, newest checkpoint first.

        A store holds many checkpoints per thread, so this de-duplicates rather than
        asking the graph about the same ticket ten times.
        """
        seen: dict[str, None] = {}
        for checkpoint in self.checkpointer.list(None):
            thread_id = (checkpoint.config.get("configurable") or {}).get("thread_id")
            if thread_id is not None:
                seen.setdefault(thread_id, None)
        return list(seen)

    def _config(self, thread_id: str | None) -> dict[str, Any] | None:
        if self.checkpointer is None:
            return None
        return {"configurable": {"thread_id": thread_id}}

    def _require_pending(self, thread_id: str) -> ReviewRequest:
        request = self.pending_review(thread_id)
        if request is None:
            raise NoReviewPending(f"no review is pending on thread {thread_id!r}")
        return request

    def _resume_with(self, thread_id: str, envelope: dict[str, Any]) -> Resolution:
        raw = self.graph.invoke(
            Command(resume=envelope),
            context=LadderDeps(llm=self.llm, kb=self.kb, settings=self.settings),
            config=self._config(thread_id),
        )
        return self._resolve(raw, time.time())

    def _resolve(self, raw: dict[str, Any], started_at: float) -> Resolution:
        state = LadderState.model_validate(raw)
        # Prefer the run's own start over this leg's, so a resumed ticket reports the
        # latency a customer actually experienced.
        started_at = state.started_at or started_at

        for pending in raw.get("__interrupt__") or ():
            try:
                request = ReviewRequest.model_validate(pending.value)
            except ValidationError:
                continue
            return to_paused_resolution(state, started_at, request)
        return to_resolution(state, started_at)

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
