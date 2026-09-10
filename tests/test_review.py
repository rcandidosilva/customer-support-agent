"""Human-in-the-loop review of escalations.

The pause is the only place in this pipeline where an outside party can move a ticket
*up* the ladder, so these tests are mostly about the ways that can go wrong: a verdict
that does not parse, one that asks for something this pause does not offer, and one that
never arrives at all. All three must land on the behaviour the ladder had before the
review node existed - the brief goes to the queue.
"""

from __future__ import annotations

import time

import pytest
from conftest import (
    make_classification,
    make_critique,
    make_draft,
    make_packet,
    make_ticket,
)
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.config import Settings
from support_agent.ladder import Ladder, NoReviewPending
from support_agent.llm import ScriptedLLM
from support_agent.models import ReviewVerdict

#: A ticket the ladder escalates: the reviewer says the blocker is not something the
#: customer can supply, which puts it below the clarify floor and onto the handoff path.
ESCALATES = {
    "classify": make_classification(),
    "draft": make_draft(),
    "critique": make_critique(
        groundedness=0.3, coverage=0.4, action_safety=0.5,
        blocked_on_customer_input=False,
        problems=["The refund amount is not in evidence."],
    ),
    "handoff": make_packet(),
}


def reviewing(kb, settings: Settings, *, sla_minutes: int = 30, **responses) -> Ladder:
    return Ladder(
        ScriptedLLM(responses=responses or dict(ESCALATES)),
        kb,
        settings.with_review(sla_minutes=sla_minutes),
        checkpointer=InMemorySaver(),
    )


# -- the contract -----------------------------------------------------------------------


def test_review_off_by_default_changes_nothing(kb, settings: Settings):
    """The whole safety argument for putting the node in the graph unconditionally."""
    result = Ladder(ScriptedLLM(responses=dict(ESCALATES)), kb, settings).run(make_ticket())
    assert result.route == "escalate"
    assert result.review is None
    assert not result.pending_review


def test_enabling_review_without_a_checkpointer_is_refused(kb, settings: Settings):
    """A pause that cannot be persisted is a dropped ticket, so it fails at build time."""
    with pytest.raises(ValueError, match="checkpointer"):
        Ladder(ScriptedLLM(), kb, settings.with_review())


def test_a_paused_run_still_returns_a_resolution(kb, settings: Settings):
    """`run()` keeps its return type. The pause is a route, not a second shape."""
    result = reviewing(kb, settings).run(make_ticket())

    assert result.route == "review"
    assert result.pending_review
    assert result.review is not None and result.review.verdict is None
    # Everything the reviewer needs to decide is already on the resolution.
    assert result.packet is not None
    assert result.retrieved
    assert result.confidence is not None


def test_the_pending_request_is_readable_from_the_checkpoint(kb, settings: Settings):
    """What makes the step usable: the pause outlives the process that opened it."""
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket(id="TKT-77"))

    request = ladder.pending_review("TKT-77")
    assert request is not None
    assert request.ticket_id == "TKT-77" and request.thread_id == "TKT-77"
    assert request.expires_at > request.requested_at
    # `escalate` is not offered: the run is already escalating, so `approve` means that.
    assert request.allowed_actions == ["approve", "edit_and_send", "ask_customer"]


def test_a_pause_survives_the_object_that_opened_it(kb, settings: Settings):
    """A review that only one process can answer is not a review step.

    Two independently constructed ladders sharing a store stand in for a restart: the
    second has never seen this ticket and resumes it purely from the checkpoint. With a
    durable saver (`SqliteSaver`, `PostgresSaver`) the same code spans real restarts;
    `InMemorySaver` is what keeps this test offline and fast.
    """
    saver = InMemorySaver()
    reviewed = settings.with_review()

    opener = Ladder(
        ScriptedLLM(responses=dict(ESCALATES)), kb, reviewed, checkpointer=saver
    )
    assert opener.run(make_ticket(id="TKT-55")).route == "review"

    # A different process: new ladder, new scripted client, no memory of the run.
    resumer = Ladder(ScriptedLLM(), kb, reviewed, checkpointer=saver)
    assert resumer.pending_review("TKT-55") is not None

    result = resumer.resume("TKT-55", ReviewVerdict(action="approve", reviewer="alice"))
    assert result.route == "escalate"
    assert result.review.accepted
    assert result.packet is not None, "the brief came back from the checkpoint, not memory"


def test_resuming_a_thread_that_is_not_paused_is_an_error(kb, settings: Settings):
    ladder = reviewing(
        kb, settings,
        classify=make_classification(), draft=make_draft(), critique=make_critique(),
    )
    ladder.run(make_ticket())  # sends; never pauses
    with pytest.raises(NoReviewPending):
        ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))


# -- the four verdicts ------------------------------------------------------------------


def test_approve_releases_the_brief_to_the_queue(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    assert result.route == "escalate"
    assert result.packet is not None
    assert result.review.accepted
    assert result.decision.rule == "human_review"
    assert "alice" in result.decision.reason


def test_a_reviewer_can_send_an_edited_reply_instead(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="edit_and_send", reviewer="bo",
                      edited_reply="Refunded the second charge; it lands in 5-10 days."),
    )

    assert result.route == "send"
    assert result.customer_reply.startswith("Refunded the second charge")
    assert result.decision.rule == "human_review"
    # The brief stays on the record even though nobody had to work it.
    assert result.packet is not None


def test_a_reviewer_can_turn_an_escalation_into_a_question(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="ask_customer", reviewer="bo",
                      edited_reply="Which of the two charges do you want refunded?"),
    )

    assert result.route == "clarify"
    assert result.customer_reply == "Which of the two charges do you want refunded?"
    # The model was not asked to rewrite the question the human already wrote.
    assert "clarify" not in ladder.llm.stages_called()


def test_overriding_the_escalation_clears_the_stop_flag(kb, settings: Settings):
    """Otherwise the clarify branch reads the stale reason and bounces to handoff."""
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="ask_customer", reviewer="bo", edited_reply="Which charge?"),
    )
    assert result.route == "clarify"


# -- the ways it fails ------------------------------------------------------------------


def test_an_unusable_verdict_is_refused_and_the_brief_is_queued(kb, settings: Settings):
    """The resume value is the least-typed input in the system. It gets a schema."""
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    # Bypasses ReviewVerdict's own validation the way a bad caller or a bad UI would.
    result = ladder._resume_with("TKT-TEST", {"verdict": {"action": "edit_and_send",
                                                          "reviewer": "bo"}})

    assert result.route == "escalate"
    assert not result.review.accepted
    assert "edited_reply" in result.review.refused
    assert [t for t in result.trace if t.stage == "human_review" and not t.ok]


def test_an_action_this_pause_does_not_offer_is_refused(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    request = ladder.pending_review("TKT-TEST")
    result = ladder._resume_with(
        "TKT-TEST",
        {
            "request": request.model_dump(mode="json"),
            "verdict": ReviewVerdict(action="escalate", reviewer="bo",
                                     rationale="the brief misses the seat change"
                                     ).model_dump(mode="json"),
        },
    )

    assert result.route == "escalate"
    assert not result.review.accepted
    assert "not offered" in result.review.refused
    # The verdict is kept on the record even though it was not applied.
    assert result.review.verdict.reviewer == "bo"


def test_a_refused_verdict_cannot_reach_the_customer(kb, settings: Settings):
    """The refusal has to gate the *route*, not merely the record.

    Today every customer-facing action is offered at this pause, so a refused verdict
    happens to be harmless. That stops being true the moment a pause site offers a
    narrower set - so the check is pinned here rather than left to be rediscovered.
    """
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    request = ladder.pending_review("TKT-TEST")
    narrowed = request.model_dump(mode="json") | {"allowed_actions": ["approve"]}

    result = ladder._resume_with(
        "TKT-TEST",
        {
            "request": narrowed,
            "verdict": ReviewVerdict(action="edit_and_send", reviewer="bo",
                                     edited_reply="Refunded, sorry about that."
                                     ).model_dump(mode="json"),
        },
    )

    assert result.route == "escalate"
    assert result.customer_reply == ""
    assert not result.review.accepted


def test_garbage_on_the_wire_is_refused(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder._resume_with("TKT-TEST", {"verdict": "looks fine to me"})

    assert result.route == "escalate"
    assert not result.review.accepted


# -- the deadline -----------------------------------------------------------------------


def test_a_review_nobody_answers_falls_back_to_the_queue(kb, settings: Settings):
    """The failure mode of a review step is silence, so silence has to have a route."""
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.expire("TKT-TEST")

    assert result.route == "escalate"
    assert result.packet is not None
    assert not result.review.accepted
    assert "no verdict arrived" in result.review.refused


def test_a_late_verdict_is_not_applied(kb, settings: Settings):
    """Fail closed: past the deadline, even a well-formed approval does not land."""
    ladder = reviewing(kb, settings, sla_minutes=0)
    ladder.run(make_ticket())
    time.sleep(0.01)
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="edit_and_send", reviewer="bo", edited_reply="All sorted."),
    )

    assert result.route == "escalate", "an expired review must never reach the customer"
    assert result.customer_reply == ""
    assert not result.review.accepted


def test_the_deadline_is_the_persisted_one_not_a_resume_time_one(kb, settings: Settings):
    """The node re-runs its body on resume, so in-node clocks would restart the SLA.

    Guards the subtle bug: if the deadline were recomputed when the node re-executes, a
    verdict could never be late and the timeout would silently do nothing.
    """
    ladder = reviewing(kb, settings, sla_minutes=0)
    ladder.run(make_ticket())
    request = ladder.pending_review("TKT-TEST")
    assert request.expired()

    time.sleep(0.01)
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))
    assert not result.review.accepted


# -- the review band --------------------------------------------------------------------

#: Fuses to ~0.76 - under the 0.78 send threshold, over the 0.62 band floor. Good enough
#: for a person to judge the reply directly, not good enough to send unread.
IN_BAND = {
    "classify": make_classification(),
    "draft": make_draft(),
    "critique": make_critique(
        groundedness=0.72, coverage=0.66, action_safety=0.75,
        blocked_on_customer_input=False,
        problems=["The reply does not mention the SSO case."],
    ),
    "handoff": make_packet(),
}


def banded(kb, settings: Settings, **review) -> Ladder:
    return Ladder(
        ScriptedLLM(responses=dict(IN_BAND)),
        kb,
        settings.with_review(**review),
        checkpointer=InMemorySaver(),
    )


def test_a_near_miss_pauses_instead_of_escalating(kb, settings: Settings):
    """The economic point of the band: a person reads one reply, not a whole brief."""
    ladder = banded(kb, settings)
    paused = ladder.run(make_ticket())

    assert paused.route == "review"
    assert paused.review.request.site == "draft"
    assert paused.draft is not None
    assert settings.review.floor <= paused.confidence.score < settings.thresholds.auto_send
    # No brief was written: nobody has to be handed one yet.
    assert paused.packet is None
    assert "handoff" not in ladder.llm.stages_called()


def test_the_band_offers_the_action_the_escalation_pause_cannot(kb, settings: Settings):
    """`escalate` finally means something: the run is not escalating yet."""
    ladder = banded(kb, settings)
    request = ladder.run(make_ticket()).review.request
    assert request.allowed_actions == [
        "approve", "edit_and_send", "ask_customer", "escalate",
    ]


def test_approving_a_draft_sends_the_agents_own_reply(kb, settings: Settings):
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    assert result.route == "send"
    assert result.customer_reply == make_draft().reply
    assert result.decision.rule == "human_review"


def test_a_reviewed_send_does_not_claim_the_confidence_threshold(kb, settings: Settings):
    """The score is *below* auto_send - that is why a person was asked.

    Quoting the threshold here would put a false statement in the audit trail of every
    banded send.
    """
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    assert "auto-send threshold" not in result.decision.reason
    assert "alice" in result.decision.reason


def test_a_reviewer_can_reject_a_draft_to_a_person(kb, settings: Settings):
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="escalate", reviewer="bo",
                      rationale="needs the invoice line items"),
    )

    assert result.route == "escalate"
    assert result.customer_reply == ""
    assert result.packet is not None, "rejecting to a person still writes them a brief"


def test_a_rejected_draft_does_not_pause_a_second_time(kb, settings: Settings):
    """One review per ticket.

    Without this the reviewer who escalated a draft is asked to review the brief their
    own decision produced - which is both absurd and a way to strand a ticket behind two
    deadlines.
    """
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume(
        "TKT-TEST",
        ReviewVerdict(action="escalate", reviewer="bo", rationale="needs invoice lines"),
    )

    assert not result.pending_review
    assert ladder.paused_threads() == []
    assert [t.stage for t in result.trace].count("draft_review") == 1
    assert "human_review" not in [t.stage for t in result.trace]


def test_an_unanswered_draft_becomes_an_ordinary_escalation(kb, settings: Settings):
    """The band's fallback is the route the gate would have taken without it."""
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    result = ladder.expire("TKT-TEST")

    assert result.route == "escalate"
    assert result.packet is not None
    assert not result.review.accepted
    # And the escalation pause does not then fire on the way out.
    assert not result.pending_review


def test_the_band_is_off_when_review_is_off(kb, settings: Settings):
    """Scores in the band escalate exactly as they did before."""
    result = Ladder(ScriptedLLM(responses=dict(IN_BAND)), kb, settings).run(make_ticket())
    assert result.route == "escalate"
    assert result.review is None


def test_a_high_floor_switches_the_band_off(kb, settings: Settings):
    """The documented way to review escalations only."""
    ladder = banded(kb, settings, floor=settings.thresholds.auto_send)
    paused = ladder.run(make_ticket())

    assert paused.route == "review"
    assert paused.review.request.site == "escalation", "should be the brief, not the draft"


def test_a_confident_answer_still_never_pauses(kb, settings: Settings):
    """The band must not drag the happy path in front of a person."""
    ladder = Ladder(
        ScriptedLLM(responses={"classify": make_classification(), "draft": make_draft(),
                               "critique": make_critique()}),
        kb, settings.with_review(), checkpointer=InMemorySaver(),
    )
    result = ladder.run(make_ticket())

    assert result.route == "send"
    assert result.review is None


# -- the sweeper ------------------------------------------------------------------------


def _paused(kb, settings: Settings, saver, *ids: str, sla_minutes: int = 30) -> Ladder:
    reviewed = settings.with_review(sla_minutes=sla_minutes)
    for ticket_id in ids:
        Ladder(
            ScriptedLLM(responses=dict(ESCALATES)), kb, reviewed, checkpointer=saver
        ).run(make_ticket(id=ticket_id))
    return Ladder(ScriptedLLM(), kb, reviewed, checkpointer=saver)


def test_paused_threads_are_discoverable_without_knowing_their_ids(kb, settings: Settings):
    """A sweeper cannot be handed the ticket ids; it has to find them in the store."""
    sweeper = _paused(kb, settings, InMemorySaver(), "TKT-A", "TKT-B", "TKT-C")
    assert sorted(r.thread_id for r in sweeper.paused_threads()) == [
        "TKT-A", "TKT-B", "TKT-C"
    ]


def test_the_sweeper_releases_only_what_is_overdue(kb, settings: Settings):
    """Without this the deadline binds on contact, not on the clock.

    A thread nobody calls `resume` on would otherwise wait forever, which is the exact
    failure the SLA exists to prevent.
    """
    saver = InMemorySaver()
    sweeper = _paused(kb, settings, saver, "TKT-OLD", sla_minutes=0)
    _paused(kb, settings, saver, "TKT-NEW", sla_minutes=30)
    time.sleep(0.01)

    released = sweeper.expire_overdue()

    assert [r.ticket_id for r in released] == ["TKT-OLD"]
    assert released[0].route == "escalate"
    assert released[0].packet is not None
    # The one still inside its window is untouched and can still be answered.
    assert [r.thread_id for r in sweeper.paused_threads()] == ["TKT-NEW"]


def test_the_sweeper_is_idempotent(kb, settings: Settings):
    """It runs on a timer, so a second pass must not double-finalise anything."""
    sweeper = _paused(kb, settings, InMemorySaver(), "TKT-OLD", sla_minutes=0)
    time.sleep(0.01)

    assert len(sweeper.expire_overdue()) == 1
    assert sweeper.expire_overdue() == []
    assert sweeper.paused_threads() == []


def test_the_sweeper_ignores_threads_that_never_paused(kb, settings: Settings):
    saver = InMemorySaver()
    reviewed = settings.with_review()
    Ladder(
        ScriptedLLM(responses={"classify": make_classification(), "draft": make_draft(),
                               "critique": make_critique()}),
        kb, reviewed, checkpointer=saver,
    ).run(make_ticket(id="TKT-SENT"))

    sweeper = Ladder(ScriptedLLM(), kb, reviewed, checkpointer=saver)
    assert sweeper.paused_threads() == []
    assert sweeper.expire_overdue() == []


def test_a_ladder_with_no_store_has_nothing_to_sweep(kb, settings: Settings):
    ladder = Ladder(ScriptedLLM(), kb, settings)
    assert ladder.paused_threads() == []
    assert ladder.expire_overdue() == []
    assert ladder.stalled_threads() == []


# -- stalled runs -----------------------------------------------------------------------


def test_a_crash_after_a_verdict_leaves_a_thread_nothing_can_see(kb, settings: Settings):
    """A run that stopped partway has no interrupt, so no queue and no sweep finds it.

    Found by running the CLI, not by reasoning about it: rejecting a draft writes a
    brief, the model call for it failed, and the ticket vanished from every operational
    surface at once - the pause already spent, the run unfinished.
    """
    saver = InMemorySaver()
    # No `handoff` response, so writing the brief raises after the verdict is applied.
    ladder = Ladder(
        ScriptedLLM(responses={k: v for k, v in IN_BAND.items() if k != "handoff"}),
        kb, settings.with_review(), checkpointer=saver,
    )
    ladder.run(make_ticket())

    with pytest.raises(AssertionError, match="no response for stage 'handoff'"):
        ladder.resume(
            "TKT-TEST",
            ReviewVerdict(action="escalate", reviewer="bo", rationale="needs invoices"),
        )

    assert ladder.paused_threads() == [], "the pause was spent"
    assert ladder.expire_overdue() == [], "and no deadline will ever fire"
    assert ladder.stalled_threads() == ["TKT-TEST"], "so it needs its own query"


def test_a_stalled_run_can_be_driven_to_completion(kb, settings: Settings):
    """Recovery re-enters at the failed node rather than re-running the ticket."""
    saver = InMemorySaver()
    scripted = ScriptedLLM(
        responses={k: v for k, v in IN_BAND.items() if k != "handoff"}
    )
    ladder = Ladder(scripted, kb, settings.with_review(), checkpointer=saver)
    ladder.run(make_ticket())
    with pytest.raises(AssertionError):
        ladder.resume(
            "TKT-TEST",
            ReviewVerdict(action="escalate", reviewer="bo", rationale="needs invoices"),
        )

    # A second ladder with the brief available, standing in for the fixed deployment.
    recovered = Ladder(
        ScriptedLLM(responses={"handoff": make_packet()}),
        kb, settings.with_review(), checkpointer=saver,
    ).retry("TKT-TEST")

    assert recovered.route == "escalate"
    assert recovered.packet is not None
    assert recovered.review.verdict.reviewer == "bo", "the verdict survived the crash"
    # Re-entered at the failed node: nothing before it ran a second time.
    assert [t.stage for t in recovered.trace].count("classify") == 1


def test_a_finished_run_is_not_stalled(kb, settings: Settings):
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))
    assert ladder.stalled_threads() == []


def test_a_paused_run_is_not_stalled(kb, settings: Settings):
    """The distinction the query rests on: waiting on a person is not being stuck."""
    ladder = banded(kb, settings)
    ladder.run(make_ticket())
    assert ladder.paused_threads() and ladder.stalled_threads() == []


# -- the audit trail --------------------------------------------------------------------


def test_the_verdict_names_who_decided(kb, settings: Settings):
    """An anonymous approval is not an audit record."""
    with pytest.raises(ValueError, match="named reviewer"):
        ReviewVerdict(action="approve", reviewer="   ")


def test_an_override_has_to_say_why(kb, settings: Settings):
    with pytest.raises(ValueError, match="rationale"):
        ReviewVerdict(action="escalate", reviewer="bo")


def test_the_review_shows_up_on_the_trace(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    row = next(t for t in result.trace if t.stage == "human_review")
    assert row.ok and "approve by alice" in row.detail


def test_the_record_keeps_the_request_that_was_actually_made(kb, settings: Settings):
    """The node rebuilds an equivalent request on resume; the record must not use it.

    If it did, every review would look like it was answered instantly and the recorded
    wait would be fiction.
    """
    ladder = reviewing(kb, settings)
    ladder.run(make_ticket())
    opened = ladder.pending_review("TKT-TEST")

    time.sleep(0.02)
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))

    recorded = result.review.request
    assert recorded.requested_at == pytest.approx(opened.requested_at, abs=1e-6)
    assert recorded.expires_at == pytest.approx(opened.expires_at, abs=1e-6)


def test_latency_covers_the_whole_ticket_not_just_the_resume(kb, settings: Settings):
    ladder = reviewing(kb, settings)
    paused = ladder.run(make_ticket())
    time.sleep(0.01)
    result = ladder.resume("TKT-TEST", ReviewVerdict(action="approve", reviewer="alice"))
    assert result.started_at == pytest.approx(paused.started_at, abs=1e-6)
