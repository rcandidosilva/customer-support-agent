"""End-to-end routing, with every model call scripted. No network, no API key."""

from __future__ import annotations

from conftest import (
    make_clarify,
    make_classification,
    make_critique,
    make_draft,
    make_packet,
    make_ticket,
)

from support_agent.config import Settings
from support_agent.ladder import Ladder, clarify_rounds_used, last_agent_message
from support_agent.llm import LLMError, ScriptedLLM


def ladder(kb, settings, **responses) -> Ladder:
    return Ladder(ScriptedLLM(responses=responses), kb, settings)


# -- the happy path ---------------------------------------------------------------------


def test_confident_answer_is_sent(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "send"
    assert result.customer_reply == make_draft().reply
    assert result.packet is None
    assert not result.degraded
    assert result.confidence.score >= settings.thresholds.auto_send
    assert llm.stages_called() == ["classify", "draft", "critique"]


def test_send_records_a_readable_decision(kb, settings: Settings):
    result = ladder(
        kb, settings,
        classify=make_classification(), draft=make_draft(), critique=make_critique(),
    ).run(make_ticket())
    assert "auto-send threshold" in result.decision.reason
    assert not result.decision.forced_by_policy


# -- the middle rung --------------------------------------------------------------------


def test_uncertain_but_customer_answerable_asks_a_question(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(
                information_gaps=["whether the workspace enforces SSO"],
                best_clarifying_question="Does your workspace sign in through Okta?",
            ),
            "critique": make_critique(
                groundedness=0.75, coverage=0.5, action_safety=0.9,
                blocked_on_customer_input=True,
                problems=["The reply guesses which of two causes applies."],
            ),
            "clarify": make_clarify(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "clarify"
    assert result.customer_reply == make_clarify().reply
    assert result.packet is None
    assert "customer can supply" in result.decision.reason


def test_uncertain_but_not_customer_answerable_escalates(kb, settings: Settings):
    """The distinction that keeps the bot from interrogating people pointlessly."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(
                groundedness=0.5, coverage=0.5, action_safety=0.9,
                blocked_on_customer_input=False,
                blocker="needs_account_data",
            ),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert "not something the customer can supply" in result.decision.reason
    assert "clarify" not in llm.stages_called()


def test_one_clarifying_question_is_the_limit(kb, settings: Settings):
    """A second round of questions is how a ticket becomes a complaint."""
    ticket = make_ticket(
        history=[
            "CUSTOMER: my dashboard is stale",
            "AGENT: Which connector is behind, and what does its freshness badge say?",
        ]
    )
    assert clarify_rounds_used(ticket) == 1

    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(
                groundedness=0.7, coverage=0.6, action_safety=0.9,
                blocked_on_customer_input=True,
            ),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(ticket)

    assert result.route == "escalate"
    assert "already been asked" in result.decision.reason
    assert "clarify" not in llm.stages_called()


def test_confidence_below_the_clarify_floor_escalates(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(citations=["kb-999"]),  # caps the score at 0.25
            "critique": make_critique(blocked_on_customer_input=True),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert "floor for even asking" in result.decision.reason


# -- the policy gate --------------------------------------------------------------------


def test_policy_escalation_skips_the_expensive_stages(kb, settings: Settings):
    """The gate exists to save tokens as well as to be safe."""
    llm = ScriptedLLM(
        responses={"classify": make_classification(), "handoff": make_packet()}
    )
    ticket = make_ticket(body="Our lawyer has been in touch about this.")
    result = Ladder(llm, kb, settings).run(ticket)

    assert result.route == "escalate"
    assert result.decision.forced_by_policy
    assert result.decision.rule == "legal_or_privacy"
    assert llm.stages_called() == ["classify", "handoff"]
    assert result.draft is None and result.confidence is None


def test_policy_escalation_still_carries_evidence(kb, settings: Settings):
    """Retrieval is local, so the human gets KB context even on the forced path."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(
                category="billing", search_terms=["refund", "duplicate charge"]
            ),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(
        make_ticket(body="We were double charged $960 for the annual renewal.")
    )
    assert result.route == "escalate"
    assert result.retrieved
    assert "kb-011" in {c.article_id for c in result.retrieved}


# -- graceful degradation ---------------------------------------------------------------


def test_triage_failure_never_auto_sends(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": LLMError("classify", "api error 503: overloaded", retryable=True),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert result.degraded
    assert result.decision.rule == "degraded_triage"
    assert "draft" not in llm.stages_called()
    # Retrieval still ran, so the human is not starting from nothing.
    assert result.retrieved


def test_draft_failure_escalates_with_no_draft(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": LLMError("draft", "model declined (cyber)"),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert result.degraded
    assert result.decision.rule == "degraded_draft"
    assert result.draft is None


def test_unreviewable_draft_is_never_sent(kb, settings: Settings):
    """We have an answer and nothing checked it. That is what a review queue is for."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": LLMError("critique", "connection error"),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert result.degraded
    assert result.decision.rule == "degraded_critique"
    assert result.draft is not None  # the human still gets to see it
    assert result.confidence is None


def test_clarify_failure_falls_through_to_escalation(kb, settings: Settings):
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(
                groundedness=0.75, coverage=0.5, action_safety=0.9,
                blocked_on_customer_input=True,
            ),
            "clarify": LLMError("clarify", "rate limited", retryable=True),
            "handoff": make_packet(),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert result.degraded
    assert result.packet is not None


def test_handoff_failure_still_produces_a_packet(kb, settings: Settings):
    """The floor of the ladder: never lose the ticket."""
    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(groundedness=0.3, action_safety=0.4),
            "handoff": LLMError("handoff", "api error 500: internal"),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())

    assert result.route == "escalate"
    assert result.degraded
    assert result.packet is not None
    assert "degraded handoff" in result.packet.subject_line
    assert result.packet.next_steps  # the auto-assembled packet still acts
    assert result.packet.already_told_customer


def test_the_fallback_packet_passes_its_own_lint(kb, settings: Settings):
    from support_agent.handoff_lint import has_errors, lint_packet

    llm = ScriptedLLM(
        responses={
            "classify": make_classification(),
            "draft": make_draft(),
            "critique": make_critique(groundedness=0.3, action_safety=0.4),
            "handoff": LLMError("handoff", "api error 500"),
        }
    )
    result = Ladder(llm, kb, settings).run(make_ticket())
    assert not has_errors(lint_packet(result.packet))


# -- continuity across turns ------------------------------------------------------------


def test_the_packet_is_told_what_we_already_said(kb, settings: Settings):
    """The highest-consequence field: a colleague must not contradict us."""
    ticket = make_ticket(
        history=[
            "CUSTOMER: my dashboard is stale",
            "AGENT: Which connector is behind, and what does its freshness badge say?",
        ]
    )
    assert last_agent_message(ticket).startswith("Which connector")

    llm = ScriptedLLM(
        responses={
            "classify": make_classification(requires_account_data=True),
            "handoff": make_packet(),
        }
    )
    Ladder(llm, kb, settings).run(ticket)

    handoff_call = next(c for c in llm.calls if c.stage == "handoff")
    assert "Which connector is behind" in handoff_call.user
    assert "Already sent to the customer" in handoff_call.user


# -- the audit trail --------------------------------------------------------------------


def test_every_run_leaves_a_trace(kb, settings: Settings):
    result = ladder(
        kb, settings,
        classify=make_classification(), draft=make_draft(), critique=make_critique(),
    ).run(make_ticket())

    stages = [s.stage for s in result.trace]
    assert stages == ["classify", "retrieve", "policy", "draft", "critique", "score",
                      "route"]
    tin, tout = result.total_tokens
    assert tin > 0 and tout > 0


def test_failures_are_visible_in_the_trace(kb, settings: Settings):
    result = ladder(
        kb, settings,
        classify=LLMError("classify", "api error 503: overloaded"),
        handoff=make_packet(),
    ).run(make_ticket())

    failed = [s for s in result.trace if not s.ok]
    assert failed and failed[0].stage == "classify"
    assert "503" in failed[0].detail
    assert any(s.degraded for s in result.trace)


def test_lint_findings_reach_the_trace(kb, settings: Settings):
    result = ladder(
        kb, settings,
        classify=make_classification(requires_account_data=True),
        handoff=make_packet(next_steps=["Investigate further."]),
    ).run(make_ticket())

    lint = next(s for s in result.trace if s.stage == "handoff_lint")
    assert not lint.ok
    assert "actionable" in lint.detail


def test_resolution_serialises(kb, settings: Settings):
    result = ladder(
        kb, settings,
        classify=make_classification(), draft=make_draft(), critique=make_critique(),
    ).run(make_ticket())
    blob = result.model_dump(mode="json")
    assert blob["route"] == "send"
    assert blob["trace"]
