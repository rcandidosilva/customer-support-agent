"""Tunable knobs for the deflection ladder.

Every threshold in this file is a product decision disguised as a number.  They are
gathered here so that changing the deflection rate is a config review, not a code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

MODEL = "claude-opus-5"

KB_DIR = Path(__file__).parent / "kb"


@dataclass(frozen=True)
class StageConfig:
    """Per-stage model settings.

    Effort is the lever, not the model.  Triage and clarification are shallow tasks;
    critique and the handoff packet are where thinking actually buys something.
    """

    effort: str = "medium"
    max_tokens: int = 4096


@dataclass(frozen=True)
class Thresholds:
    """Where the ladder's rungs sit."""

    #: At or above this fused confidence, the reply goes out without a human.
    auto_send: float = 0.78
    #: Below ``auto_send`` but at or above this, we are allowed to ask the customer one
    #: question rather than escalating - but only if the reviewer says the missing piece
    #: is something the customer can actually supply.
    clarify_floor: float = 0.45
    #: Hard floor on the safety component.  A reply that could cost the customer money
    #: or data is never sent automatically, however confident the model is about it.
    min_action_safety: float = 0.70
    #: Hard floor on groundedness, for the same reason.
    min_groundedness: float = 0.70
    #: Hard floor on coverage.  A reply that answers half the question is not a
    #: deflection - the customer writes back, and now there are two tickets.
    min_coverage: float = 0.60
    #: A retrieval that returns nothing this good is treated as "no evidence".
    min_retrieval_score: float = 0.12
    #: How many times the handoff brief may be rewritten when it fails its own lint.
    #: One rewrite: the second failure is a prompt problem, not a sampling problem, and
    #: burning calls on it hides that.
    max_handoff_attempts: int = 2
    #: How many clarifying questions we are willing to ask before handing to a human.
    #: One.  Two rounds of automated questions is how a ticket becomes a complaint.
    max_clarify_rounds: int = 1


@dataclass(frozen=True)
class ReviewConfig:
    """Human-in-the-loop review of escalations.

    Off by default, and deliberately a config question rather than a topology one: the
    review node is always in the graph and is a pass-through when this is disabled, so
    there is one shape of ladder to reason about rather than two.

    Enabling it requires a checkpointer - a pause that cannot be persisted is a pause
    that loses the ticket - and :class:`~support_agent.ladder.Ladder` refuses to build
    without one.
    """

    enabled: bool = False
    #: The review band.  At or above this but below ``auto_send``, a draft is held for a
    #: human to approve or edit rather than being escalated with a brief.
    #:
    #: ``auto_send`` is a cliff: a draft scoring 0.74 with clean citations becomes a full
    #: escalation, and a person writes a reply from scratch.  That is the most expensive
    #: possible outcome for a near miss.  This band converts it into the cheapest human
    #: action there is - reading one reply and saying yes.
    #:
    #: Set to ``auto_send`` or above to switch the band off and review only escalations.
    floor: float = 0.62
    #: How long a brief may sit unreviewed.  On expiry the run finalises as an ordinary
    #: escalation: the packet reaches the queue as it would have without review at all.
    #: The deadline exists because the failure mode of a review step is silence, and
    #: silence must not be able to hold a ticket forever.
    sla_minutes: int = 30

    @property
    def sla_seconds(self) -> float:
        return self.sla_minutes * 60.0


@dataclass(frozen=True)
class PolicyConfig:
    """Non-negotiable escalation triggers, evaluated before confidence is even read."""

    #: Refund or credit asks above this amount always see a human.
    refund_approval_usd: float = 200.0
    #: Accounts above this MRR never get a fully automated reply on a high-severity
    #: ticket.  Not because the agent is worse at them - because the blast radius is.
    high_touch_mrr_usd: float = 2000.0
    #: Words that mean the ticket has left the support domain.
    legal_terms: tuple[str, ...] = (
        "lawyer", "attorney", "legal action", "sue", "lawsuit", "litigation",
        "subpoena", "gdpr request", "data subject", "dsar", "right to erasure",
        "breach", "regulator", "ico ", "dpa ",
    )
    churn_terms: tuple[str, ...] = (
        "cancel our contract", "cancel my contract", "not renewing", "won't renew",
        "will not renew", "switching to", "moving to a competitor", "churn",
        "terminate our agreement", "escalate to my account manager",
    )
    security_terms: tuple[str, ...] = (
        "vulnerability", "exploit", "compromised", "unauthorized access",
        "data leak", "leaked", "penetration test", "cve-",
    )
    #: Public-pressure signals.  Not a reason to give in; a reason a human should read it.
    exposure_terms: tuple[str, ...] = (
        "twitter", "linkedin post", "hacker news", "trustpilot", "g2 review",
        "our ceo", "your ceo", "press",
    )


@dataclass(frozen=True)
class Settings:
    model: str = MODEL
    kb_dir: Path = KB_DIR
    thresholds: Thresholds = field(default_factory=Thresholds)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    stages: dict[str, StageConfig] = field(
        default_factory=lambda: {
            "classify": StageConfig(effort="low", max_tokens=2048),
            "draft": StageConfig(effort="medium", max_tokens=6000),
            "critique": StageConfig(effort="high", max_tokens=4096),
            "clarify": StageConfig(effort="low", max_tokens=2048),
            "handoff": StageConfig(effort="high", max_tokens=8000),
        }
    )
    #: How many KB passages to put in front of the drafter.
    top_k: int = 5

    def stage(self, name: str) -> StageConfig:
        return self.stages.get(name, StageConfig())

    def with_model(self, model: str) -> Settings:
        return replace(self, model=model)

    def with_review(
        self,
        *,
        enabled: bool = True,
        sla_minutes: int | None = None,
        floor: float | None = None,
    ) -> Settings:
        current = self.review
        return replace(
            self,
            review=ReviewConfig(
                enabled=enabled,
                floor=current.floor if floor is None else floor,
                sla_minutes=current.sla_minutes if sla_minutes is None else sla_minutes,
            ),
        )

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls()
        if model := os.getenv("SUPPORT_AGENT_MODEL"):
            settings = settings.with_model(model)
        if os.getenv("SUPPORT_AGENT_REVIEW", "").strip().lower() in ("1", "true", "yes"):
            settings = settings.with_review(
                floor=_env_float("SUPPORT_AGENT_REVIEW_FLOOR", settings.review.floor),
                sla_minutes=_env_int(
                    "SUPPORT_AGENT_REVIEW_SLA", settings.review.sla_minutes
                ),
            )
        return settings


def _env_float(name: str, fallback: float) -> float:
    """A malformed override falls back rather than crashing the process.

    These are operational knobs read at startup, and a typo in a deployment variable
    should not be able to take the review queue down.
    """
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return fallback


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return fallback
