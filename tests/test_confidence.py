"""The fusion step is where model judgement stops being the only vote."""

from __future__ import annotations

from conftest import make_critique, make_draft

from support_agent.config import Settings
from support_agent.models import RetrievedChunk
from support_agent.stages.critic import score_confidence


def chunks(*ids_and_scores) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(article_id=i, title=f"{i} title", score=s, text="passage text")
        for i, s in ids_and_scores
    ]


def test_clean_answer_scores_above_the_send_threshold(kb, settings: Settings):
    report = score_confidence(
        make_critique(), make_draft(), chunks(("kb-001", 0.7), ("kb-002", 0.2)), kb,
        settings,
    )
    assert report.score >= settings.thresholds.auto_send
    assert report.penalties == []
    assert report.invalid_citations == []


def test_fabricated_citation_caps_the_score(kb, settings: Settings):
    """A citation retrieval never returned means the draft is not reading the evidence."""
    report = score_confidence(
        make_critique(),
        make_draft(citations=["kb-001", "kb-777"]),
        chunks(("kb-001", 0.7)),
        kb,
        settings,
    )
    assert report.invalid_citations == ["kb-777"]
    assert report.score <= 0.25
    assert any("unverifiable" in p for p in report.penalties)


def test_citing_an_article_retrieval_did_not_surface_is_also_invalid(kb, settings):
    """kb-011 exists, but the drafter was never shown it - so it cannot have used it."""
    report = score_confidence(
        make_critique(), make_draft(citations=["kb-011"]), chunks(("kb-001", 0.7)),
        kb, settings,
    )
    assert report.invalid_citations == ["kb-011"]
    assert report.score <= 0.25


def test_low_groundedness_blocks_auto_send_however_high_the_rest(kb, settings: Settings):
    report = score_confidence(
        make_critique(groundedness=0.4, coverage=1.0, action_safety=1.0),
        make_draft(),
        chunks(("kb-001", 0.9)),
        kb,
        settings,
    )
    assert report.score < settings.thresholds.auto_send
    assert any("groundedness" in p for p in report.penalties)


def test_low_action_safety_blocks_auto_send(kb, settings: Settings):
    report = score_confidence(
        make_critique(groundedness=1.0, coverage=1.0, action_safety=0.3),
        make_draft(),
        chunks(("kb-001", 0.9)),
        kb,
        settings,
    )
    assert report.score < settings.thresholds.auto_send
    assert any("action safety" in p for p in report.penalties)


def test_no_retrieval_means_no_confidence(kb, settings: Settings):
    report = score_confidence(
        make_critique(), make_draft(citations=[]), [], kb, settings
    )
    assert report.score <= 0.35
    assert any("knowledge base" in p for p in report.penalties)


def test_self_reported_unsupported_claims_cost_score(kb, settings: Settings):
    clean = score_confidence(
        make_critique(), make_draft(), chunks(("kb-001", 0.7)), kb, settings
    )
    dirty = score_confidence(
        make_critique(),
        make_draft(unsupported_claims=["The 60-minute expiry is configurable."]),
        chunks(("kb-001", 0.7)),
        kb,
        settings,
    )
    assert dirty.score < clean.score
    assert any("unsupported" in p for p in dirty.penalties)


def test_weak_retrieval_caps_the_score(kb, settings: Settings):
    report = score_confidence(
        make_critique(), make_draft(), chunks(("kb-001", 0.05)), kb, settings
    )
    assert report.score <= 0.40
    assert any("weak retrieval" in p for p in report.penalties)


def test_components_and_margin_are_reported(kb, settings: Settings):
    report = score_confidence(
        make_critique(), make_draft(), chunks(("kb-001", 0.8), ("kb-002", 0.3)),
        kb, settings,
    )
    assert set(report.components) == {
        "groundedness", "action_safety", "coverage", "retrieval"
    }
    assert report.retrieval_top_score == 0.8
    assert abs(report.retrieval_margin - 0.5) < 1e-9


def test_score_stays_in_range(kb, settings: Settings):
    report = score_confidence(
        make_critique(groundedness=0.0, coverage=0.0, action_safety=0.0),
        make_draft(
            citations=[],
            kb_covers_this=False,
            unsupported_claims=["a", "b", "c", "d", "e", "f"],
        ),
        [],
        kb,
        settings,
    )
    assert 0.0 <= report.score <= 1.0


def test_low_coverage_blocks_auto_send(kb, settings: Settings):
    """Half an answer is not a deflection - the customer writes back."""
    report = score_confidence(
        make_critique(groundedness=0.95, coverage=0.45, action_safety=0.95),
        make_draft(),
        chunks(("kb-001", 0.9)),
        kb,
        settings,
    )
    assert report.score < settings.thresholds.auto_send
    assert any("coverage" in p for p in report.penalties)


def test_needing_customer_input_blocks_auto_send(kb, settings: Settings):
    """The pipeline may not send a finished answer and ask a question at the same time."""
    report = score_confidence(
        make_critique(blocked_on_customer_input=True),
        make_draft(),
        chunks(("kb-001", 0.9)),
        kb,
        settings,
    )
    assert report.score < settings.thresholds.auto_send
    assert report.score >= settings.thresholds.clarify_floor  # still clarifiable
    assert any("customer input" in p for p in report.penalties)
