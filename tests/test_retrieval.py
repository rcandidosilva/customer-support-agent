from __future__ import annotations

import pytest

from support_agent.retrieval import KnowledgeBase, tokenize


def test_loads_every_article(kb: KnowledgeBase):
    assert len(kb.articles) == 12
    assert all(a.id.startswith("kb-") for a in kb.articles)
    assert all(a.title and a.body for a in kb.articles)


def test_tokenize_drops_support_boilerplate():
    tokens = tokenize("Hi team, please help, my API token is broken")
    assert "hi" not in tokens
    assert "please" not in tokens
    assert "token" in tokens and "api" in tokens


@pytest.mark.parametrize(
    "query,expected",
    [
        ("password reset email never arrives", "kb-001"),
        ("saml sso okta enforcement", "kb-002"),
        ("lost my 2fa device recovery codes", "kb-003"),
        ("refund annual plan 14 days", "kb-011"),
        ("card declined dunning read-only", "kb-012"),
        ("dashboard stale freshness backfill", "kb-021"),
        ("429 rate limit retry-after burst", "kb-031"),
        ("sla credit uptime incident", "kb-041"),
    ],
)
def test_search_finds_the_right_article(kb: KnowledgeBase, query: str, expected: str):
    hits = kb.search(query, top_k=3)
    assert hits, f"no hits for {query!r}"
    assert expected in {h.article_id for h in hits}


def test_scores_are_normalised_and_ordered(kb: KnowledgeBase):
    hits = kb.search("proration seat changes invoice", top_k=5)
    assert hits
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_off_topic_query_scores_low(kb: KnowledgeBase):
    """The threshold in config.py only means something if noise scores below it."""
    hits = kb.search("recipe for sourdough starter hydration", top_k=3)
    assert not hits or hits[0].score < 0.12


def test_empty_query_returns_nothing(kb: KnowledgeBase):
    assert kb.search("the and of a", top_k=5) == []


def test_chunks_are_deduplicated_by_heading(kb: KnowledgeBase):
    hits = kb.search("refund cancel policy annual monthly", top_k=8)
    keys = [(h.article_id, h.title) for h in hits]
    assert len(keys) == len(set(keys))


def test_has_article(kb: KnowledgeBase):
    assert kb.has_article("kb-011")
    assert not kb.has_article("kb-999")
