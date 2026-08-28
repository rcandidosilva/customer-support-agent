"""A small, dependency-free BM25 index over the markdown knowledge base.

This is deliberately not a vector store.  For a twelve-article help centre, lexical
retrieval with field boosts beats embeddings on both accuracy and explainability, and it
keeps the sample runnable with nothing but the Anthropic SDK installed.  Swap
``KnowledgeBase.search`` for an embedding lookup and nothing else in the pipeline changes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .models import KBArticle, RetrievedChunk

_WORD = re.compile(r"[a-z0-9][a-z0-9_\-/.]*")

# Support vocabulary is full of these and they carry no signal.
_STOPWORDS = frozenset("""
a an and are as at be been but by can cannot do does doing for from get got had has have
how i if in into is it its me my not of on or our so than that the their then there these
they this to too us was we were what when where which who why will with would you your
please hi hello thanks thank regards team support ticket issue problem
""".split())

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _parse_list(value: str) -> list[str]:
    return [p.strip() for p in value.strip("[]").split(",") if p.strip()]


def load_articles(kb_dir: Path) -> list[KBArticle]:
    articles: list[KBArticle] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        articles.append(
            KBArticle(
                id=meta.get("id", path.stem),
                title=meta.get("title", path.stem),
                category=meta.get("category", "other"),
                tags=_parse_list(meta.get("tags", "")),
                body=body.strip(),
                path=str(path),
            )
        )
    return articles


def _chunk(article: KBArticle) -> list[tuple[str, str]]:
    """Split an article on ``##`` headings, keeping each chunk self-describing.

    Every chunk carries the article title so a passage lifted out of context still says
    what it is about - which matters when five chunks from four articles are pasted into
    one prompt.
    """
    parts: list[tuple[str, str]] = []
    current_heading = article.title
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            parts.append((current_heading, text))

    for line in article.body.splitlines():
        if line.startswith("## "):
            flush()
            current_heading = line[3:].strip()
            buf = [line]
        else:
            buf.append(line)
    flush()
    return parts or [(article.title, article.body)]


class KnowledgeBase:
    """BM25 over heading-level chunks, with a boost for title and tag matches."""

    K1 = 1.4
    B = 0.72
    TITLE_BOOST = 2.0
    TAG_BOOST = 1.6
    MAX_CHUNKS_PER_ARTICLE = 2

    def __init__(self, articles: list[KBArticle]) -> None:
        self.articles = articles
        self.by_id = {a.id: a for a in articles}
        self._chunk_meta: list[tuple[str, str, str]] = []  # (article_id, heading, text)
        self._chunk_tf: list[Counter[str]] = []
        self._chunk_len: list[int] = []

        for article in articles:
            boost_tokens = (
                tokenize(article.title) * int(self.TITLE_BOOST)
                + tokenize(" ".join(article.tags)) * int(self.TAG_BOOST)
                + tokenize(article.category)
            )
            for heading, text in _chunk(article):
                tokens = tokenize(text) + tokenize(heading) + boost_tokens
                self._chunk_meta.append((article.id, heading, text))
                self._chunk_tf.append(Counter(tokens))
                self._chunk_len.append(len(tokens))

        n = len(self._chunk_tf) or 1
        self._avg_len = sum(self._chunk_len) / n
        df: Counter[str] = Counter()
        for tf in self._chunk_tf:
            df.update(tf.keys())
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        #: Highest achievable raw score, used to normalise into 0..1 so thresholds in
        #: ``config.py`` mean something stable as the corpus grows.
        self._max_idf = max(self._idf.values(), default=1.0)

    @classmethod
    def from_dir(cls, kb_dir: Path) -> KnowledgeBase:
        return cls(load_articles(kb_dir))

    def _score(self, index: int, query_terms: list[str]) -> float:
        tf = self._chunk_tf[index]
        length = self._chunk_len[index] or 1
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.K1 * (1 - self.B + self.B * length / self._avg_len)
            score += idf * (freq * (self.K1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        terms = tokenize(query)
        if not terms:
            return []
        ceiling = self._max_idf * (self.K1 + 1) * len(set(terms))
        scored: list[tuple[float, int]] = []
        for i in range(len(self._chunk_tf)):
            raw = self._score(i, terms)
            if raw > 0:
                scored.append((raw, i))
        scored.sort(reverse=True)

        results: list[RetrievedChunk] = []
        seen_headings: set[tuple[str, str]] = set()
        per_article: Counter[str] = Counter()
        for raw, i in scored:
            article_id, heading, text = self._chunk_meta[i]
            if (article_id, heading) in seen_headings:
                continue
            # Cap chunks per article so a single strongly-matching document cannot fill
            # the whole window.  Breadth matters downstream: the drafter may only cite
            # articles that were actually surfaced, and the reviewer needs to see the
            # article that contradicts the draft as well as the one that supports it.
            if per_article[article_id] >= self.MAX_CHUNKS_PER_ARTICLE:
                continue
            per_article[article_id] += 1
            seen_headings.add((article_id, heading))
            article = self.by_id[article_id]
            results.append(
                RetrievedChunk(
                    article_id=article_id,
                    title=f"{article.title} - {heading}"
                    if heading != article.title
                    else article.title,
                    score=round(min(raw / ceiling, 1.0), 4) if ceiling else 0.0,
                    text=text,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def has_article(self, article_id: str) -> bool:
        return article_id in self.by_id
