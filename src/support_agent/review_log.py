"""A record of what human reviewers decided, and what the agent had predicted.

Every verdict is a **label on the confidence gate**.  The agent scored a reply 0.74 and a
person either sent it or did not; that pair is the only evidence that can answer "what is
our actual false-send rate at 0.78?".

Without it, moving ``auto_send`` is an argument nobody can win - the number that decides
how many customers get an automated answer would be tuned on instinct, which is exactly
the failure the thresholds were gathered into :mod:`support_agent.config` to avoid.

Deliberately JSONL and deliberately append-only: one decision per line, no schema
migration, readable with ``jq``, and safe to tail while it is being written.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from .models import Resolution

__all__ = ["append", "read", "summarise"]


def _entry(resolution: Resolution) -> dict | None:
    """One line's worth, or ``None`` if this resolution has nothing to say.

    A run that never paused, and one still waiting on its reviewer, are both skipped -
    the first has no verdict and the second does not have one *yet*.
    """
    record = resolution.review
    if record is None or resolution.pending_review:
        return None

    verdict = record.verdict
    confidence = resolution.confidence
    return {
        "ticket_id": resolution.ticket_id,
        "logged_at": time.time(),
        "site": record.request.site,
        # What the agent believed, in the same breath as what the human decided. Kept
        # side by side on one line so the join never has to be reconstructed later.
        "score": None if confidence is None else round(confidence.score, 4),
        "components": {} if confidence is None else {
            k: round(v, 4) for k, v in confidence.components.items()
        },
        "penalties": [] if confidence is None else list(confidence.penalties),
        "action": None if verdict is None else verdict.action,
        "reviewer": None if verdict is None else verdict.reviewer,
        "edited": bool(verdict and verdict.edited_reply),
        "waited_s": round(
            (verdict.decided_at if verdict else time.time()) - record.request.requested_at,
            2,
        ),
        # Non-empty for a timeout or a rejected verdict. These are the interesting rows:
        # they say the review step itself is not working, not that the agent was wrong.
        "refused": record.refused,
        "route": resolution.route,
    }


def append(path: str | Path, resolution: Resolution) -> bool:
    """Record one finished review.  Returns whether anything was written."""
    entry = _entry(resolution)
    if entry is None:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return True


def read(path: str | Path) -> Iterator[dict]:
    """Every entry, skipping any line that is not readable.

    A log is worth more than its worst line: a truncated final write - which is what a
    crashed process leaves behind - should not make the history unreadable.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def summarise(entries: list[dict]) -> dict:
    """What the log says about the gate.

    ``agreed`` is the number worth watching: how often a human released what the agent
    produced, unchanged.  A high rate at a given score band is the argument for lowering
    ``auto_send``; a low one is the argument against, and neither is available without
    this log.
    """
    answered = [e for e in entries if e.get("action") and not e.get("refused")]
    agreed = [e for e in answered if e["action"] == "approve"]
    scores = [e["score"] for e in answered if e.get("score") is not None]

    actions: dict[str, int] = {}
    for entry in answered:
        actions[entry["action"]] = actions.get(entry["action"], 0) + 1

    waits = [e["waited_s"] for e in answered if e.get("waited_s") is not None]
    return {
        "reviews": len(entries),
        "answered": len(answered),
        "refused": sum(1 for e in entries if e.get("refused")),
        "actions": dict(sorted(actions.items())),
        "agreement": len(agreed) / len(answered) if answered else 0.0,
        "median_score": sorted(scores)[len(scores) // 2] if scores else None,
        "median_wait_s": sorted(waits)[len(waits) // 2] if waits else None,
        "reviewers": sorted({e["reviewer"] for e in answered if e.get("reviewer")}),
    }
