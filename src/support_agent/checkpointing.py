"""Where a paused run lives while it waits for a person.

Two problems have to be solved before :class:`~support_agent.ladder.Ladder` can suspend a
ticket on a human, and both are about the store rather than the graph:

* **It has to outlive the process.** A review that only the process which opened it can
  answer is not a review step; it is a callback with extra ceremony.
* **It has to be able to read our own types back.** LangGraph will not deserialise
  arbitrary classes out of a checkpoint - reasonably, since that is an arbitrary-import
  gadget - so every pipeline model has to be named in an allowlist.

Both are handled here so that the call site stays a one-liner.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel

from . import models, nodes

__all__ = ["checkpoint_types", "serializer", "sqlite_saver"]


#: Every module that defines a type which reaches a checkpoint.  ``nodes`` is easy to
#: forget and matters most: ``LadderState`` is the graph state itself, so leaving it out
#: does not degrade gracefully - with an explicit allowlist in place, an omission is a
#: hard block rather than the warning an empty allowlist produces.
_CHECKPOINTED_MODULES = (models, nodes)


def checkpoint_types() -> list[tuple[str, str]]:
    """Every persisted type, as the ``(module, name)`` pairs the allowlist wants.

    Derived from the modules rather than typed out, because a hand-maintained list of
    this shape rots silently: the failure mode of forgetting an entry is not a test
    failure but a review queue that has become unreadable, discovered by whoever is
    on call.
    """
    return sorted(
        (module.__name__, name)
        for module in _CHECKPOINTED_MODULES
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, BaseModel)
        and obj.__module__ == module.__name__
    )


def serializer() -> JsonPlusSerializer:
    """A serialiser that will read this pipeline's models back out of a checkpoint.

    Without it LangGraph warns on every load today and refuses outright in a later
    version - which would turn a queue of paused tickets into an unrecoverable one.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())


@contextmanager
def sqlite_saver(path: str | Path) -> Iterator[BaseCheckpointSaver]:
    """A durable checkpointer, for a ladder that pauses on human review.

    SQLite is enough for a single support queue and needs no infrastructure; the same
    ``Ladder`` code takes a Postgres saver unchanged when one queue becomes several.
    Requires the optional dependency::

        pip install "support-agent[sqlite]"
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "a durable checkpointer needs langgraph-checkpoint-sqlite: "
            'pip install "support-agent[sqlite]"'
        ) from exc

    connection = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(connection, serde=serializer())
        saver.setup()
        yield saver
    finally:
        connection.close()
