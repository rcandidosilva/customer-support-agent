"""Command line entry point.

    support-agent run examples/tickets/*.json      # the full ladder (needs an API key)
    support-agent run t.json --provider openai     # the same ladder, other provider
    support-agent search "dashboard is stale"      # retrieval only, no API calls
    support-agent lint packet.json                 # check a saved handoff packet
    support-agent review --list                    # the human-review queue
    support-agent review --approve TKT-1 --as you  # answer one

The ``review`` verb is the operator surface for runs paused on a person.  It needs the
same ``--db`` the run used, because a paused ticket lives in the checkpoint rather than in
any process.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from . import review_log
from .checkpointing import sqlite_saver
from .config import PROVIDER_MODELS, Settings
from .handoff_lint import has_errors, lint_packet
from .ladder import NoReviewPending
from .llm import MissingCredentials, RoutedLLM, build_llm
from .models import HandoffPacket, ReviewVerdict, Ticket
from .render import render_packet, render_resolution
from .retrieval import KnowledgeBase


def _load_tickets(paths: list[str]) -> list[Ticket]:
    tickets: list[Ticket] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"no such ticket file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data if isinstance(data, list) else [data]:
            tickets.append(Ticket.model_validate(item))
    return tickets


def _model_settings(args: argparse.Namespace) -> Settings:
    """Environment first, then the flags, most specific last.

    The same order :meth:`Settings.from_env` uses internally, so ``--provider openai
    --stage-model handoff=anthropic:claude-opus-5`` composes the way it reads.
    """
    settings = Settings.from_env()
    if args.provider:
        settings = settings.with_provider(args.provider)
    if args.model:
        settings = settings.with_model(args.model)
    for pair in args.stage_model or ():
        stage, _, ref = pair.partition("=")
        if not ref:
            raise SystemExit(f"--stage-model wants STAGE=MODEL, got {pair!r}")
        settings = settings.with_stage_model(stage.strip(), ref)
    return settings


def cmd_run(args: argparse.Namespace) -> int:
    try:
        settings = _model_settings(args)
    except ValueError as exc:
        # A bad provider or model reference.  Caught here so it reads as a usage error
        # rather than a traceback out of config.
        print(exc, file=sys.stderr)
        return 2

    kb = KnowledgeBase.from_dir(settings.kb_dir)
    tickets = _load_tickets(args.tickets)

    try:
        llm = build_llm(settings)
    except (MissingCredentials, ImportError) as exc:
        print(
            f"{exc}\n"
            "`support-agent search` and the test suite run without any credentials.",
            file=sys.stderr,
        )
        return 2

    if isinstance(llm, RoutedLLM):
        # Worth saying out loud: a mixed route is the one configuration where reading
        # the trace without knowing the plan is misleading.
        route = "  ".join(f"{k}={v}" for k, v in llm.routing().items())
        print(f"models: {route}", file=sys.stderr)
    else:
        print(f"model: {settings.model_ref()}", file=sys.stderr)

    from .ladder import Ladder

    ladder = Ladder(llm, kb, settings)
    resolutions = [ladder.run(t) for t in tickets]

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in resolutions], indent=2))
    else:
        for r in resolutions:
            print(render_resolution(r, show_trace=not args.no_trace))
            print("\n" + "=" * 78 + "\n")

    counts: dict[str, int] = {}
    for r in resolutions:
        counts[r.route] = counts.get(r.route, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    deflected = counts.get("send", 0) + counts.get("clarify", 0)
    print(
        f"{len(resolutions)} ticket(s): {summary}  "
        f"(deflection {deflected / max(len(resolutions), 1):.0%})",
        file=sys.stderr,
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    kb = KnowledgeBase.from_dir(settings.kb_dir)
    chunks = kb.search(args.query, top_k=args.top_k)
    if not chunks:
        print("no matches")
        return 1
    for c in chunks:
        print(f"{c.score:6.3f}  [{c.article_id}] {c.title}")
        if args.verbose:
            body = c.text.strip().splitlines()
            print("        " + "\n        ".join(body[:6]))
            print()
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.packet).read_text(encoding="utf-8"))
    # Accept either a bare packet or a full resolution containing one.
    if "packet" in data and isinstance(data["packet"], dict):
        data = data["packet"]
    packet = HandoffPacket.model_validate(data)

    if args.render:
        print(render_packet(packet))
        print()

    findings = lint_packet(packet)
    if not findings:
        print("clean")
        return 0
    for f in findings:
        print(f)
    return 1 if has_errors(findings) else 0


# --------------------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------------------


def _reviewing_ladder(args: argparse.Namespace, saver):
    """A ladder wired for the queue.

    Most verdicts need no model call - the reviewer supplies the words, and everything
    that needed the model ran before the pause.  Two do not: rejecting a draft to a
    person, and a sweep that releases an unreviewed draft, both have to write a brief.

    So a real client is used when one can be built, and :class:`UnavailableLLM` when it
    cannot.  That degrades a keyless escalation into the deterministically assembled
    packet the ladder already falls back to, rather than a traceback.
    """
    from .ladder import Ladder
    from .llm import UnavailableLLM

    settings = Settings.from_env().with_review()
    try:
        llm = build_llm(settings)
    except (MissingCredentials, ImportError) as exc:
        llm = UnavailableLLM(str(exc))

    return Ladder(
        llm,
        KnowledgeBase.from_dir(settings.kb_dir),
        settings,
        checkpointer=saver,
        verdict_log=args.log,
    )


def _read_reply(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8").strip()
    return (args.reply or "").strip()


def _verdict(args: argparse.Namespace, action: str) -> ReviewVerdict:
    return ReviewVerdict(
        action=action,
        reviewer=args.reviewer,
        edited_reply=_read_reply(args),
        rationale=args.why or "",
    )


def _print_summary(log: str) -> int:
    entries = list(review_log.read(log))
    if not entries:
        print(f"no verdicts recorded in {log}")
        return 0
    summary = review_log.summarise(entries)
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        if isinstance(value, float):
            value = f"{value:.0%}" if key == "agreement" else f"{value:.2f}"
        print(f"  {key:<{width}}  {value}")
    return 0


def _print_queue(ladder) -> int:
    pending = ladder.paused_threads()
    stalled = ladder.stalled_threads()
    if not pending and not stalled:
        print("the review queue is empty")
        return 0

    now = time.time()
    for request in sorted(pending, key=lambda r: r.expires_at):
        left = (request.expires_at - now) / 60
        clock = f"{left:.0f} min left" if left > 0 else "OVERDUE"
        print(f"  {request.thread_id:<16} {request.site:<10} {clock:>12}   "
              f"{request.reason[:60]}")
    for thread_id in stalled:
        # Surfaced beside the queue because these are invisible everywhere else: no
        # interrupt means no deadline, and no sweep will ever reach them.
        print(f"  {thread_id:<16} {'STALLED':<10} {'--retry':>12}   "
              "stopped partway; nothing is waiting on a person")
    return 0


def _print_pause(ladder, thread_id: str, *, show_trace: bool) -> int:
    request = ladder.pending_review(thread_id)
    if request is None:
        print(f"no review pending on {thread_id}", file=sys.stderr)
        return 1

    from .nodes import LadderState, to_paused_resolution

    # Reading the checkpoint rather than re-running: the state already holds everything
    # the reviewer needs, and re-running would walk back into the pause.
    state = ladder.graph.get_state({"configurable": {"thread_id": thread_id}})
    print(render_resolution(
        to_paused_resolution(
            LadderState.model_validate(state.values), time.time(), request
        ),
        show_trace=show_trace,
    ))
    return 0


_VERDICT_FLAGS = ("approve", "edit_and_send", "ask_customer", "escalate")


def cmd_review(args: argparse.Namespace) -> int:
    if args.summary:
        # The only verb that reads the log rather than the queue.
        return _print_summary(args.log)

    with sqlite_saver(args.db) as saver:
        ladder = _reviewing_ladder(args, saver)

        if args.list:
            return _print_queue(ladder)

        if args.show:
            return _print_pause(ladder, args.show, show_trace=not args.no_trace)

        if args.retry:
            resolution = ladder.retry(args.retry)
            print(render_resolution(resolution, show_trace=not args.no_trace))
            return 0

        if args.sweep:
            released = ladder.expire_overdue()
            if not released:
                print("nothing overdue")
                return 0
            for resolution in released:
                print(f"  {resolution.ticket_id} released to the queue "
                      f"({resolution.review.refused})")
            return 0

        action = next((a for a in _VERDICT_FLAGS if getattr(args, a)), None)
        if action is None:
            print(
                "nothing to do: pass --list, --show, --sweep, --retry, --summary, "
                "or a verdict",
                file=sys.stderr,
            )
            return 2

        try:
            resolution = ladder.resume(getattr(args, action), _verdict(args, action))
        except NoReviewPending as exc:
            print(exc, file=sys.stderr)
            return 1
        except ValidationError as exc:
            print(f"the verdict is incomplete: {exc.errors()[0]['msg']}", file=sys.stderr)
            return 2

    print(render_resolution(resolution, show_trace=not args.no_trace))
    if resolution.review is not None and not resolution.review.accepted:
        print(f"\nnot applied: {resolution.review.refused}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-agent",
        description="Customer-support deflection with a confidence-gated "
        "escalation ladder.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run tickets through the full ladder")
    run.add_argument("tickets", nargs="+", help="ticket JSON files")
    run.add_argument("--json", action="store_true", help="emit resolutions as JSON")
    run.add_argument("--no-trace", action="store_true", help="hide the per-stage trace")
    run.add_argument("--provider", choices=sorted(PROVIDER_MODELS),
                     help="which provider every stage runs on, with its default model")
    run.add_argument("--model",
                     help="the default model: `gpt-5`, or `openai:gpt-5` to switch "
                          "provider at the same time")
    run.add_argument("--stage-model", dest="stage_model", metavar="STAGE=MODEL",
                     action="append",
                     help="pin one stage to its own model; repeatable. e.g. "
                          "--stage-model classify=openai:gpt-5-mini")
    run.set_defaults(func=cmd_run)

    search = sub.add_parser("search", help="query the knowledge base (no API calls)")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=5)
    search.add_argument("-v", "--verbose", action="store_true")
    search.set_defaults(func=cmd_search)

    review = sub.add_parser(
        "review",
        help="the human-review queue: list, inspect, and answer paused tickets",
        description="A paused ticket lives in the checkpoint, not in a process, so every "
        "verb here needs the same --db the run used.",
    )
    review.add_argument("--db", default="reviews.db", help="the checkpoint store")
    review.add_argument("--log", default="verdicts.jsonl",
                        help="where verdicts are recorded")
    review.add_argument("--as", dest="reviewer", default="",
                        help="who is deciding; required to answer")
    review.add_argument("--no-trace", action="store_true")

    what = review.add_mutually_exclusive_group()
    what.add_argument("--list", action="store_true", help="everything waiting")
    what.add_argument("--show", metavar="TICKET", help="the full brief or draft")
    what.add_argument("--sweep", action="store_true",
                      help="release everything past its deadline")
    what.add_argument("--retry", metavar="TICKET",
                      help="drive a STALLED run to completion from where it stopped")
    what.add_argument("--summary", action="store_true",
                      help="what the verdict log says about the gate")
    what.add_argument("--approve", metavar="TICKET",
                      help="release what the agent produced, unchanged")
    what.add_argument("--edit-and-send", dest="edit_and_send", metavar="TICKET",
                      help="send your own reply instead (needs --reply or --file)")
    what.add_argument("--ask-customer", dest="ask_customer", metavar="TICKET",
                      help="ask the customer for what is missing (needs --reply/--file)")
    what.add_argument("--escalate", metavar="TICKET",
                      help="send it to a person (needs --why)")

    review.add_argument("--reply", help="the reply or question to send")
    review.add_argument("--file", help="read the reply from a file instead")
    review.add_argument("--why", help="rationale, required when escalating")
    review.set_defaults(func=cmd_review)

    lint = sub.add_parser("lint", help="lint a saved handoff packet")
    lint.add_argument("packet", help="JSON file: a packet, or a resolution containing one")
    lint.add_argument("--render", action="store_true", help="also print the brief")
    lint.set_defaults(func=cmd_lint)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
