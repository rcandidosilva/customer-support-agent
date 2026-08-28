"""Command line entry point.

    support-agent run examples/tickets/*.json      # the full ladder (needs an API key)
    support-agent search "dashboard is stale"      # retrieval only, no API calls
    support-agent lint packet.json                 # check a saved handoff packet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .handoff_lint import has_errors, lint_packet
from .llm import AnthropicLLM, MissingCredentials
from .models import HandoffPacket, Ticket
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


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.model:
        settings = settings.with_model(args.model)

    kb = KnowledgeBase.from_dir(settings.kb_dir)
    tickets = _load_tickets(args.tickets)

    try:
        llm = AnthropicLLM(settings.model)
    except (MissingCredentials, ImportError) as exc:
        print(
            f"{exc}\n"
            "`support-agent search` and the test suite run without any credentials.",
            file=sys.stderr,
        )
        return 2

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
    run.add_argument("--model", help="override the model id")
    run.set_defaults(func=cmd_run)

    search = sub.add_parser("search", help="query the knowledge base (no API calls)")
    search.add_argument("query")
    search.add_argument("-k", "--top-k", type=int, default=5)
    search.add_argument("-v", "--verbose", action="store_true")
    search.set_defaults(func=cmd_search)

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
