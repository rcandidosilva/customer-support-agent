# Customer-support deflection with an escalation ladder

A worked example of **confidence-gated routing**: a ticket comes in, an automated tier-1
agent tries to answer it from a knowledge base, and an independent reviewer decides
whether that answer is good enough to send, good enough to ask one follow-up question
about, or not good enough at all — in which case the agent writes a **handoff packet** for
a human colleague and gets out of the way. Optionally, it stops there and waits for that
colleague: see [Human review](#human-review).

The interesting artifact here is the handoff packet, not the answer. Any RAG pipeline can
answer easy tickets. The thing worth building carefully is what happens on the ~30% it
cannot: whether the human who inherits the ticket starts a minute ahead of where they
would have started with the raw email, or a minute behind.

```
                    ┌─────────────┐
   ticket  ────────▶│  classify   │  category, severity, risk, "does this need
                    └──────┬──────┘   account data I cannot see?"
                           ▼
                    ┌─────────────┐
                    │  retrieve   │  BM25 over the help centre (local, cannot fail)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐   legal · security · churn · money over the
                    │ policy gate │   approval line · pasted secrets · repeat
                    └──────┬──────┘   escalations  ──────────────────────┐
                           ▼                                             │
                    ┌─────────────┐                                      │
                    │ tier-1 draft│  writes the reply AND declares what   │
                    └──────┬──────┘  it could not support                 │
                           ▼                                             │
                    ┌─────────────┐  a separate call that did not write   │
                    │  critique   │  the draft and is told to distrust it │
                    └──────┬──────┘                                       │
                           ▼                                             │
                    ┌─────────────┐  model judgement × mechanical checks  │
                    │    fuse     │  (do the cited articles exist?)       │
                    └──────┬──────┘                                       │
                           ▼                                             │
              ┌────────────┼────────────┐                                │
         ≥0.78│       0.45─0.78│        │<0.45                           │
              ▼            ▼            ▼                                ▼
          ┌───────┐   ┌─────────┐   ┌──────────────────────────────────────┐
          │ SEND  │   │ CLARIFY │   │  ESCALATE  →  handoff packet          │
          └───────┘   └─────────┘   └──────────────────────────────────────┘
              ▲            ▲          only if the      written for a colleague,
              │            │          missing piece    not a customer
              │            │          is the           ├─ lint-checked for voice
              │            │          customer's       └─ falls back to an auto-
              │            │          to give             assembled packet if the
              │            │                              model is unavailable
              │            │                     │
              │            │                     ▼  (optional, off by default)
              │            │            ┌──────────────────┐
              └────────────┴────────────│  HUMAN REVIEW    │  a named reviewer may
             only on a validated verdict└────────┬─────────┘  approve, edit and send,
             from a named reviewer               │            or turn it into a
                                                 ▼            question — or the SLA
                                        the queue, unchanged  runs out and it queues
```

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

**Everything except the live model calls runs with no API key.** The demo replays all
three routes — send, clarify, escalate — through the real ladder, the real confidence
fusion, and the real renderer, with only the model's words canned:

```bash
.venv/bin/python examples/offline_demo.py
```

```bash
.venv/bin/support-agent search "dashboard numbers are stale" -v
```

```bash
.venv/bin/python -m pytest -q
```

The full ladder needs one. Set `ANTHROPIC_API_KEY`, then:

```bash
.venv/bin/support-agent run examples/tickets/*.json
```

Each ticket prints its route, the confidence breakdown, the reply or the rendered handoff
brief, and a per-stage trace with token counts and the model that spent them. `--json`
emits the whole `Resolution` object instead; `support-agent lint packet.json` re-checks a
saved packet.

### Which model runs which stage

Anthropic is the default. OpenAI needs its own extra and its own key
(`pip install -e ".[openai]"`, `OPENAI_API_KEY`), and then either provider can run any
stage:

```bash
.venv/bin/support-agent run examples/tickets/*.json --provider openai
```

A **model reference** is `provider:model`, or a bare `model` belonging to the configured
provider. It is the same spelling everywhere a model can be named — flag, env var, or
`StageConfig` — and per-stage references are what make a route mixed:

```bash
.venv/bin/support-agent run examples/tickets/*.json \
    --provider openai \
    --stage-model critique=anthropic:claude-opus-5 \
    --stage-model handoff=anthropic:claude-opus-5
```

Cheap triage, drafting, and clarification on one provider; the two stages where being
wrong is expensive — the critique that sets the confidence score, and the brief a person
has to act on — on another. The same thing in the environment, so it is deployment
config rather than a command line:

```bash
SUPPORT_AGENT_PROVIDER=openai
SUPPORT_AGENT_MODEL_CRITIQUE=anthropic:claude-opus-5
```

The stage code does not change, and neither do the schemas. `build_llm(settings)` reads
the table, builds one client per distinct model, and returns a plain adapter when every
stage agrees or a `RoutedLLM` when they do not — so a single-model setup pays nothing for
a feature it is not using. Every model on the route is built at startup, which keeps a
missing key a startup failure rather than something the first escalation discovers.

Each trace row names the model that answered it, because with a mixed route unattributed
token counts cannot be costed:

```
| stage    | ok | model                  | in | out | ms | detail                |
| classify | ok | openai:gpt-5-mini      | .. | ..  | .. | account/normal        |
| critique | ok | anthropic:claude-opus-5| .. | ..  | .. | blocker=low_confidence|
```

## The three design decisions worth stealing

### 1. Confidence is not what the model says its confidence is

Asking a model to rate the answer it just wrote produces a number that is fluent and
uncalibrated. Two things fix most of that:

**The reviewer is a separate call.** [`stages/critic.py`](src/support_agent/stages/critic.py)
gets the same passages and the finished draft, is told it did not write it, and is told to
assume the draft is wrong until the passages say otherwise. It scores groundedness,
coverage, and *action safety* — how expensive it is to be wrong — independently, because
those fail in different directions. A reply can be perfectly grounded and still unsafe to
send unreviewed.

**Then the score is fused with checks the model cannot argue with.** Chiefly: did the
drafter cite an article that retrieval never actually returned? If so, it is not reasoning
from the evidence, and the score is capped at 0.25 no matter how good the prose is.
Retrieval strength, self-reported unsupported claims, and hard floors on groundedness,
action safety, and coverage all feed in as caps rather than weights — a floor is not
something a strong score elsewhere should be able to buy its way past.

```python
score = 0.40*groundedness + 0.25*action_safety + 0.20*coverage + 0.15*retrieval
# then capped by: fabricated citations · weak retrieval · KB gaps
#                 floors on groundedness, action safety, and coverage
#                 and one consistency rule (below)
```

The last cap is not a tuning knob at all. If the reviewer set
`blocked_on_customer_input`, the score cannot reach the send threshold — the pipeline may
not hand over a finished answer and say it needs more information in the same breath. That
one line is what turned a plausible-looking 0.81 into a clarifying question during
development; the demo's second scenario is the case that found it.

### 2. "Ask a question" and "escalate" are different failures

Both mean *not confident*. The thing that separates them is one boolean:
`blocked_on_customer_input` — can the customer themselves supply the missing piece? An
invoice number, which of two things they meant, what the error actually said: ask. Data
that lives in our systems, a judgement call, or a draft that is simply wrong: escalate.

Asking a customer a question we could have answered ourselves is worse than escalating,
because it spends their patience *and* still ends up on a human's desk. So the ladder caps
clarifying questions at one round, and after that the ticket goes to a person carrying the
question we already asked.

### 3. The handoff packet is written for a colleague, and that is harder than it sounds

Every other stage in this pipeline writes to a customer. By the time the model reaches the
handoff, its priors about "support text" are loaded, and they are exactly wrong: the brief
comes out opening with "Thanks for reaching out", apologising to nobody in the room, and
hedging a clear policy statement into mush.

Two things push back. The system prompt in
[`stages/handoff.py`](src/support_agent/stages/handoff.py) spends most of its length on
audience — *your reader has eleven other tickets and forty seconds* — and specifies each
field by what makes it useful rather than what it contains. And
[`handoff_lint.py`](src/support_agent/handoff_lint.py) checks the result mechanically
and **sends it back to be rewritten once if it fails**:
customer-voice phrases, greetings, stacked hedges, next steps that are not steps
("investigate further"), findings that trace to no article, bare open questions, and a
missing statement of what was already sent. Being plain string rules, they are asserted in
the test suite with no judge model and no API key.

Errors trigger a rewrite that names the specific defects — repeating the style rules would
just repeat what the model already had and ignored. Warnings are recorded and shipped: a
slightly long subject line is not worth another model call. Two failures in a row means a
prompt problem rather than a sampling problem, so the loop stops and the packet ships with
the complaint attached to the trace.

The field that matters most is `already_told_customer`. A colleague who contradicts
something the bot already said turns a support ticket into a complaint.

Here is a real one, from `examples/offline_demo.py`:

```markdown
# [Billing/High] Duplicate $960 annual charge - refund authorisation  ·  `TKT-1003`

**Priority:** high  ·  **Blocked on:** Needs account data the agent cannot read

Brightpath Logistics was charged $960 on consecutive days and wants the second charge
refunded today with written confirmation.

> **Why this is not answered already:** Confirming this is a duplicate rather than a
> proration requires invoice line items the agent cannot read, and the amount is above
> the agent's approval line.

## Already said to the customer

Nothing has been sent to the customer.

## What the agent already did

- Retrieved the refund policy and the billing-cycle article.
- Confirmed duplicate charges are refundable in full with no approval step (kb-011).
- Could not confirm the two charges are actually duplicates - invoice line items are not
  visible to the agent.

## Findings

- Duplicate or double charges are refunded in full, no approval needed (kb-011).
- Adding seats mid-cycle charges a prorated amount immediately, which can look like a
  second subscription charge (kb-010).
- Refunds go to the original payment method and take 5-10 business days (kb-011).
- Account renews annually in advance; two same-amount charges a day apart is not a normal
  renewal pattern.

## Open questions

- Whether the second $960 charge is a true duplicate or a mid-cycle seat addition that
  happens to match the renewal amount - the agent cannot read invoice line items.
- Whether the customer's finance team has already raised a chargeback, which would change
  how this is resolved.

## Next steps

1. Pull both charges for acct_2290 in Stripe and compare line items and invoice IDs.
2. If both are renewals, refund the later charge - no approval needed under kb-011.
3. If the second is a seat addition, reply explaining the proration rather than refunding,
   and copy the invoice line items.
4. Either way, send written confirmation the same day; they have committed to their CFO.

## Risk flags

- Customer has escalated this internally to their CFO.
- $1,600/mo account, 26 months tenure, CSAT 5 - a good account having a bad week.
```

Note what the second finding is doing. The agent is not just assembling the case for a
refund — it surfaced the article that says a mid-cycle seat addition produces a second
charge that *looks* like a duplicate, and put the alternative explanation in front of the
human before they act. That is the difference between a summary and a brief.

## Graceful degradation

Every model call is wrapped. A 500, a rate limit, a schema violation, and a refusal are
the same event to the ladder: *this rung is unavailable, take the next one down.* There is
no path out of `Ladder.run` that does not return a `Resolution`.

| What fails | What happens |
|---|---|
| Triage | Escalate immediately, with the honest reason. Retrieval still ran, so the packet has evidence. |
| Draft | Escalate with no draft. The packet says so. |
| Critique | Escalate **with** the draft attached. We have an answer that nothing checked — that is what a review queue is for. |
| Clarify | Fall through to escalation rather than send the unverified draft. |
| Handoff | Assemble a packet deterministically from whatever was collected. Worse than a written one, but it still says what is known, what was said, and why it is here. |

Note the direction: a degraded pipeline is never a *more permissive* one. Triage failing
does not mean the ticket sails through the policy gate on a default classification — it
means the ticket goes to a human with "we never classified this" written on it.

## The graph

The ladder is a LangGraph `StateGraph`. The split is deliberate: the topology lives alone
in [`graph.py`](src/support_agent/graph.py), so the entire routing policy of the
application is about forty lines with every branch named after the reason it exists, and
the node bodies live in [`nodes.py`](src/support_agent/nodes.py).
[`ladder.py`](src/support_agent/ladder.py) is a facade — `Ladder(llm, kb).run(ticket)`
still takes a ticket and returns a `Resolution`, which is all most callers want.

`Ladder.draw()` emits the compiled topology as Mermaid:

```mermaid
graph TD;
    classify --> retrieve;
    retrieve -.-> policy;
    retrieve -.-> handoff;
    policy -.-> draft;
    policy -.-> handoff;
    draft -.-> critique;
    draft -.-> handoff;
    critique -.-> gate;
    critique -.-> handoff;
    gate -.-> send;
    gate -.-> clarify;
    gate -.-> handoff;
    clarify -.-> handoff;
    handoff --> handoff_lint;
    handoff_lint -.-> handoff;
    handoff_lint -.-> finalise_escalation;
```

Three things this bought that the imperative version did not have:

**The lint loop became a real edge.** Previously `lint_packet` ran, wrote its findings to
the trace, and shipped the packet anyway — a quality check that only ever complained. It
is now a conditional edge back to the handoff node, bounded at one rewrite.

**A structural guarantee, asserted rather than asserted-by-comment.** Every failure edge
points at `handoff`, and `gate` is the only node that can reach `send`. That was true of
the imperative version too, but only if you read all 440 lines carefully.
`test_every_failure_edge_points_at_the_handoff` now checks it against the *compiled*
graph, so a future stage that quietly falls through to `send` fails a test rather than
escaping review.

**Durability.** `LadderState` is serialisable end to end, so `Ladder(..., checkpointer=...)`
persists a run and threads default to the ticket id — a ticket coming back after a
clarifying question resumes its own history instead of re-deriving it. That is also what
made the human-review step below possible: a pause is only a pause if it can be resumed.

What stayed out of the graph: [`llm.py`](src/support_agent/llm.py) still talks to Claude
through the Anthropic SDK directly, and [`policy.py`](src/support_agent/policy.py) is
still plain deterministic Python. Neither gains anything from the framework, and `llm.py`
would lose per-stage `effort`, cache-control placement, refusal handling, and — the
expensive one — `ScriptedLLM`, which is why 122 tests run in under half a second with no
network.

## Human review

The ladder's third rung hands a ticket to a person, but until recently "a person" meant a
brief landing in a queue — the graph finished and a human read the output later. Setting
`Settings.review.enabled` turns that into a real pause on a named reviewer, using
LangGraph's `interrupt()`.

```python
ladder = Ladder(llm, kb, settings.with_review(sla_minutes=30),
                checkpointer=SqliteSaver.from_conn_string("reviews.db"))

paused = ladder.run(ticket)
paused.route            # "review"  — non-terminal
paused.packet           # the brief the reviewer is looking at
paused.pending_review   # True

final = ladder.resume(ticket.id, ReviewVerdict(
    action="edit_and_send", reviewer="alice",
    edited_reply="Refunded the duplicate charge — it lands in 5-10 days.",
))
final.route             # "send"
```

Four things about the design are load-bearing.

**`run()` still returns a `Resolution`.** The obvious way to add a pause is to change the
return type, and it poisons every caller — including the ones that never enable review. So
`review` is a fourth `Route` instead, and a paused run is a resolution that happens not to
be terminal. Callers with review off cannot observe one.

**The verdict is a schema, not a resume payload.** `interrupt()` hands back whatever the
caller passes, which makes it the least-typed input in the system and the only one that
can put text in front of a customer. `ReviewVerdict` gets the same treatment as a model
response: `extra="forbid"`, a named reviewer, and cross-field checks — an `edit_and_send`
with no reply and an override with no rationale are both rejected.

**Every way of failing lands on the old behaviour.** A verdict that will not parse, one
asking for an action this pause does not offer, and a deadline that passes with nobody
looking all route to `finalise_escalation` — the brief goes to the queue exactly as it
would have without review. The failure mode of a review step is silence, and silence must
not be able to hold a ticket forever, so `review.sla_minutes` bounds it and
`expire_overdue()` enforces it on a timer:

```python
with sqlite_saver("reviews.db") as saver:
    sweeper = Ladder(llm, kb, settings.with_review(), checkpointer=saver)
    sweeper.paused_threads()   # every ticket waiting, discovered from the store
    sweeper.expire_overdue()   # release the ones past their deadline
```

`paused_threads()` enumerates the store rather than remembering anything, so the sweeper
does not have to be the process that opened the pause. Without it the deadline would bind
only on contact — when somebody happened to call `resume` — and a thread nobody touched
would wait forever, which is the failure it exists to prevent.

**The deadline lives in the checkpoint.** This is the subtle one. LangGraph re-runs a
suspended node's body from the top on resume, so any clock read inside the node restarts
at resume time — which would make a verdict impossible to be late and the SLA silently
inert. The authoritative `ReviewRequest` is the one persisted alongside the interrupt;
`Ladder.pending_review()` reads it back, and `Ladder.resume()` judges the deadline against
it before the node ever sees a verdict.

The structural guarantee in `graph.py` had to be restated rather than quietly relaxed.
`send` used to be reachable only from `gate`; it is now reachable from `gate` or from
`human_review`, and `test_every_failure_edge_points_at_the_handoff` asserts exactly that
pair. Automation still cannot promote itself — the one edge back up the ladder needs a
named human on it.

**Persistence is a first-class concern, not a config flag.**
[`checkpointing.py`](src/support_agent/checkpointing.py) exists because LangGraph will not
deserialise arbitrary classes out of a checkpoint — reasonably, since that is an
arbitrary-import gadget — so every persisted type has to be named in an allowlist. It is
derived from the modules rather than typed out, which matters more than it looks: an
*empty* allowlist only warns, but once an explicit one exists an omission becomes a hard
block. A half-complete allowlist is worse than none. `LadderState` lives in `nodes.py`
rather than `models.py` and was exactly the omission that proved the point.

## The review band

`auto_send` is a cliff. A draft scoring 0.74 with clean citations used to become a full
escalation — a person read a brief and wrote a reply from scratch, the most expensive
possible outcome for a near miss. `review.floor` (0.62) opens a band beneath the send
threshold where the draft is held for approval instead:

```
  ≥0.78          0.62–0.78            <0.62
  SEND      →  DRAFT REVIEW  →  CLARIFY / ESCALATE
               approve · edit · ask · reject
```

The same four verbs serve both pauses, but they mean different things at each, so routing
is keyed by *site* rather than by action: `approve` releases the agent's draft at the band
and queues the brief at an escalation. `escalate` is only offered at the band, because a
run that is already escalating has nowhere to send it.

**One review per ticket.** A reviewer who rejects a draft must not then be asked to review
the brief their own decision produced — that is both absurd and a way to strand a ticket
behind two deadlines.

## The verdict log

Every verdict is a label on the confidence gate. `review_log` appends one JSON object per
decision, carrying the agent's score, components and penalties *beside* what the human
did:

```bash
support-agent review --summary
  reviews       48
  answered      44
  agreement     82%
  median_score  0.71
```

That is the difference between tuning `auto_send` on evidence and tuning it on instinct.
Refused rows — a timeout, a rejected verdict — stay distinguishable from disagreement:
counting "nobody looked" as "the human disagreed" would bias every threshold decision
downstream.

## Driving the queue

```bash
support-agent review --list                      # what is waiting, and for how long
support-agent review --show TKT-4471             # the draft or brief, and the options
support-agent review --approve TKT-4471 --as you
support-agent review --edit-and-send TKT-4471 --file reply.md --as you
support-agent review --escalate TKT-4471 --why "needs invoice lines" --as you
support-agent review --sweep                     # release everything overdue
support-agent review --retry TKT-4471            # drive a STALLED run to completion
```

Three bugs came out of *running* this, not from reasoning about it, and each left a mark
on the design:

**Answering a review can need a model call.** Rejecting a draft writes a brief. The first
version assumed otherwise and crashed. `UnavailableLLM` now raises `LLMError` — the event
the ladder already survives — so a missing key degrades into the deterministic fallback
packet rather than a traceback.

**That crash stranded a ticket.** The pause was consumed and the verdict applied, then the
run stopped partway: no interrupt, so `paused_threads()` could not see it and no sweep
would ever reach it. `stalled_threads()` and `retry()` exist because of it, and `--list`
shows stalled runs beside the queue.

**Two tests were making real, billed API calls.** Credentials resolve from
`~/.config/anthropic` as well as the environment, so unsetting `ANTHROPIC_API_KEY` does
not make a test offline — it makes it quietly billable. The tell was a 52-second suite. An
autouse `no_network` fixture now fails any test that reaches for a real client.

What is still deliberately *not* here: per-reviewer routing, and any UI beyond the CLI.

## The policy gate

Some tickets see a person regardless of how confident the model is. Encoding that as a
threshold would be a mistake: thresholds get tuned, and the day someone lowers `auto_send`
from 0.78 to 0.72 to lift the deflection rate should not be the day the agent starts
denying refunds to accounts that mentioned their lawyer.

So [`policy.py`](src/support_agent/policy.py) is a separate, readable, deterministic layer
that runs *before* any confidence score is computed, and it short-circuits the two most
expensive stages when it fires. Rules are ordered most-serious-first so the reason attached
to the escalation is the one a human would lead with.

## Layout

| Path | What it is |
|---|---|
| [`models.py`](src/support_agent/models.py) | Every stage contract. LLM-facing schemas have no optional fields — "nothing to report" is an empty list, never an absent key. |
| [`config.py`](src/support_agent/config.py) | Thresholds and policy constants. Every number here is a product decision. |
| [`retrieval.py`](src/support_agent/retrieval.py) | BM25 over heading-level chunks, no dependencies. Swap `search()` for an embedding lookup and nothing else changes. |
| [`policy.py`](src/support_agent/policy.py) | Deterministic escalation rules. |
| [`stages/`](src/support_agent/stages) | One module per rung. All the prompts live here. |
| [`graph.py`](src/support_agent/graph.py) | The LangGraph topology — the whole routing policy, on one screen. |
| [`nodes.py`](src/support_agent/nodes.py) | The node bodies and the graph state. |
| [`ladder.py`](src/support_agent/ladder.py) | The facade: `Ladder(llm, kb).run(ticket) -> Resolution`, plus `pending_review` / `resume` / `expire` for the human-review pause. |
| [`handoff_lint.py`](src/support_agent/handoff_lint.py) | Mechanical quality checks on the brief. |
| [`checkpointing.py`](src/support_agent/checkpointing.py) | The durable store a paused review lives in, and the serde allowlist that lets it be read back. |
| [`review_log.py`](src/support_agent/review_log.py) | Append-only JSONL of what reviewers decided, beside what the agent predicted. |
| [`llm.py`](src/support_agent/llm.py) | The only module that talks to a provider. `AnthropicLLM` and `OpenAILLM` behind one interface, `RoutedLLM` picking between them per stage, and `ScriptedLLM` for the same interface with no network. |
| [`kb/`](src/support_agent/kb) | Twelve help-centre articles for a fictional analytics SaaS. |
| [`examples/offline_demo.py`](examples/offline_demo.py) | All three routes, scripted end to end, no credentials. |

## Testing without an API key

`ScriptedLLM` implements the same protocol as the real client and returns whatever objects
you hand it — including `LLMError` instances, which it raises. That makes every branch
assertable in milliseconds:

```python
llm = ScriptedLLM(responses={
    "classify": make_classification(),
    "draft":    make_draft(),
    "critique": LLMError("critique", "connection error"),
    "handoff":  make_packet(),
})
result = Ladder(llm, kb, settings).run(ticket)

assert result.route == "escalate"
assert result.draft is not None      # the human still gets to see it
assert result.confidence is None     # but nothing verified it
```

187 tests cover retrieval, every policy rule, the fusion caps and floors, every lint rule,
all four routes, all five degradation paths, the lint rewrite loop, both human-review
pauses (including every way a verdict can be refused), the expiry sweeper, stalled-run
recovery, the verdict log, the CLI, and the topology itself. Two spawn a real subprocess
to prove a pause survives the process that opened it — the one claim an in-memory
checkpointer cannot make — and an autouse fixture guarantees none of them can reach the
network.

The review tests were written against mutations rather than against the implementation:
each invariant was checked by breaking it on purpose and confirming a test caught it.
Two of the first three mutations survived, which is how
`test_a_refused_verdict_cannot_reach_the_customer` and
`test_the_record_keeps_the_request_that_was_actually_made` came to exist.

## Adapting this

- **Real knowledge base**: replace `KnowledgeBase.search`. The rest of the pipeline only
  needs `RetrievedChunk` objects with an `article_id`, a `score` in 0..1, and text.
- **Real thresholds**: the ones in `config.py` are guesses. Label a few hundred resolved
  tickets, run the ladder over them, and put `auto_send` where the send-when-wrong rate
  crosses whatever your team can live with. The `Resolution.trace` is built for exactly
  this.
- **Real handoff destination**: `HandoffPacket` is a typed object precisely so it can
  become a Zendesk internal note, a Linear issue, or a Slack blockquote.
  `render.py` does the markdown one.

## Notes on the API usage

Every stage uses structured outputs, so a stage either returns a validated Pydantic
object or raises — there is no "the model returned prose this time" branch anywhere in
the pipeline. On Anthropic that is
[`client.messages.parse(output_format=Model)`](https://platform.claude.com/docs/en/build-with-claude/structured-outputs);
on OpenAI it is `client.responses.parse(text_format=Model)`. Per-stage `effort` is set in
`config.py` (triage is shallow; critique and the handoff are not).

The two adapters differ in three places, all absorbed inside `llm.py` rather than leaked
into config:

| | Anthropic | OpenAI |
|---|---|---|
| Effort | `output_config={"effort": ...}` on every model | `reasoning={"effort": ...}`, and only for the reasoning families — sending it to a chat model is a 400 |
| Prompt caching | An explicit `cache_control` breakpoint on the long system prompt | Automatic on long prefixes; there is no breakpoint to place |
| Refusal / truncation | `stop_reason: "refusal"` / `"max_tokens"` | A `refusal` content part / `status: "incomplete"` |

All of it normalises onto one `LLMError`, so a refusal, a 500, a schema violation, and a
truncation are the same event to the ladder — this rung is unavailable, take the next one
down — whichever provider produced it. Adding a third provider is an adapter with two
methods plus one entry in `PROVIDERS`; a name in `config.PROVIDER_MODELS` with no client
behind it fails at import rather than on a ticket.
