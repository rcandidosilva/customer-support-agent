"""The one place in this project that talks to a model provider.

Every stage goes through :class:`LLMClient`, which exists for four reasons:

1. **Graceful degradation.** A stage that cannot get a well-formed answer raises
   :class:`LLMError`, and the ladder catches it and drops to a lower rung.  Nothing in
   ``stages/`` handles HTTP.
2. **Accounting.** Token usage per stage is recorded on the trace, so the cost of a
   deflection is a number you can look at rather than a vibe.
3. **Testability.** :class:`ScriptedLLM` implements the same interface with no network,
   which is how the whole ladder is tested end to end without an API key.
4. **Provider independence.** :class:`AnthropicLLM` and :class:`OpenAILLM` implement the
   same two methods, and :class:`RoutedLLM` picks between them per stage from config.
   Both adapters normalise their SDK's failures onto :class:`LLMError`, and their
   refusals and truncations onto it too, so the ladder's degradation paths do not know
   or care which provider answered.

:func:`build_llm` is the entry point: hand it a :class:`~support_agent.config.Settings`
and it returns the one client, or the router, that configuration describes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import PROVIDER_MODELS, ModelRef, Settings

T = TypeVar("T", bound=BaseModel)


class MissingCredentials(RuntimeError):
    """No API credentials are configured.  A startup problem, not a stage failure."""


class LLMError(RuntimeError):
    """A stage could not get a usable answer out of the model.

    Raised for transport failures, schema violations, and refusals alike: from the
    ladder's point of view these are the same event - this rung is unavailable, take the
    next one down.
    """

    def __init__(self, stage: str, detail: str, *, retryable: bool = False) -> None:
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail
        self.retryable = retryable


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    #: The qualified model reference that did the work, as ``provider:model``.  Recorded
    #: because with per-stage routing the tokens on a trace row are otherwise
    #: unattributable - and unattributable tokens cannot be costed.
    model: str = ""


@dataclass
class Call:
    """A recorded call, for assertions in tests and for the ``--trace`` output."""

    stage: str
    system: str
    user: str
    schema: str = ""


def _usage_of(response, started: float, model: str) -> Usage:
    """Token counts off a response, defensively.

    Both SDKs spell the fields ``usage.input_tokens`` / ``usage.output_tokens``, so one
    reader serves both; anything missing counts as zero rather than raising, because a
    stage that succeeded must not fail on its own bookkeeping.
    """
    usage = getattr(response, "usage", None)
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        duration_ms=(time.perf_counter() - started) * 1000,
        model=model,
    )


class LLMClient(Protocol):
    def structured(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[T, Usage]: ...

    def text(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[str, Usage]: ...


# --------------------------------------------------------------------------------------
# Live implementation
# --------------------------------------------------------------------------------------


class AnthropicLLM:
    """Claude-backed client using structured outputs for every schema'd stage."""

    provider = "anthropic"

    def __init__(self, model: str = PROVIDER_MODELS["anthropic"], client=None) -> None:
        self.model = model
        self.ref = str(ModelRef(self.provider, model))
        if client is None:
            import anthropic  # imported lazily so tests need no SDK install

            client = anthropic.Anthropic()
            # The SDK resolves credentials lazily, at request time.  Finding out on the
            # first stage of the first ticket means every stage then fails and every
            # ticket "gracefully degrades" into a useless packet - degradation is for
            # transient faults, not for a missing key.  Fail loudly here instead.
            if not (client.api_key or client.auth_token or client.credentials):
                raise MissingCredentials(
                    "no Anthropic credentials found; set ANTHROPIC_API_KEY"
                )
        self._client = client
        self.calls: list[Call] = []

    # -- internals ---------------------------------------------------------------------

    def _usage(self, response, started: float) -> Usage:
        return _usage_of(response, started, self.ref)

    def _wrap_api_errors(self, stage: str, fn):
        import anthropic

        try:
            return fn()
        except anthropic.RateLimitError as exc:
            raise LLMError(stage, f"rate limited: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise LLMError(
                stage, f"api error {exc.status_code}: {exc.message}", retryable=retryable
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(stage, f"connection error: {exc}", retryable=True) from exc
        except anthropic.AnthropicError as exc:
            # Anything else the SDK raises - malformed request, validation failure.
            # Still just an unavailable rung as far as the ladder is concerned.
            raise LLMError(stage, f"sdk error: {exc}") from exc

    @staticmethod
    def _check_stop_reason(stage: str, response) -> None:
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(stage, f"model declined ({category})")
        if stop == "max_tokens":
            raise LLMError(stage, "output truncated at max_tokens", retryable=True)

    # -- interface ---------------------------------------------------------------------

    def structured(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[T, Usage]:
        self.calls.append(Call(stage, system, user, schema.__name__))
        started = time.perf_counter()

        response = self._wrap_api_errors(
            stage,
            lambda: self._client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config={"effort": effort},
            ),
        )
        self._check_stop_reason(stage, response)

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise LLMError(stage, f"no parsed output for {schema.__name__}")
        if not isinstance(parsed, schema):
            try:
                parsed = schema.model_validate(parsed)
            except ValidationError as exc:
                raise LLMError(
                    stage, f"schema violation: {exc.error_count()} errors"
                ) from exc
        return parsed, self._usage(response, started)

    def text(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[str, Usage]:
        self.calls.append(Call(stage, system, user))
        started = time.perf_counter()

        response = self._wrap_api_errors(
            stage,
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            ),
        )
        self._check_stop_reason(stage, response)

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise LLMError(stage, "empty response")
        return text, self._usage(response, started)


#: Model families that accept a reasoning effort.  Used only as a fallback guess when
#: the caller does not say: effort is silently ignored by the chat models, and sending it
#: to one is a 400, so the wrong guess in either direction costs a stage.  Pass
#: ``reasoning=`` explicitly to :class:`OpenAILLM` for a model this list has not caught
#: up with.
REASONING_MODELS = ("o1", "o3", "o4", "gpt-5")


def _openai_sdk(*, required: bool = True):
    """The OpenAI SDK module, or ``None`` when it is not installed.

    ``required=False`` is for the error-classification path, which can only be reached
    without the SDK by an injected client - a test - and has nothing to classify there.
    """
    try:
        import openai
    except ImportError as exc:
        if not required:
            return None
        # Re-raised with the fix in it: ``openai`` is an optional dependency precisely
        # so that the default provider does not pull it in.
        raise ImportError(
            "the openai provider needs the OpenAI SDK: "
            "pip install 'support-agent[openai]'"
        ) from exc
    return openai


def new_openai_client():
    """A credentialed OpenAI client, or a loud failure.

    Separate from :class:`OpenAILLM` so that the two ways this goes wrong - no SDK, no
    key - are reachable in a test without constructing a live client.
    """
    openai = _openai_sdk()
    try:
        # Unlike the Anthropic SDK, this one resolves credentials eagerly and raises on
        # construction.  Same outcome either way: a missing key is a startup failure,
        # not something to degrade every ticket over.
        return openai.OpenAI()
    except openai.OpenAIError as exc:
        raise MissingCredentials(
            f"no OpenAI credentials found; set OPENAI_API_KEY ({exc})"
        ) from exc


class OpenAILLM:
    """OpenAI-backed client, same two methods, same failure vocabulary.

    Structured stages go through the Responses API's ``parse`` helper, which enforces the
    schema server-side exactly as the Anthropic path does - so a stage still either gets
    a validated Pydantic object or an :class:`LLMError`, and ``stages/`` cannot tell the
    two providers apart.

    Two provider differences are absorbed here rather than leaking into config:

    * **Effort.** Only the reasoning families take one, so it is passed as
      ``reasoning.effort`` for those and dropped for the rest.
    * **Caching.** There is no breakpoint to place; long prefixes are cached
      automatically, which is why the system prompt is passed plainly where the
      Anthropic path marks it.
    """

    provider = "openai"

    def __init__(
        self,
        model: str = PROVIDER_MODELS["openai"],
        client=None,
        *,
        reasoning: bool | None = None,
    ) -> None:
        self.model = model
        self.ref = str(ModelRef(self.provider, model))
        self.reasoning = (
            model.startswith(REASONING_MODELS) if reasoning is None else reasoning
        )
        self._client = client if client is not None else new_openai_client()
        self.calls: list[Call] = []

    # -- internals ---------------------------------------------------------------------

    def _usage(self, response, started: float) -> Usage:
        return _usage_of(response, started, self.ref)

    def _params(self, effort: str, max_tokens: int) -> dict:
        params = {"max_output_tokens": max_tokens}
        if self.reasoning:
            params["reasoning"] = {"effort": effort}
        return params

    def _wrap_api_errors(self, stage: str, fn):
        openai = _openai_sdk(required=False)
        if openai is None:
            # No SDK, so an injected client: there are no SDK exceptions to classify.
            return fn()

        try:
            return fn()
        except openai.RateLimitError as exc:
            raise LLMError(stage, f"rate limited: {exc}", retryable=True) from exc
        except openai.APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise LLMError(
                stage, f"api error {exc.status_code}: {exc.message}", retryable=retryable
            ) from exc
        except openai.APIConnectionError as exc:
            # Timeouts are a subclass of this, and both are worth retrying.
            raise LLMError(stage, f"connection error: {exc}", retryable=True) from exc
        except openai.OpenAIError as exc:
            raise LLMError(stage, f"sdk error: {exc}") from exc

    @staticmethod
    def _check_response(stage: str, response) -> None:
        """The counterpart of the Anthropic stop-reason check.

        A refusal here is a content part rather than a stop reason, and a truncation is
        an ``incomplete`` status; both are mapped onto the same :class:`LLMError` the
        ladder already survives.
        """
        for item in getattr(response, "output", None) or ():
            for part in getattr(item, "content", None) or ():
                if getattr(part, "type", None) == "refusal":
                    detail = getattr(part, "refusal", "") or "unspecified"
                    raise LLMError(stage, f"model declined ({detail})")

        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unspecified"
            if reason == "max_output_tokens":
                raise LLMError(stage, "output truncated at max_tokens", retryable=True)
            raise LLMError(stage, f"incomplete response ({reason})")

    # -- interface ---------------------------------------------------------------------

    def structured(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[T, Usage]:
        self.calls.append(Call(stage, system, user, schema.__name__))
        started = time.perf_counter()

        response = self._wrap_api_errors(
            stage,
            lambda: self._client.responses.parse(
                model=self.model,
                instructions=system,
                input=user,
                text_format=schema,
                **self._params(effort, max_tokens),
            ),
        )
        self._check_response(stage, response)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise LLMError(stage, f"no parsed output for {schema.__name__}")
        if not isinstance(parsed, schema):
            try:
                parsed = schema.model_validate(parsed)
            except ValidationError as exc:
                raise LLMError(
                    stage, f"schema violation: {exc.error_count()} errors"
                ) from exc
        return parsed, self._usage(response, started)

    def text(
        self,
        *,
        stage: str,
        system: str,
        user: str,
        effort: str = "medium",
        max_tokens: int = 4096,
    ) -> tuple[str, Usage]:
        self.calls.append(Call(stage, system, user))
        started = time.perf_counter()

        response = self._wrap_api_errors(
            stage,
            lambda: self._client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                **self._params(effort, max_tokens),
            ),
        )
        self._check_response(stage, response)

        text = (getattr(response, "output_text", "") or "").strip()
        if not text:
            raise LLMError(stage, "empty response")
        return text, self._usage(response, started)


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------

#: Provider name -> adapter.  The one place a new provider is registered.
PROVIDERS: dict[str, Callable[[str], object]] = {
    "anthropic": AnthropicLLM,
    "openai": OpenAILLM,
}

# config.py owns the provider names and their default models; this module owns the
# clients.  Kept honest here rather than discovered by a ticket routed to a provider
# nothing implements.
assert set(PROVIDERS) == set(PROVIDER_MODELS), (
    f"providers without a client: {sorted(set(PROVIDER_MODELS) - set(PROVIDERS))}"
)


def client_for(ref: ModelRef) -> LLMClient:
    """The live client for one :class:`~support_agent.config.ModelRef`."""
    return PROVIDERS[ref.provider](ref.model)


class RoutedLLM:
    """Sends each stage to the model its config names.

    Which is all "multi-model" has to mean: the stage code is unchanged, the schemas are
    unchanged, and the choice of engine per stage is a table.  That makes the useful
    arrangement cheap to express - triage and clarification on a small fast model, the
    critique and the handoff brief on the strongest one available - and reversible
    without a deploy.

    A stage with no entry runs on ``default``, so a partially configured route is a
    working route.
    """

    def __init__(self, default: LLMClient, per_stage: dict[str, LLMClient]) -> None:
        self.default = default
        # Only genuine overrides are kept, so ``routing()`` reports the stages that
        # actually differ rather than every stage in the config.
        self.per_stage = {
            stage: client for stage, client in per_stage.items() if client is not default
        }

    def for_stage(self, stage: str) -> LLMClient:
        return self.per_stage.get(stage, self.default)

    @property
    def calls(self) -> list[Call]:
        """Every recorded call, grouped by client rather than in wall-clock order.

        Enough for the assertions tests make on it ("was this stage called, with what"),
        and the trace is the thing to read for ordering.
        """
        return [
            call
            for client in (self.default, *self.per_stage.values())
            for call in getattr(client, "calls", ())
        ]

    def routing(self) -> dict[str, str]:
        """``{"default": ref, stage: ref}`` - what will run where, for logging."""
        routing = {"default": getattr(self.default, "ref", "")}
        routing.update(
            {stage: getattr(c, "ref", "") for stage, c in self.per_stage.items()}
        )
        return routing

    def structured(self, *, stage, system, user, schema, effort="medium",
                   max_tokens=4096):
        return self.for_stage(stage).structured(
            stage=stage, system=system, user=user, schema=schema,
            effort=effort, max_tokens=max_tokens,
        )

    def text(self, *, stage, system, user, effort="medium", max_tokens=4096):
        return self.for_stage(stage).text(
            stage=stage, system=system, user=user, effort=effort, max_tokens=max_tokens,
        )


def build_llm(
    settings: Settings | None = None,
    *,
    factory: Callable[[ModelRef], LLMClient] = client_for,
) -> LLMClient:
    """The client this configuration describes.

    Returns a single adapter when every stage resolves to the same model - the common
    case, and worth keeping free of a routing layer - and a :class:`RoutedLLM` when they
    do not.  Either way the return value satisfies :class:`LLMClient`, so nothing
    downstream branches on which it got.

    Every distinct model is built here, eagerly, which is what keeps a missing key a
    startup failure: a route that sends the handoff brief to a provider with no
    credentials should fail now, not on the first ticket that escalates.

    ``factory`` exists for the tests, which need routing behaviour without an SDK.
    """
    settings = settings or Settings()
    clients: dict[str, LLMClient] = {}

    def get(ref: ModelRef) -> LLMClient:
        # Keyed by the qualified ref so two stages naming the same model share one
        # client, and so identity is a usable test for "same model" in RoutedLLM.
        return clients.setdefault(str(ref), factory(ref))

    default = get(settings.model_ref())
    per_stage = {stage: get(ref) for stage, ref in settings.model_refs().items()}
    if all(client is default for client in per_stage.values()):
        return default
    return RoutedLLM(default, per_stage)


# --------------------------------------------------------------------------------------
# Test double
# --------------------------------------------------------------------------------------


class UnavailableLLM:
    """A client that reports every stage as unavailable, using the normal failure path.

    Used where a model *might* be needed but no credentials are configured - answering a
    review, for instance, needs one only if the verdict sends the ticket to a person and
    a brief has to be written.

    It raises :class:`LLMError` rather than anything else on purpose: that is the event
    the ladder already knows how to survive, so a missing key degrades into a
    deterministically assembled packet instead of a traceback.
    """

    def __init__(self, reason: str = "no API credentials are configured") -> None:
        self.reason = reason

    def structured(self, *, stage, system, user, schema, effort="medium", max_tokens=4096):
        raise LLMError(stage, self.reason)

    def text(self, *, stage, system, user, effort="medium", max_tokens=4096):
        raise LLMError(stage, self.reason)


@dataclass
class ScriptedLLM:
    """Returns canned objects per stage; raises :class:`LLMError` where told to.

    ``responses`` maps a stage name to either a value or a list of values consumed in
    order.  A value that is an exception instance is raised instead of returned, which
    is how the degradation paths are tested.
    """

    responses: dict[str, object] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    #: What the trace should say ran this stage.  Settable so a test can assert that
    #: per-stage routing reached the client it was supposed to reach.
    ref: str = "scripted"

    def _next(self, stage: str):
        if stage not in self.responses:
            raise AssertionError(f"ScriptedLLM has no response for stage {stage!r}")
        slot = self.responses[stage]
        if isinstance(slot, list):
            if not slot:
                raise AssertionError(f"ScriptedLLM exhausted responses for {stage!r}")
            value = slot.pop(0)
        else:
            value = slot
        if isinstance(value, Exception):
            raise value
        return value

    def structured(self, *, stage, system, user, schema, effort="medium", max_tokens=4096):
        self.calls.append(Call(stage, system, user, schema.__name__))
        value = self._next(stage)
        if not isinstance(value, schema):
            value = schema.model_validate(value)
        return value, self._usage()

    def text(self, *, stage, system, user, effort="medium", max_tokens=4096):
        self.calls.append(Call(stage, system, user))
        return str(self._next(stage)), self._usage()

    def _usage(self) -> Usage:
        return Usage(input_tokens=100, output_tokens=50, duration_ms=1.0, model=self.ref)

    def stages_called(self) -> list[str]:
        return [c.stage for c in self.calls]
