"""The one place in this project that talks to Claude.

Every stage goes through :class:`LLMClient`, which exists for three reasons:

1. **Graceful degradation.** A stage that cannot get a well-formed answer raises
   :class:`LLMError`, and the ladder catches it and drops to a lower rung.  Nothing in
   ``stages/`` handles HTTP.
2. **Accounting.** Token usage per stage is recorded on the trace, so the cost of a
   deflection is a number you can look at rather than a vibe.
3. **Testability.** :class:`ScriptedLLM` implements the same interface with no network,
   which is how the whole ladder is tested end to end without an API key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

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


@dataclass
class Call:
    """A recorded call, for assertions in tests and for the ``--trace`` output."""

    stage: str
    system: str
    user: str
    schema: str = ""


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

    def __init__(self, model: str, client=None) -> None:
        self.model = model
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

    @staticmethod
    def _usage(response, started: float) -> Usage:
        usage = getattr(response, "usage", None)
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

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
        return value, Usage(input_tokens=100, output_tokens=50, duration_ms=1.0)

    def text(self, *, stage, system, user, effort="medium", max_tokens=4096):
        self.calls.append(Call(stage, system, user))
        return str(self._next(stage)), Usage(input_tokens=100, output_tokens=50,
                                             duration_ms=1.0)

    def stages_called(self) -> list[str]:
        return [c.stage for c in self.calls]
