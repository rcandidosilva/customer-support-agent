"""Which model runs a stage, and what happens when that model misbehaves.

Two things are being pinned down here.  First, the resolution order for model config -
default, provider, per-stage override, environment, flags - because it is the part a
person changes under pressure and the part that fails silently if it is wrong.  Second,
that the OpenAI adapter turns every provider-specific failure into the one
:class:`LLMError` the ladder already survives, so a mixed route degrades the same way a
single-provider one does.

No SDK is installed for the OpenAI tests.  The client is injected, and where the error
classification itself is under test a stub module stands in for ``openai`` - the real
hierarchy is three exception classes deep and that is all the adapter reads.
"""

from __future__ import annotations

import sys
import types

import pytest
from conftest import make_classification, make_critique, make_draft, make_ticket

from support_agent.config import (
    PROVIDER_MODELS,
    ModelRef,
    Settings,
    UnknownProvider,
)
from support_agent.ladder import Ladder
from support_agent.llm import (
    PROVIDERS,
    LLMError,
    MissingCredentials,
    OpenAILLM,
    RoutedLLM,
    ScriptedLLM,
    build_llm,
    new_openai_client,
)
from support_agent.models import DraftAnswer

# --------------------------------------------------------------------------------------
# Model references
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("claude-opus-5", ("anthropic", "claude-opus-5")),
        ("openai:gpt-5", ("openai", "gpt-5")),
        ("anthropic:claude-opus-5", ("anthropic", "claude-opus-5")),
        ("  openai : gpt-5-mini  ", ("openai", "gpt-5-mini")),
        # A bare provider name means "that provider's default", which is what makes
        # `--provider openai` sufficient on its own.
        ("openai", ("openai", PROVIDER_MODELS["openai"])),
    ],
)
def test_a_model_reference_parses_to_a_provider_and_a_model(ref, expected):
    parsed = ModelRef.parse(ref)
    assert (parsed.provider, parsed.model) == expected


def test_a_bare_id_belongs_to_the_configured_provider_not_the_default_one():
    """The point of `provider`: after switching, bare ids stop meaning Anthropic."""
    assert ModelRef.parse("gpt-5-mini", default_provider="openai").provider == "openai"


@pytest.mark.parametrize("ref", ["", "   ", "openai:", ":gpt-5"])
def test_an_unusable_model_reference_is_rejected(ref):
    with pytest.raises(ValueError):
        ModelRef.parse(ref)


def test_an_unknown_provider_names_the_ones_that_exist():
    """A typo in one env var should not need a source dive to diagnose."""
    with pytest.raises(UnknownProvider) as exc:
        ModelRef.parse("anthropicc:claude-opus-5")
    assert "anthropic" in str(exc.value) and "openai" in str(exc.value)


def test_every_configured_provider_has_a_client():
    """config.py names the providers; llm.py implements them. They must agree."""
    assert set(PROVIDERS) == set(PROVIDER_MODELS)


# --------------------------------------------------------------------------------------
# Config resolution
# --------------------------------------------------------------------------------------


def test_a_stage_with_no_opinion_runs_on_the_default_model():
    settings = Settings()
    assert settings.model_ref("critique") == settings.model_ref()


def test_a_qualified_model_moves_the_provider_with_it():
    """`--model openai:gpt-5` has to be enough; a model id without its provider is a 404."""
    settings = Settings().with_model("openai:gpt-5")
    assert (settings.provider, settings.model) == ("openai", "gpt-5")


def test_switching_provider_brings_that_providers_default_model():
    settings = Settings().with_provider("openai")
    assert settings.model_ref().model == PROVIDER_MODELS["openai"]


def test_switching_provider_rejects_a_name_with_no_client():
    with pytest.raises(UnknownProvider):
        Settings().with_provider("bedrock")


def test_a_stage_override_beats_the_default_and_leaves_the_others_alone():
    settings = Settings().with_stage_model("critique", "openai:gpt-5")
    assert settings.model_ref("critique") == ModelRef("openai", "gpt-5")
    assert settings.model_ref("draft") == settings.model_ref()


def test_a_stage_override_is_validated_when_it_is_set():
    """Not at first call: a mistyped ref must not surface mid-ticket."""
    with pytest.raises(UnknownProvider):
        Settings().with_stage_model("critique", "openai2:gpt-5")


def test_the_environment_configures_a_mixed_route(monkeypatch):
    monkeypatch.setenv("SUPPORT_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("SUPPORT_AGENT_MODEL", "gpt-5-mini")
    monkeypatch.setenv("SUPPORT_AGENT_MODEL_HANDOFF", "anthropic:claude-opus-5")
    settings = Settings.from_env()

    assert settings.model_ref("classify") == ModelRef("openai", "gpt-5-mini")
    assert settings.model_ref("handoff") == ModelRef("anthropic", "claude-opus-5")


def test_the_environment_default_is_unchanged_when_nothing_is_set(monkeypatch):
    for name in ("SUPPORT_AGENT_PROVIDER", "SUPPORT_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    default = ModelRef("anthropic", PROVIDER_MODELS["anthropic"])
    assert Settings.from_env().model_ref() == default


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------


class FakeClient:
    """Records what it was asked for. Stands in for a live adapter in `build_llm`."""

    def __init__(self, ref: ModelRef) -> None:
        self.ref = str(ref)
        self.calls: list[str] = []

    def structured(self, *, stage, system, user, schema, effort="medium",
                   max_tokens=4096):
        self.calls.append(stage)
        return schema(), None

    def text(self, *, stage, system, user, effort="medium", max_tokens=4096):
        self.calls.append(stage)
        return self.ref, None


def test_a_single_model_configuration_gets_no_routing_layer():
    """The common case should not pay for a feature it is not using."""
    llm = build_llm(Settings(), factory=FakeClient)
    assert isinstance(llm, FakeClient)


def test_a_mixed_configuration_gets_a_router():
    settings = Settings().with_stage_model("classify", "openai:gpt-5-mini")
    llm = build_llm(settings, factory=FakeClient)
    assert isinstance(llm, RoutedLLM)
    assert llm.routing() == {
        "default": "anthropic:claude-opus-5",
        "classify": "openai:gpt-5-mini",
    }


def test_stages_naming_the_same_model_share_one_client():
    """Otherwise a five-stage route opens five connection pools to the same endpoint."""
    settings = (
        Settings()
        .with_stage_model("classify", "openai:gpt-5-mini")
        .with_stage_model("clarify", "openai:gpt-5-mini")
    )
    llm = build_llm(settings, factory=FakeClient)
    assert llm.for_stage("classify") is llm.for_stage("clarify")


def test_the_router_sends_each_stage_to_its_own_model():
    settings = Settings().with_stage_model("critique", "openai:gpt-5")
    llm = build_llm(settings, factory=FakeClient)

    llm.text(stage="critique", system="s", user="u")
    llm.text(stage="draft", system="s", user="u")

    assert llm.for_stage("critique").calls == ["critique"]
    assert llm.for_stage("draft").calls == ["draft"]


def test_the_router_falls_back_to_the_default_for_an_unconfigured_stage():
    settings = Settings().with_stage_model("critique", "openai:gpt-5")
    llm = build_llm(settings, factory=FakeClient)
    assert llm.for_stage("a-stage-nobody-configured") is llm.default


def test_every_model_on_the_route_is_built_up_front():
    """A route whose handoff model has no key must fail now, not on the first escalation.

    Escalation is the rung that runs last and least often, so a lazily built client is a
    landmine: the run that finds it is the one that was already going badly.
    """
    def factory(ref: ModelRef):
        if ref.provider == "openai":
            raise MissingCredentials("no OpenAI credentials found; set OPENAI_API_KEY")
        return FakeClient(ref)

    settings = Settings().with_stage_model("handoff", "openai:gpt-5")
    with pytest.raises(MissingCredentials):
        build_llm(settings, factory=factory)


def test_the_trace_says_which_model_answered_each_stage(kb, settings):
    """The payoff. Mixed-route token counts are only costable if they are attributed."""
    triage = ScriptedLLM(responses={"classify": make_classification()},
                         ref="openai:gpt-5-mini")
    rest = ScriptedLLM(
        responses={"draft": make_draft(), "critique": make_critique()},
        ref="anthropic:claude-opus-5",
    )
    llm = RoutedLLM(rest, {"classify": triage})

    resolution = Ladder(llm, kb, settings).run(make_ticket())
    by_stage = {row.stage: row.model for row in resolution.trace}

    assert by_stage["classify"] == "openai:gpt-5-mini"
    assert by_stage["draft"] == "anthropic:claude-opus-5"


def test_the_router_reports_the_calls_its_clients_made(kb, settings):
    """`calls` is how tests assert what a stage was actually sent; routing keeps it."""
    triage = ScriptedLLM(responses={"classify": make_classification()})
    rest = ScriptedLLM(responses={"draft": make_draft(), "critique": make_critique()})
    llm = RoutedLLM(rest, {"classify": triage})

    Ladder(llm, kb, settings).run(make_ticket())
    assert {call.stage for call in llm.calls} == {"classify", "draft", "critique"}


# --------------------------------------------------------------------------------------
# The OpenAI adapter
# --------------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, *, parsed=None, output_text="", status="completed",
                 incomplete_reason=None, refusal=None, usage=(120, 34)):
        self.output_parsed = parsed
        self.output_text = output_text
        self.status = status
        self.incomplete_details = (
            types.SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        )
        self.output = (
            [types.SimpleNamespace(
                content=[types.SimpleNamespace(type="refusal", refusal=refusal)]
            )]
            if refusal
            else []
        )
        self.usage = types.SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1])


class FakeResponses:
    """The two Responses API methods the adapter uses, and the kwargs it passed."""

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.kwargs: dict = {}

    def _answer(self, **kwargs):
        self.kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return self.response

    parse = _answer
    create = _answer


def fake_openai(responses: FakeResponses):
    return types.SimpleNamespace(responses=responses)


def test_a_structured_stage_returns_a_validated_object_and_its_usage():
    draft = make_draft()
    responses = FakeResponses(FakeResponse(parsed=draft))
    llm = OpenAILLM("gpt-5", fake_openai(responses))

    result, usage = llm.structured(
        stage="draft", system="sys", user="usr", schema=DraftAnswer, max_tokens=6000
    )

    assert result is draft
    assert (usage.input_tokens, usage.output_tokens) == (120, 34)
    # Qualified, so a trace row from a mixed route is unambiguous.
    assert usage.model == "openai:gpt-5"
    assert responses.kwargs["model"] == "gpt-5"
    assert responses.kwargs["instructions"] == "sys"
    assert responses.kwargs["text_format"] is DraftAnswer
    assert responses.kwargs["max_output_tokens"] == 6000


def test_a_text_stage_returns_the_output_text():
    responses = FakeResponses(FakeResponse(output_text="  a brief  "))
    llm = OpenAILLM("gpt-5", fake_openai(responses))
    text, usage = llm.text(stage="handoff", system="sys", user="usr")
    assert text == "a brief"
    assert usage.model == "openai:gpt-5"


def test_effort_is_sent_as_reasoning_effort_to_a_reasoning_model():
    responses = FakeResponses(FakeResponse(output_text="ok"))
    OpenAILLM("gpt-5", fake_openai(responses)).text(
        stage="critique", system="s", user="u", effort="high"
    )
    assert responses.kwargs["reasoning"] == {"effort": "high"}


def test_effort_is_dropped_for_a_model_that_would_reject_it():
    """Sending `reasoning` to a chat model is a 400, which would cost the stage."""
    responses = FakeResponses(FakeResponse(output_text="ok"))
    OpenAILLM("gpt-4o", fake_openai(responses)).text(
        stage="critique", system="s", user="u", effort="high"
    )
    assert "reasoning" not in responses.kwargs


def test_reasoning_can_be_forced_for_a_model_the_guess_has_not_caught_up_with():
    responses = FakeResponses(FakeResponse(output_text="ok"))
    OpenAILLM("some-new-model", fake_openai(responses), reasoning=True).text(
        stage="critique", system="s", user="u", effort="low"
    )
    assert responses.kwargs["reasoning"] == {"effort": "low"}


def test_a_refusal_is_an_unavailable_rung_not_a_crash():
    responses = FakeResponses(FakeResponse(parsed=make_draft(), refusal="policy"))
    llm = OpenAILLM("gpt-5", fake_openai(responses))
    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert "declined" in exc.value.detail and not exc.value.retryable


def test_a_truncated_response_is_retryable():
    """Same classification the Anthropic path gives `stop_reason: max_tokens`."""
    responses = FakeResponses(
        FakeResponse(status="incomplete", incomplete_reason="max_output_tokens")
    )
    llm = OpenAILLM("gpt-5", fake_openai(responses))
    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert exc.value.retryable


def test_an_incomplete_response_for_any_other_reason_is_not_retryable():
    responses = FakeResponses(
        FakeResponse(status="incomplete", incomplete_reason="content_filter")
    )
    llm = OpenAILLM("gpt-5", fake_openai(responses))
    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert "content_filter" in exc.value.detail and not exc.value.retryable


def test_a_missing_parsed_output_is_an_llm_error():
    llm = OpenAILLM("gpt-5", fake_openai(FakeResponses(FakeResponse(parsed=None))))
    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert "DraftAnswer" in exc.value.detail


def test_an_empty_text_response_is_an_llm_error():
    llm = OpenAILLM("gpt-5", fake_openai(FakeResponses(FakeResponse(output_text="  "))))
    with pytest.raises(LLMError):
        llm.text(stage="handoff", system="s", user="u")


def test_a_dict_shaped_parsed_output_is_validated_rather_than_trusted():
    responses = FakeResponses(FakeResponse(parsed={"reply": "hi"}))
    llm = OpenAILLM("gpt-5", fake_openai(responses))
    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert "schema violation" in exc.value.detail


# -- SDK failures ----------------------------------------------------------------------


def install_stub_sdk(monkeypatch):
    """A stand-in `openai` module with the exception hierarchy the adapter reads.

    The adapter only ever asks "which of these four is it", so four classes in the real
    inheritance order are a faithful stub - and testing the mapping is worth more than
    an install.
    """
    class OpenAIError(Exception):
        def __init__(self, message="", status_code=500):
            super().__init__(message)
            self.message = message
            self.status_code = status_code

    class APIConnectionError(OpenAIError):
        pass

    class APIStatusError(OpenAIError):
        pass

    class RateLimitError(APIStatusError):
        pass

    stub = types.ModuleType("openai")
    stub.OpenAIError = OpenAIError
    stub.APIConnectionError = APIConnectionError
    stub.APIStatusError = APIStatusError
    stub.RateLimitError = RateLimitError
    monkeypatch.setitem(sys.modules, "openai", stub)
    return stub


@pytest.mark.parametrize(
    "make,detail,retryable",
    [
        (lambda sdk: sdk.RateLimitError("slow down", 429), "rate limited", True),
        (lambda sdk: sdk.APIStatusError("bad request", 400), "api error 400", False),
        (lambda sdk: sdk.APIStatusError("upstream", 503), "api error 503", True),
        (lambda sdk: sdk.APIConnectionError("dns"), "connection error", True),
        (lambda sdk: sdk.OpenAIError("something else"), "sdk error", False),
    ],
)
def test_sdk_failures_become_llm_errors_classified_by_whether_a_retry_could_help(
    monkeypatch, make, detail, retryable
):
    sdk = install_stub_sdk(monkeypatch)
    llm = OpenAILLM("gpt-5", fake_openai(FakeResponses(raises=make(sdk))))

    with pytest.raises(LLMError) as exc:
        llm.structured(stage="draft", system="s", user="u", schema=DraftAnswer)
    assert detail in exc.value.detail
    assert exc.value.retryable is retryable


def test_a_missing_sdk_says_how_to_install_it(monkeypatch):
    # `None` in sys.modules is how the import system spells "definitively absent", so
    # this test says the same thing whether or not the SDK happens to be installed.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError) as exc:
        new_openai_client()
    assert "support-agent[openai]" in str(exc.value)


def test_a_missing_key_is_a_startup_failure_not_a_degraded_ticket(monkeypatch):
    sdk = install_stub_sdk(monkeypatch)

    def no_key():
        raise sdk.OpenAIError("api_key must be set")

    sdk.OpenAI = no_key
    with pytest.raises(MissingCredentials) as exc:
        new_openai_client()
    assert "OPENAI_API_KEY" in str(exc.value)
