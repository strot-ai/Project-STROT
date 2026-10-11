import json

import pytest

from strot.exceptions import FallbackException, RouterConfigurationError
from strot.router import (
    HostedApiStrategy,
    LocalEncoderStrategy,
    RouteDecision,
    Router,
    chunk_options,
    create_strategy,
)


class TestFactory:
    def test_default_resolves_to_local(self):
        strategy = create_strategy()
        assert isinstance(strategy, LocalEncoderStrategy)
        assert strategy.nlu_model == "SupersonicLabs/Julia-1"

    @pytest.mark.parametrize("name", ["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"])
    def test_provider_prefixes_resolve_to_hosted(self, name):
        assert isinstance(create_strategy(name), HostedApiStrategy)

    def test_hosted_requires_provider_slash_model(self):
        with pytest.raises(RouterConfigurationError):
            HostedApiStrategy("openai")


class TestOptionTreeSplitter:
    def test_small_lists_stay_flat(self):
        assert chunk_options(list("abc")) == [["a", "b", "c"]]

    def test_splits_beyond_twenty_into_balanced_groups(self):
        options = [f"opt_{i}" for i in range(45)]
        groups = chunk_options(options)
        assert all(len(g) <= 20 for g in groups)
        assert [o for g in groups for o in g] == options

    def test_hierarchical_selection_preserves_global_index(self):
        options = [f"opt_{i}" for i in range(25)]

        class FakeLocal(LocalEncoderStrategy):
            def _select_flat(self, context, question, opts):
                # Always pick the last option of whatever tier it is given.
                return RouteDecision(
                    index=len(opts) - 1,
                    option=opts[-1],
                    confidence=0.9,
                    distribution={},
                )

        decision = FakeLocal().select("ctx", "which?", options)
        assert decision.option == "opt_24"
        assert decision.index == 24
        assert decision.confidence == pytest.approx(0.81)  # product of two tiers


class TestJulia1PromptFormat:
    def test_token_format_matches_spec(self):
        prompt = LocalEncoderStrategy.format_prompt("ctx", "which?", ["a", "b"])
        assert prompt == "Context: ctx\nQuestion: which?\nOptions: (1) a (2) b\nAnswer:"


class _FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        class _Msg:
            content = json.dumps(self.payload)

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeClient:
    def __init__(self, payload):
        self.chat = type("chat", (), {"completions": _FakeCompletions(payload)})()


class TestHostedStrategy:
    def make(self, payload):
        client = _FakeClient(payload)
        strategy = HostedApiStrategy("openai/gpt-4o-mini", client_factory=lambda: client)
        return strategy, client

    def test_selects_index_from_structured_output(self):
        strategy, client = self.make({"selected_index": 1, "confidence": 0.92})
        decision = strategy.select("ctx", "which?", ["a", "b", "c"])
        assert decision.option == "b"
        assert decision.confidence == pytest.approx(0.92)
        sent = client.chat.completions.last_kwargs
        assert sent["model"] == "gpt-4o-mini"
        assert sent["response_format"]["json_schema"]["strict"] is True

    def test_out_of_range_index_triggers_fallback(self):
        strategy, _ = self.make({"selected_index": 9, "confidence": 0.99})
        with pytest.raises(FallbackException):
            strategy.select("ctx", "which?", ["a", "b"])

    def test_missing_env_key_is_a_configuration_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        strategy = HostedApiStrategy("openai/gpt-4o-mini")
        with pytest.raises(RouterConfigurationError):
            strategy.select("ctx", "which?", ["a"])


class TestConfidenceEnforcement:
    class StubStrategy(LocalEncoderStrategy):
        def __init__(self, confidence):
            super().__init__()
            self._confidence = confidence

        def select(self, context, question, options):
            return RouteDecision(0, options[0], self._confidence, {})

    def test_low_confidence_raises_fallback_instead_of_executing(self):
        router = Router(self.StubStrategy(confidence=0.39))
        with pytest.raises(FallbackException) as excinfo:
            router.route("ctx", "which?", ["a", "b"])
        assert excinfo.value.threshold == pytest.approx(0.40)
        assert excinfo.value.confidence == pytest.approx(0.39)

    def test_confident_decision_passes_through(self):
        decision = Router(self.StubStrategy(confidence=0.41)).route("ctx", "q", ["a"])
        assert decision.option == "a"
