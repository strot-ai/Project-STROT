"""Tests for strot.run(): the production JIT facade.

The compiled local artifact imports torch/transformers at module scope, so
these tests install a minimal pure-Python fake of both libraries that routes
by keyword overlap between the record context and the option descriptions —
enough to exercise the full compiled code path (route -> threshold ->
dispatch) without the heavyweight extras.
"""

import json
import math
import sys
import types

import pytest

import strot
from strot.exceptions import RouterConfigurationError, SchemaIntrospectionError
from strot.jit import as_records, cache_key, schema_fingerprint
from strot.schema import introspect_schema

SAMPLE = [
    {"ticket_id": 1, "body": "double billing charge", "tier": "pro"},
    {"ticket_id": 2, "body": "app crash defect", "tier": "free"},
]

REGISTRY = {
    "billing": "Billing disputes, charges, and refunds",
    "defect": "Crashes, bugs, and broken behavior",
}

QUERY = "Triage incoming tickets into the correct workflow."

VALID_PATHS = '''```python
def execute_billing(record: dict) -> dict:
    try:
        return {"queue": "billing", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "billing", "error": str(exc)}


def execute_defect(record: dict) -> dict:
    try:
        return {"queue": "engineering", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "engineering", "error": str(exc)}
```'''


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.responses.pop(0)


class ExplodingLLM:
    """Proves laziness: must never be invoked on a cache hit."""

    def __call__(self, prompt):  # pragma: no cover - the point is it never runs
        raise AssertionError("LLM was called despite a warm cache.")


# --------------------------------------------------------------------- #
# Fake torch / transformers: keyword-overlap router                      #
# --------------------------------------------------------------------- #


class _Vec:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        if isinstance(index, tuple):  # logits[0, -1, :]
            return self
        if isinstance(index, list):  # logits[option_token_ids]
            return _Vec([self.values[i] for i in index])
        return self.values[index]

    def __float__(self):
        return float(self.values[0])


class _Argmax:
    def __init__(self, index):
        self.index = index

    def item(self):
        return self.index


def _build_fake_torch():
    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda name: name

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch_mod.no_grad = _NoGrad

    def softmax(vec, dim=-1):
        exps = [math.exp(v) for v in vec.values]
        total = sum(exps)
        return _Vec([e / total for e in exps])

    torch_mod.softmax = softmax
    torch_mod.argmax = lambda vec: _Argmax(
        max(range(len(vec.values)), key=lambda i: vec.values[i])
    )
    return torch_mod


def _keyword_scores(prompt):
    """Score each numbered option by word overlap with the Context line."""
    context = prompt.split("Context: ", 1)[1].split("\nQuestion:", 1)[0].lower()
    options_blob = prompt.split("Options: ", 1)[1].split("\nAnswer:", 1)[0]
    import re

    option_texts = re.split(r"\(\d+\)\s*", options_blob)[1:]
    context_words = set(re.findall(r"[a-z]+", context))
    scores = []
    for text in option_texts:
        words = set(re.findall(r"[a-z]+", text.lower()))
        scores.append(float(len(words & context_words)))
    return scores


def _build_fake_transformers():
    transformers_mod = types.ModuleType("transformers")

    class _FakeTokenizer:
        last_prompt = ""

        def __call__(self, prompt, return_tensors=None):
            _FakeTokenizer.last_prompt = prompt
            return {}

        def encode(self, text, add_special_tokens=False):
            return [int(text) - 1]  # option "1" -> token id 0, etc.

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            scores = _keyword_scores(_FakeTokenizer.last_prompt)
            return _FakeOutput(_Vec([s * 4.0 for s in scores]))

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    transformers_mod.AutoTokenizer = _FakeTokenizer
    transformers_mod.AutoModelForCausalLM = _FakeModel
    return transformers_mod


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _build_fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", _build_fake_transformers())
    # Each test loads a freshly compiled artifact under a key-derived module
    # name; drop any compiled modules cached by earlier tests.
    for name in [n for n in sys.modules if n.startswith("strot_compiled_")]:
        del sys.modules[name]


# --------------------------------------------------------------------- #
# Record normalization and cache keys                                    #
# --------------------------------------------------------------------- #


class TestAsRecords:
    def test_list_of_dicts_passthrough(self):
        assert as_records(SAMPLE) == SAMPLE

    def test_dict_of_lists_becomes_records(self):
        records = as_records({"a": [1, 2], "b": ["x", "y"]})
        assert records == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_dataframe_like_uses_to_dict(self):
        class FakeFrame:
            def to_dict(self, orient):
                assert orient == "records"
                return [{"a": 1}]

        assert as_records(FakeFrame()) == [{"a": 1}]

    def test_unsupported_type_raises(self):
        with pytest.raises(SchemaIntrospectionError):
            as_records("not a dataset")

    def test_empty_raises(self):
        with pytest.raises(SchemaIntrospectionError):
            as_records([])


class TestCacheKey:
    def make_key(self, sample=SAMPLE, query=QUERY, registry=REGISTRY, nlu="SupersonicLabs/Julia-1"):
        schema = introspect_schema(sample)
        return cache_key(query, registry, nlu, schema, None, 0.40)

    def test_deterministic(self):
        assert self.make_key() == self.make_key()

    def test_query_change_busts_cache(self):
        assert self.make_key() != self.make_key(query="A different goal entirely.")

    def test_registry_change_busts_cache(self):
        bigger = dict(REGISTRY, refund="Refund processing")
        assert self.make_key() != self.make_key(registry=bigger)

    def test_schema_drift_busts_cache(self):
        drifted = [dict(r, severity="high") for r in SAMPLE]
        assert self.make_key() != self.make_key(sample=drifted)

    def test_sample_values_do_not_churn_key(self):
        other_batch = [
            {"ticket_id": 9, "body": "totally different words", "tier": "enterprise"},
            {"ticket_id": 10, "body": "yet another body", "tier": "pro"},
        ]
        assert self.make_key() == self.make_key(sample=other_batch)

    def test_fingerprint_ignores_field_order(self):
        reordered = [
            {"tier": r["tier"], "body": r["body"], "ticket_id": r["ticket_id"]}
            for r in SAMPLE
        ]
        fp_a = schema_fingerprint(introspect_schema(SAMPLE))
        fp_b = schema_fingerprint(introspect_schema(reordered))
        assert fp_a == fp_b


# --------------------------------------------------------------------- #
# The JIT loop: miss -> compile -> persist; hit -> execute only          #
# --------------------------------------------------------------------- #


class TestRun:
    def test_cache_miss_compiles_persists_and_executes(self, tmp_path, fake_runtime):
        llm = ScriptedLLM([VALID_PATHS])
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path)

        assert llm.calls == 1
        assert result.cache_hit is False
        assert result.attempts == 1
        assert result.artifact_path.exists()
        metadata = json.loads(
            result.artifact_path.with_suffix(".json").read_text()
        )
        assert metadata["cache_key"] == result.cache_key
        assert metadata["runtime_kind"] == "local"

        assert len(result.outcomes) == 2
        routed = {o["record"]["ticket_id"]: o["action"] for o in result.outcomes}
        assert routed == {1: "billing", 2: "defect"}

    def test_cache_hit_never_touches_the_llm(self, tmp_path, fake_runtime):
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=ExplodingLLM(), state=tmp_path)
        assert result.cache_hit is True
        assert result.attempts == 0
        assert len(result.outcomes) == 2

    def test_lazy_string_spec_not_resolved_on_hit(self, tmp_path, fake_runtime):
        strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        # openai is not installed here: resolving this spec would raise.
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm="openai/gpt-4o", state=tmp_path)
        assert result.cache_hit is True

    def test_schema_drift_triggers_one_recompile(self, tmp_path, fake_runtime):
        first = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        drifted = [dict(r, severity="high") for r in SAMPLE]
        second = strot.run(drifted, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        assert second.cache_hit is False
        assert second.cache_key != first.cache_key
        assert first.artifact_path.exists() and second.artifact_path.exists()

    def test_low_confidence_records_collect_as_fallbacks(self, tmp_path, fake_runtime):
        # Three actions so a zero-overlap record ties at ~0.33 < 0.40; with
        # only two, a tie sits at exactly 0.5 and would still dispatch.
        registry3 = dict(REGISTRY, lifecycle="Account deletion and data export requests")
        paths3 = VALID_PATHS[: VALID_PATHS.rindex("```")] + '''

def execute_lifecycle(record: dict) -> dict:
    try:
        return {"queue": "trust", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "trust", "error": str(exc)}
```'''
        with_unroutable = SAMPLE + [{"ticket_id": 3, "body": "zzz qqq", "tier": None}]
        result = strot.run(
            with_unroutable, QUERY, registry3, llm=ScriptedLLM([paths3]), state=tmp_path
        )
        assert len(result.outcomes) == 2
        assert len(result.fallbacks) == 1
        fallback = result.fallbacks[0]
        assert fallback["record"]["ticket_id"] == 3
        assert fallback["confidence"] < 0.40

    def test_execute_false_warms_cache_without_runtime(self, tmp_path):
        # No fake torch here: loading the artifact would fail, proving the
        # warm-only path never touches the runtime.
        llm = ScriptedLLM([VALID_PATHS])
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=llm, state=tmp_path, execute=False)
        assert result.artifact_path.exists()
        assert result.pipeline is None
        assert result.outcomes == []

    def test_summary_counts_actions(self, tmp_path, fake_runtime):
        result = strot.run(SAMPLE, QUERY, REGISTRY, llm=ScriptedLLM([VALID_PATHS]), state=tmp_path)
        summary = result.summary()
        assert summary["processed"] == 2
        assert summary["actions"] == {"billing": 1, "defect": 1}

    def test_unresolvable_llm_spec_raises(self, tmp_path):
        with pytest.raises(RouterConfigurationError):
            strot.run(SAMPLE, QUERY, REGISTRY, llm=12345, state=tmp_path)
