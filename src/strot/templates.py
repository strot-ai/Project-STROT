"""Code templates baked into the compiled production asset.

The synthesizer LLM generates only the task-specific *execution paths*;
the runtime router architecture is rendered deterministically from these
templates so the compiled artifact's routing behavior is guaranteed, fully
standalone (zero imports from ``strot``), and matched exactly to the
``nlu_model`` chosen at compile time.
"""

from __future__ import annotations

ASSET_HEADER_TEMPLATE = '''\
"""Compiled STROT execution asset. DO NOT EDIT BY HAND.

Synthesized by strot-ai (STROT: Structured Task Reasoning and
Output Transformation, arXiv:2505.01636).

    nlu_model : {nlu_model}
    runtime   : {runtime_kind}
    goal      : {goal}

This file is completely self-contained (zero strot-package imports) and
can be dropped into any production pipeline as-is.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

NLU_MODEL = {nlu_model!r}
CONFIDENCE_THRESHOLD = {threshold}
MAX_FLAT_OPTIONS = 20

DATASET_SCHEMA: Dict[str, Any] = {schema_literal}


class FallbackException(Exception):
    """Low-confidence routing refusal: do not execute any path blindly."""

    def __init__(self, message: str, confidence: float, threshold: float) -> None:
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold


@dataclass(frozen=True)
class RouteDecision:
    index: int
    option: str
    confidence: float
    distribution: Dict[str, float] = field(default_factory=dict)
'''

LOCAL_RUNTIME_TEMPLATE = '''\
# --- Runtime architecture: Implementation Alpha (local Julia decision model) ---
# Julia is NOT a Transformers AutoModel; it ships its own inference package
# ("supersonic-julia") inside the checkpoint repository.
try:
    from julia import load_model
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This compiled asset routes locally with {nlu_model} and needs the "
        "'supersonic-julia' package (do NOT `pip install julia` - that is an "
        "unrelated package). Install:\\n"
        "  python -c \\"from huggingface_hub import snapshot_download; "
        "snapshot_download('{nlu_model}', local_dir='Julia-1')\\"\\n"
        "  pip install ./Julia-1"
    ) from _exc


class _RuntimeRouter:
    """Resident Julia decision engine over {nlu_model} on CPU."""

    def __init__(self) -> None:
        checkpoint = os.environ.get("STROT_NLU_CHECKPOINT")
        if checkpoint is None:
            from pathlib import Path

            if Path(NLU_MODEL).is_dir():
                checkpoint = NLU_MODEL
            else:
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(NLU_MODEL)
        self._engine = load_model(checkpoint, device="cpu")

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        if len(options) == 1:  # Julia accepts 2-20 options; one action is trivial
            return RouteDecision(0, options[0], 1.0, {{options[0]: 1.0}})
        if len(options) <= MAX_FLAT_OPTIONS:
            return self._select_flat(context, question, options)
        return self._select_hierarchical(context, question, options)

    def _select_hierarchical(
        self, context: str, question: str, options: Sequence[str]
    ) -> RouteDecision:
        # Option Tree Splitter: Julia-1 accuracy degrades beyond 20 choices,
        # so oversized registries route through a multi-tier decision tree.
        n_groups = math.ceil(len(options) / MAX_FLAT_OPTIONS)
        size = math.ceil(len(options) / n_groups)
        groups = [list(options[i : i + size]) for i in range(0, len(options), size)]
        labels = [
            "Options {{0}}-{{1}}: {{2}} ... {{3}}".format(
                list(options).index(g[0]) + 1, list(options).index(g[-1]) + 1, g[0], g[-1]
            )
            for g in groups
        ]
        branch = self.select(
            context, question + " (choose the group containing the best option)", labels
        )
        leaf = self.select(context, question, groups[branch.index])
        winner = groups[branch.index][leaf.index]
        return RouteDecision(
            index=list(options).index(winner),
            option=winner,
            confidence=branch.confidence * leaf.confidence,
            distribution=dict(leaf.distribution),
        )

    def _select_flat(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        result = self._engine.predict(
            rows=[{{"state": context, "question": question, "options": list(options)}}]
        )[0]
        index = int(result["index"])
        probs = [float(p) for p in result["probabilities"]]
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=probs[index],
            distribution=dict(zip(options, probs)),
        )
'''

HOSTED_RUNTIME_TEMPLATE = '''\
# --- Runtime architecture: Implementation Beta (hosted cloud API) ---
# Deliberately torch/transformers-free to keep the container footprint tiny.
from openai import OpenAI

_PROVIDER, _, _MODEL_ID = NLU_MODEL.partition("/")
_ENV_KEYS = {{
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}}

_RESPONSE_SCHEMA: Dict[str, Any] = {{
    "name": "route_decision",
    "strict": True,
    "schema": {{
        "type": "object",
        "properties": {{
            "selected_index": {{"type": "integer"}},
            "confidence": {{"type": "number"}},
        }},
        "required": ["selected_index", "confidence"],
        "additionalProperties": False,
    }},
}}


class _RuntimeRouter:
    """Structured-Outputs router over {nlu_model}: strict integer index only."""

    def __init__(self) -> None:
        env_key = _ENV_KEYS.get(_PROVIDER, "OPENAI_API_KEY")
        api_key = os.environ.get(env_key)
        if not api_key:
            raise RuntimeError("Missing {{0}} in environment.".format(env_key))
        self._client = OpenAI(api_key=api_key)

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        numbered = "\\n".join("{{0}}: {{1}}".format(i, opt) for i, opt in enumerate(options))
        response = self._client.chat.completions.create(
            model=_MODEL_ID,
            temperature=0.0,
            messages=[
                {{
                    "role": "system",
                    "content": (
                        "You are a deterministic router. Given a context, a question, and a "
                        "zero-indexed option list, respond with the single best option index "
                        "and your confidence in [0, 1]."
                    ),
                }},
                {{
                    "role": "user",
                    "content": "Context: {{0}}\\nQuestion: {{1}}\\nOptions:\\n{{2}}".format(
                        context, question, numbered
                    ),
                }},
            ],
            response_format={{"type": "json_schema", "json_schema": _RESPONSE_SCHEMA}},
        )
        payload = json.loads(response.choices[0].message.content)
        index = int(payload["selected_index"])
        if not 0 <= index < len(options):
            raise FallbackException(
                "Hosted router returned out-of-range index {{0}}.".format(index),
                confidence=0.0,
                threshold=CONFIDENCE_THRESHOLD,
            )
        confidence = max(0.0, min(1.0, float(payload["confidence"])))
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=confidence,
            distribution={{options[index]: confidence}},
        )
'''

RESEARCH_HEADER_TEMPLATE = '''\
"""Compiled STROT research asset. DO NOT EDIT BY HAND.

Synthesized by strot-ai (STROT: Structured Task Reasoning and
Output Transformation, arXiv:2505.01636) in research mode: the SQL plan
below was synthesized once at compile time and validated by executing every
query against the dataset sample. Every number in the output is computed by
DuckDB over the records - never generated by a model, and SELECT-only by
contract: no side effects are possible.

    goal : {goal}

Runtime dependency: duckdb (pip install strot-ai[research]).
"""

import json
from typing import Any, Dict, List, Mapping

try:
    import duckdb
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This compiled research asset executes its SQL plan with DuckDB: "
        "pip install strot-ai[research] (or: pip install duckdb)"
    ) from _exc

DATASET_SCHEMA: Dict[str, Any] = {schema_literal}

#: Profiler semantic type -> DuckDB column type (temporal stays VARCHAR; the
#: plan CASTs where needed, so ingestion never fails on odd date formats).
DUCK_TYPES: Dict[str, str] = {{
    "numerical": "DOUBLE",
    "boolean": "BOOLEAN",
    "temporal": "VARCHAR",
    "categorical": "VARCHAR",
    "text": "VARCHAR",
    "unknown": "VARCHAR",
}}
'''

RESEARCH_FOOTER_TEMPLATE = '''\
# --- Statically synthesized SQL plan (STROT synthesizer output) ---
{generated_paths}


class CompiledAnalysis:
    """Self-contained research asset: deterministic DuckDB execution."""

    def __init__(self) -> None:
        self._fields = [
            (field["name"], DUCK_TYPES.get(field["semantic_type"], "VARCHAR"))
            for field in DATASET_SCHEMA["fields"]
        ]

    def process(self, records: List[Mapping[str, Any]]) -> Dict[str, Any]:
        con = duckdb.connect(database=":memory:")
        try:
            columns = ", ".join(
                '"{{0}}" {{1}}'.format(name, duck_type) for name, duck_type in self._fields
            )
            con.execute("CREATE TABLE records ({{0}})".format(columns))
            rows = [
                tuple(record.get(name) for name, _ in self._fields)
                for record in records
            ]
            if rows:
                placeholders = ", ".join("?" for _ in self._fields)
                con.executemany(
                    "INSERT INTO records VALUES ({{0}})".format(placeholders), rows
                )
            results: Dict[str, Any] = {{}}
            for name, sql in QUERIES.items():
                cursor = con.execute(sql)
                out_columns = [description[0] for description in cursor.description]
                fetched = cursor.fetchall()
                if len(fetched) == 1 and len(out_columns) == 1:
                    results[name] = fetched[0][0]
                else:
                    results[name] = [dict(zip(out_columns, row)) for row in fetched]
            return results
        finally:
            con.close()


if __name__ == "__main__":
    import sys

    print(json.dumps(CompiledAnalysis().process(json.load(sys.stdin)), default=str))
'''

ASSET_FOOTER_TEMPLATE = '''\
# --- Statically generated execution paths (STROT synthesizer output) ---
{generated_paths}

#: Ordered action registry: list position == routing option index.
ACTION_REGISTRY: Dict[str, str] = {registry_literal}

_HANDLERS = {handlers_literal}

#: The decision model sees ONLY these compile-time-synthesized labels;
#: short discriminative labels measurably outperform verbose descriptions.
_ROUTING_OPTIONS: List[str] = [ROUTING_LABELS[key] for key in ACTION_REGISTRY]
_ACTION_KEYS: List[str] = list(ACTION_REGISTRY)


class CompiledPipeline:
    """Self-routing execution asset: route each record, enforce confidence,
    dispatch to the statically compiled path."""

    def __init__(self) -> None:
        self._router = _RuntimeRouter()

    def process(
        self,
        record: Mapping[str, Any],
        question: str = {default_question!r},
    ) -> Dict[str, Any]:
        context = route_context(dict(record))  # compile-time-synthesized projection
        decision = self._router.select(context, question, _ROUTING_OPTIONS)
        if decision.confidence < CONFIDENCE_THRESHOLD:
            raise FallbackException(
                "Routing confidence {{0:.3f}} below threshold {{1:.2f}}; refusing to "
                "execute a path for this record.".format(
                    decision.confidence, CONFIDENCE_THRESHOLD
                ),
                confidence=decision.confidence,
                threshold=CONFIDENCE_THRESHOLD,
            )
        action_key = _ACTION_KEYS[decision.index]
        result = _HANDLERS[action_key](dict(record))
        return {{
            "action": action_key,
            "confidence": decision.confidence,
            "result": result,
        }}


if __name__ == "__main__":
    import sys

    pipeline = CompiledPipeline()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            print(json.dumps(pipeline.process(json.loads(line)), default=str))
        except FallbackException as exc:
            print(json.dumps({{"action": None, "error": str(exc)}}))
'''
