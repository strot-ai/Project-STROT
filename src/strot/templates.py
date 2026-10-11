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
# --- Runtime architecture: Implementation Alpha (local CPU transformer) ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class _RuntimeRouter:
    """Single-forward-pass logit router over {nlu_model} on CPU."""

    def __init__(self) -> None:
        device = torch.device("cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(NLU_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(NLU_MODEL).to(device).eval()

    @staticmethod
    def _format_prompt(context: str, question: str, options: Sequence[str]) -> str:
        numbered = " ".join("({{0}}) {{1}}".format(i + 1, opt) for i, opt in enumerate(options))
        return "Context: {{0}}\\nQuestion: {{1}}\\nOptions: {{2}}\\nAnswer:".format(
            context, question, numbered
        )

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
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
        prompt = self._format_prompt(context, question, options)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]  # single forward pass
        option_token_ids = [
            self._tokenizer.encode(str(i + 1), add_special_tokens=False)[0]
            for i in range(len(options))
        ]
        probs = torch.softmax(logits[option_token_ids], dim=-1)
        winner = int(torch.argmax(probs).item())
        return RouteDecision(
            index=winner,
            option=options[winner],
            confidence=float(probs[winner]),
            distribution={{opt: float(probs[i]) for i, opt in enumerate(options)}},
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

ASSET_FOOTER_TEMPLATE = '''\
# --- Statically generated execution paths (STROT synthesizer output) ---
{generated_paths}

#: Ordered action registry: list position == routing option index.
ACTION_REGISTRY: Dict[str, str] = {registry_literal}

_HANDLERS = {handlers_literal}

_ROUTING_OPTIONS: List[str] = [
    "{{0}} -- {{1}}".format(key, desc) for key, desc in ACTION_REGISTRY.items()
]
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
        context = json.dumps(record, default=str, sort_keys=True)
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
