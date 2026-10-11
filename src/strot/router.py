"""Runtime routing: the Strategy Pattern over NLU backends.

Two completely distinct runtime implementations are supported, selected by
the ``nlu_model`` string:

* **Implementation Alpha** — :class:`LocalEncoderStrategy`: a local
  transformer (default ``SupersonicLabs/Julia-1``) executed on CPU, scoring
  all options in a single forward pass over the logits. When the action
  registry exceeds :data:`MAX_FLAT_OPTIONS` choices, an Option Tree Splitter
  decomposes the decision into a multi-tier hierarchy (Julia-1 accuracy
  degrades beyond 20 options).

* **Implementation Beta** — :class:`HostedApiStrategy`: an ultra-lightweight
  cloud call that never imports ``torch``/``transformers``, authenticates via
  ``os.environ``, and uses Structured Outputs (strict JSON Schema) to force
  the API to return only the winning execution path index.

Both are wrapped by :class:`Router`, which enforces the calibrated-confidence
threshold and raises :class:`~strot.exceptions.FallbackException` instead of
blindly executing a low-confidence path.
"""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .exceptions import FallbackException, RouterConfigurationError

DEFAULT_NLU_MODEL = "SupersonicLabs/Julia-1"
CONFIDENCE_THRESHOLD = 0.40
MAX_FLAT_OPTIONS = 20

#: Model-name prefixes that resolve to the hosted (Implementation Beta) runtime.
HOSTED_PROVIDER_PREFIXES: Tuple[str, ...] = ("openai/", "anthropic/", "azure/", "gemini/")

#: Environment variable holding the API key for each hosted provider.
PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of a single routing pass."""

    index: int
    option: str
    confidence: float
    distribution: Dict[str, float] = field(default_factory=dict)


class RuntimeStrategy(ABC):
    """Abstract NLU backend: pick one option given context and a question."""

    nlu_model: str

    @abstractmethod
    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        """Return the winning option with a calibrated confidence score."""


def chunk_options(options: Sequence[str], max_flat: int = MAX_FLAT_OPTIONS) -> List[List[str]]:
    """Split an oversized option list into balanced groups of <= ``max_flat``.

    This is the Option Tree Splitter primitive: each group becomes one branch
    of a hierarchical decision tree.
    """
    if max_flat < 2:
        raise ValueError("max_flat must be >= 2 to form a decision tree.")
    n_groups = math.ceil(len(options) / max_flat)
    size = math.ceil(len(options) / n_groups)
    return [list(options[i : i + size]) for i in range(0, len(options), size)]


class LocalEncoderStrategy(RuntimeStrategy):
    """Implementation Alpha: local transformer on CPU, single forward pass.

    The prompt follows the Julia-1 token format::

        Context: <serialized record>
        Question: <dynamic selection question>
        Options: <numbered choices>
        Answer:

    The model is run once; the logits of the final position are gathered at
    the token ids of each option number and softmaxed, yielding a calibrated
    distribution over options in <40 ms on commodity CPUs.
    """

    def __init__(self, nlu_model: str = DEFAULT_NLU_MODEL, max_flat_options: int = MAX_FLAT_OPTIONS) -> None:
        self.nlu_model = nlu_model
        self.max_flat_options = max_flat_options
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RouterConfigurationError(
                f"Local runtime for {self.nlu_model!r} requires the 'local' extra: "
                "pip install strot-ai[local]"
            ) from exc
        self._torch = torch
        device = torch.device("cpu")  # Julia-1 is sized for standard CPUs.
        self._tokenizer = AutoTokenizer.from_pretrained(self.nlu_model)
        self._model = AutoModelForCausalLM.from_pretrained(self.nlu_model).to(device).eval()

    @staticmethod
    def format_prompt(context: str, question: str, options: Sequence[str]) -> str:
        """Render the Julia-1 specific token format."""
        numbered = " ".join(f"({i + 1}) {opt}" for i, opt in enumerate(options))
        return f"Context: {context}\nQuestion: {question}\nOptions: {numbered}\nAnswer:"

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        if not options:
            raise RouterConfigurationError("Cannot route with an empty option list.")
        if len(options) <= self.max_flat_options:
            return self._select_flat(context, question, options)
        return self._select_hierarchical(context, question, options)

    def _select_hierarchical(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        """Option Tree Splitter: multi-tier decision when options exceed the cap."""
        groups = chunk_options(options, self.max_flat_options)
        labels = [
            f"Options {options.index(g[0]) + 1}-{options.index(g[-1]) + 1}: {g[0]} ... {g[-1]}"
            for g in groups
        ]
        branch_question = f"{question} (choose the group containing the best option)"
        branch = self.select(context, branch_question, labels)
        leaf = self.select(context, question, groups[branch.index])
        winner = groups[branch.index][leaf.index]
        return RouteDecision(
            index=list(options).index(winner),
            option=winner,
            confidence=branch.confidence * leaf.confidence,
            distribution={**branch.distribution, **leaf.distribution},
        )

    def _select_flat(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        self._ensure_loaded()
        torch = self._torch
        prompt = self.format_prompt(context, question, options)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]  # single forward pass
        option_token_ids = [
            self._tokenizer.encode(str(i + 1), add_special_tokens=False)[0]
            for i in range(len(options))
        ]
        probs = torch.softmax(logits[option_token_ids], dim=-1)
        winner = int(torch.argmax(probs).item())
        distribution = {opt: float(probs[i]) for i, opt in enumerate(options)}
        return RouteDecision(
            index=winner,
            option=options[winner],
            confidence=float(probs[winner]),
            distribution=distribution,
        )


class HostedApiStrategy(RuntimeStrategy):
    """Implementation Beta: hosted cloud API with Structured Outputs.

    Keeps the production footprint tiny: no ``torch``/``transformers`` import
    anywhere on this code path. The JSON Schema forces the provider to return
    a strict integer index plus a self-reported confidence.
    """

    RESPONSE_SCHEMA: Dict[str, Any] = {
        "name": "route_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "selected_index": {"type": "integer"},
                "confidence": {"type": "number"},
            },
            "required": ["selected_index", "confidence"],
            "additionalProperties": False,
        },
    }

    def __init__(self, nlu_model: str, client_factory: Optional[Callable[[], Any]] = None) -> None:
        self.nlu_model = nlu_model
        self.provider, _, self.model_id = nlu_model.partition("/")
        if not self.model_id:
            raise RouterConfigurationError(
                f"Hosted model names must look like 'provider/model', got {nlu_model!r}."
            )
        self._client_factory = client_factory
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        env_key = PROVIDER_ENV_KEYS.get(self.provider, "OPENAI_API_KEY")
        api_key = os.environ.get(env_key)
        if not api_key:
            raise RouterConfigurationError(
                f"Hosted runtime for {self.nlu_model!r} needs {env_key} in the environment."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RouterConfigurationError(
                "Hosted runtime requires the 'hosted' extra: pip install strot-ai[hosted]"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        return self._client

    @classmethod
    def build_messages(
        cls, context: str, question: str, options: Sequence[str]
    ) -> List[Dict[str, str]]:
        numbered = "\n".join(f"{i}: {opt}" for i, opt in enumerate(options))
        return [
            {
                "role": "system",
                "content": (
                    "You are a deterministic router. Given a context, a question, and a "
                    "zero-indexed option list, respond with the single best option index "
                    "and your confidence in [0, 1]."
                ),
            },
            {
                "role": "user",
                "content": f"Context: {context}\nQuestion: {question}\nOptions:\n{numbered}",
            },
        ]

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        if not options:
            raise RouterConfigurationError("Cannot route with an empty option list.")
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model_id,
            temperature=0.0,
            messages=self.build_messages(context, question, options),
            response_format={"type": "json_schema", "json_schema": self.RESPONSE_SCHEMA},
        )
        payload = json.loads(response.choices[0].message.content)
        index = int(payload["selected_index"])
        if not 0 <= index < len(options):
            raise FallbackException(
                f"Hosted router returned out-of-range index {index}.",
                confidence=0.0,
                threshold=CONFIDENCE_THRESHOLD,
                options=options,
            )
        confidence = max(0.0, min(1.0, float(payload["confidence"])))
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=confidence,
            distribution={options[index]: confidence},
        )


def create_strategy(nlu_model: str = DEFAULT_NLU_MODEL, **kwargs: Any) -> RuntimeStrategy:
    """Factory: resolve an ``nlu_model`` string to its runtime strategy.

    Names with a known hosted provider prefix (``openai/...``) resolve to
    :class:`HostedApiStrategy`; everything else is treated as a Hugging Face
    style local checkpoint and resolves to :class:`LocalEncoderStrategy`.
    """
    if nlu_model.lower().startswith(HOSTED_PROVIDER_PREFIXES):
        return HostedApiStrategy(nlu_model, **kwargs)
    return LocalEncoderStrategy(nlu_model, **kwargs)


class Router:
    """Confidence-enforcing wrapper around a :class:`RuntimeStrategy`.

    If the winning option's calibrated probability falls below ``threshold``
    (default 0.40), :class:`~strot.exceptions.FallbackException` is raised so
    corrupt or out-of-distribution inputs never execute a random path.
    """

    def __init__(
        self,
        strategy: RuntimeStrategy,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise RouterConfigurationError("threshold must be within [0, 1].")
        self.strategy = strategy
        self.threshold = threshold

    def route(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        decision = self.strategy.select(context, question, options)
        if decision.confidence < self.threshold:
            raise FallbackException(
                f"Routing confidence {decision.confidence:.3f} is below the "
                f"{self.threshold:.2f} enforcement threshold for option {decision.option!r}.",
                confidence=decision.confidence,
                threshold=self.threshold,
                options=options,
            )
        return decision


def create_router(
    nlu_model: str = DEFAULT_NLU_MODEL,
    threshold: float = CONFIDENCE_THRESHOLD,
    **kwargs: Any,
) -> Router:
    """Build a confidence-enforcing :class:`Router` for ``nlu_model``."""
    return Router(create_strategy(nlu_model, **kwargs), threshold=threshold)
