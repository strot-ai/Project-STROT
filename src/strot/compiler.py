"""Compile time: the STROT Synthesizer Engine.

:class:`StrotCompiler` implements the paper's three phases at compile time:

1. **Schema-aware context construction** — introspect the dataset sample
   into a verified structural schema (:mod:`strot.schema`).
2. **Prompt scaffolding** — build a strict, sectioned prompt for a large
   generative LLM containing the schema, the functional goal, and the
   requested ``nlu_model``.
3. **Feedback-driven synthesis** — ask the LLM for the execution-path code,
   validate its syntax and output contract, and loop error traces back for
   self-correction within a bounded retry budget.

The validated paths are then woven together with a deterministic runtime
router template (:mod:`strot.templates`) matched to the ``nlu_model``,
producing a completely standalone production asset.
"""

from __future__ import annotations

import ast
import json
import re
import types as _types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from . import templates
from .exceptions import CompilationError, RouterConfigurationError
from .router import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_NLU_MODEL,
    HOSTED_PROVIDER_PREFIXES,
)
from .schema import DatasetSchema, introspect_schema

#: A generative LLM client is any callable mapping a prompt to raw model text.
LLMClient = Callable[[str], str]

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def openai_llm_client(model: str = "gpt-4o", temperature: float = 0.0) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RouterConfigurationError(
            "openai_llm_client requires the 'hosted' extra: "
            "pip install strot-ai[hosted]"
        ) from exc
    client = OpenAI()

    def _call(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    return _call


def anthropic_llm_client(
    model: str = "claude-opus-4-8",
    max_tokens: int = 16000,
) -> LLMClient:
    """Convenience factory: a synthesizer client backed by the Anthropic API.

    Requires the ``anthropic`` package (``pip install strot-ai[anthropic]``)
    and ``ANTHROPIC_API_KEY`` in the environment.
    """
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RouterConfigurationError(
            "anthropic_llm_client requires the 'anthropic' extra: "
            "pip install strot-ai[anthropic]"
        ) from exc
    client = Anthropic()

    def _call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise CompilationError(
                "Anthropic synthesizer refused the request.", attempts=0
            )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    return _call


def claude_cli_llm_client(
    model: Optional[str] = None,
    cli: str = "claude",
    timeout: float = 600.0,
) -> LLMClient:
    """Convenience factory: use a local Claude Code CLI as an LLM proxy.

    Shells out to ``claude -p`` in headless mode, so synthesis rides the
    developer's existing Claude session — no API key, no SDK dependency.
    """
    import shutil
    import subprocess

    cli_path = shutil.which(cli)
    if cli_path is None:
        raise RouterConfigurationError(
            f"claude_cli_llm_client requires the {cli!r} CLI on PATH "
            "(install Claude Code: https://claude.com/claude-code)."
        )

    def _call(prompt: str) -> str:
        command = [cli_path, "-p", "--output-format", "text"]
        if model:
            command += ["--model", model]
        proc = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise CompilationError(
                f"claude CLI synthesizer exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}",
                attempts=0,
            )
        return proc.stdout

    return _call


def handler_name(action_key: str) -> str:
    """Map an action-registry key to its compiled function name."""
    sanitized = re.sub(r"\W+", "_", action_key).strip("_").lower()
    if not sanitized:
        raise CompilationError(f"Action key {action_key!r} cannot form a function name.", attempts=0)
    return f"execute_{sanitized}"


@dataclass(frozen=True)
class CompiledAsset:
    """A self-contained, self-routing production artifact."""

    source: str
    nlu_model: str
    runtime_kind: str  # "local" | "hosted"
    goal: str
    schema: DatasetSchema
    attempts: int

    def save(self, path: Union[str, Path] = "compiled_pipeline.py") -> Path:
        """Persist the asset as an independent Python file."""
        target = Path(path)
        target.write_text(self.source, encoding="utf-8")
        return target


class StrotCompiler:
    """Orchestrates schema introspection, LLM code generation, and assembly
    of the standalone runtime asset."""

    def __init__(
        self,
        nlu_model: str = DEFAULT_NLU_MODEL,
        llm_client: Optional[LLMClient] = None,
        max_retries: int = 3,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        self.nlu_model = nlu_model
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

    @property
    def runtime_kind(self) -> str:
        if self.nlu_model.lower().startswith(HOSTED_PROVIDER_PREFIXES):
            return "hosted"
        return "local"

    # ------------------------------------------------------------------ #
    # Phase 2: prompt scaffolding                                         #
    # ------------------------------------------------------------------ #

    def build_synthesis_prompt(
        self,
        schema: DatasetSchema,
        goal: str,
        action_registry: Mapping[str, str],
    ) -> str:
        """Construct the strict, sectioned synthesizer prompt."""
        contract_lines = [
            f"- `def {handler_name(key)}(record: dict) -> dict:` handles action "
            f"{key!r} ({desc})"
            for key, desc in action_registry.items()
        ]
        contract_lines += [
            "- `ROUTING_LABELS: Dict[str, str]` module-level literal dict mapping "
            f"EVERY action key ({', '.join(map(repr, action_registry))}) to a "
            "short routing label: 1-4 lowercase words, mutually distinct, using "
            "vocabulary that appears in the records themselves. The runtime "
            "decision model sees ONLY these labels as its options - short, "
            "discriminative labels dramatically outperform long descriptions.",
            "- `def route_context(record: dict) -> str:` projects ONLY the "
            "fields relevant to choosing an action into a compact string for "
            "the decision model; drop irrelevant or noisy fields.",
        ]
        return "\n".join(
            [
                "You are the STROT synthesizer. Generate production Python code.",
                "",
                "## VERIFIED DATASET SCHEMA",
                schema.to_json(),
                "",
                "## FUNCTIONAL GOAL",
                goal,
                "",
                "## RUNTIME NLU MODEL",
                f"The routing decision at runtime is made by {self.nlu_model!r} "
                f"({self.runtime_kind} runtime). Do NOT generate routing code; "
                "it is compiled in separately.",
                "",
                "## OUTPUT CONTRACT",
                "Emit exactly one ```python code block containing ONLY these "
                "top-level functions, fully implemented against the schema above:",
                *contract_lines,
                "",
                "Rules: pure standard library only; comprehensive exception "
                "handling inside each function; full type hints; no I/O, no "
                "network calls, no imports of torch/transformers/openai/julia; "
                "return a JSON-serializable dict.",
            ]
        )

    def _synthesize_paths(
        self,
        prompt: str,
        expected_functions: List[str],
        label_keys: Optional[List[str]] = None,
    ) -> "tuple[str, int]":
        return self._synthesize_with(
            prompt,
            lambda code: self._validate_candidate(code, expected_functions, label_keys),
        )

    def _synthesize_with(
        self, prompt: str, validate: Callable[[str], Optional[str]]
    ) -> "tuple[str, int]":
        if self.llm_client is None:
            raise CompilationError(
                "No llm_client configured. Pass a callable (e.g. "
                "strot.compiler.openai_llm_client()) to StrotCompiler.",
                attempts=0,
            )
        current_prompt = prompt
        last_error: Optional[str] = None
        candidate = ""
        for attempt in range(1, self.max_retries + 1):
            raw = self.llm_client(current_prompt)
            candidate = self._extract_code(raw)
            error = validate(candidate)
            if error is None:
                return candidate, attempt
            last_error = error
            current_prompt = self._build_repair_prompt(prompt, candidate, error)
        raise CompilationError(
            f"Synthesizer failed after {self.max_retries} attempts; "
            f"last validation error: {last_error}",
            attempts=self.max_retries,
            last_error=last_error,
        )

    @staticmethod
    def _extract_code(raw: str) -> str:
        match = _CODE_FENCE_RE.search(raw)
        return (match.group(1) if match else raw).strip()

    @staticmethod
    def _validate_candidate(
        code: str,
        expected_functions: List[str],
        label_keys: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return a human-readable validation error, or None if the code is valid."""
        if not code:
            return "The response contained no code."
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError at line {exc.lineno}: {exc.msg}"
        defined = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        missing = [name for name in expected_functions if name not in defined]
        if missing:
            return f"Missing required function definitions: {', '.join(missing)}"
        if label_keys is not None:
            error = StrotCompiler._validate_routing_labels(tree, label_keys)
            if error:
                return error
        forbidden = {"torch", "transformers", "openai", "julia", "huggingface_hub"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            bad = names & forbidden
            if bad:
                return f"Forbidden imports in execution paths: {', '.join(sorted(bad))}"
        return None

    @staticmethod
    def _validate_routing_labels(tree: ast.Module, label_keys: List[str]) -> Optional[str]:
        """Check the module-level ROUTING_LABELS literal against the registry."""
        value_node = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ROUTING_LABELS" for t in node.targets
            ):
                value_node = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "ROUTING_LABELS"
            ):
                value_node = node.value
        if value_node is None:
            return "Missing required module-level constant ROUTING_LABELS."
        try:
            labels = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return "ROUTING_LABELS must be a literal dict of str -> str."
        if not isinstance(labels, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and v.strip()
            for k, v in labels.items()
        ):
            return "ROUTING_LABELS must be a literal dict of str -> non-empty str."
        if set(labels) != set(label_keys):
            return (
                "ROUTING_LABELS keys must exactly match the action keys: "
                f"{', '.join(label_keys)}"
            )
        return None

    @staticmethod
    def _build_repair_prompt(original_prompt: str, failed_code: str, error: str) -> str:
        return "\n".join(
            [
                original_prompt,
                "",
                "## PREVIOUS ATTEMPT (REJECTED)",
                "```python",
                failed_code,
                "```",
                "",
                "## VALIDATION ERROR",
                error,
                "",
                "Fix the error and emit the corrected code block. Output only code.",
            ]
        )

    # ------------------------------------------------------------------ #
    # Assembly of the standalone asset                                    #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Golden-record promotion gate                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_candidate_module(source: str) -> Any:
        """Exec a candidate asset in-memory for the golden gate.

        A load failure here is an environment problem (missing decision
        runtime on the compile box), never the synthesizer's fault - raise
        instead of burning repair-loop retries.
        """
        import sys

        module = _types.ModuleType("strot_golden_gate")
        # Registered so stdlib machinery (e.g. dataclasses) can resolve the
        # module during exec; dropped again by the gate methods when done.
        sys.modules["strot_golden_gate"] = module
        try:
            exec(compile(source, "<strot-golden-gate>", "exec"), module.__dict__)
        except Exception as exc:
            sys.modules.pop("strot_golden_gate", None)
            raise RouterConfigurationError(
                "Golden-record gate could not load the candidate asset; the "
                f"decision runtime must be installed on the compile box: {exc}"
            ) from exc
        return module

    @staticmethod
    def validate_golden_run(golden: Any, action_registry: Mapping[str, str]) -> None:
        if not isinstance(golden, list) or not golden:
            raise RouterConfigurationError("golden must be a non-empty list.")
        for i, case in enumerate(golden):
            if (
                not isinstance(case, Mapping)
                or not isinstance(case.get("record"), Mapping)
                or case.get("expect") not in action_registry
            ):
                raise RouterConfigurationError(
                    f"golden[{i}] must be {{'record': {{...}}, 'expect': <action key>}} "
                    f"with expect one of {list(action_registry)}."
                )

    def _test_golden_pipeline(self, source: str, golden: List[Mapping[str, Any]]) -> Optional[str]:
        import sys

        module = self._load_candidate_module(source)
        try:
            pipeline = module.CompiledPipeline()
        except Exception as exc:
            raise RouterConfigurationError(
                f"Golden-record gate: decision runtime failed to initialize: {exc}"
            ) from exc
        failures = []
        for i, case in enumerate(golden):
            expect = case["expect"]
            try:
                outcome = pipeline.process(dict(case["record"]))
            except module.FallbackException as exc:
                failures.append(
                    f"golden[{i}] fell to fallback (confidence {exc.confidence:.2f}) "
                    f"but expected action {expect!r}"
                )
                continue
            if outcome["action"] != expect:
                failures.append(
                    f"golden[{i}] decided {outcome['action']!r} (confidence "
                    f"{outcome['confidence']:.2f}) but expected {expect!r}"
                )
        sys.modules.pop("strot_golden_gate", None)
        if failures:
            return (
                "Golden-record promotion gate failed - adjust ROUTING_LABELS and "
                "route_context so these known answers hold:\n" + "\n".join(failures)
            )
        return None

    def compile(
        self,
        sample: Any,
        goal: str,
        action_registry: Mapping[str, str],
        question: Optional[str] = None,
        golden: Optional[List[Mapping[str, Any]]] = None,
    ) -> CompiledAsset:
        """Run the full compile-time pipeline and return the production asset.

        Args:
            sample: A representative dataset sample (records, columns, or a
                DataFrame-like object) used for schema introspection.
            goal: The user's functional goal in natural language.
            action_registry: Ordered mapping of action key -> description;
                list position defines the routing option index at runtime.
            question: Optional override for the dynamic selection question
                baked into the asset.
            golden: Optional known-answer records
                (``[{"record": {...}, "expect": "<action key>"}, ...]``). A
                candidate asset is executed against them inside the repair
                loop and only promoted if every expectation holds. Requires
                the decision runtime on the compile box.
        """
        if not action_registry:
            raise CompilationError("action_registry must contain at least one action.", attempts=0)
        if golden is not None:
            self.validate_golden_run(golden, action_registry)
        schema = introspect_schema(sample)
        expected = [handler_name(key) for key in action_registry]
        if len(set(expected)) != len(expected):
            raise CompilationError(
                "Action keys collide after sanitization; use distinct keys.", attempts=0
            )
        expected = expected + ["route_context"]
        prompt = self.build_synthesis_prompt(schema, goal, action_registry)
        promoted: Dict[str, str] = {}

        def validate(code: str) -> Optional[str]:
            error = self._validate_candidate(code, expected, list(action_registry))
            if error:
                return error
            source = self._assemble(schema, goal, dict(action_registry), code, question)
            try:
                ast.parse(source)
            except SyntaxError as exc:  # template regression guard
                return f"Assembled asset failed syntax validation: {exc}"
            if golden:
                error = self._test_golden_pipeline(source, golden)
                if error:
                    return error
            promoted["source"] = source
            return None

        _, attempts = self._synthesize_with(prompt, validate)
        source = promoted["source"]
        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind=self.runtime_kind,
            goal=goal,
            schema=schema,
            attempts=attempts,
        )

    @staticmethod
    def validate_questions_config(questions: Mapping[str, Any]) -> None:
        """Validate a typed-question config before any synthesis."""
        if not isinstance(questions, Mapping) or not questions:
            raise RouterConfigurationError("questions must be a non-empty mapping.")
        for name, spec in questions.items():
            if not isinstance(spec, Mapping) or spec.get("type") not in (
                "noul", "score", "choice",
            ):
                raise RouterConfigurationError(
                    f"question {name!r}: type must be 'noul', 'score', or 'choice'."
                )
            if spec["type"] == "score":
                minimum, maximum = spec.get("min"), spec.get("max")
                if (
                    type(minimum) is not int
                    or type(maximum) is not int
                    or not 2 <= (maximum - minimum + 1) <= 20
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: score needs int min < max spanning "
                        "2-20 levels (the decision model's option range)."
                    )
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: score requires instructions anchoring the scale."
                    )
            elif spec["type"] == "noul":
                if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
                    raise RouterConfigurationError(
                        f"question {name!r}: noul requires instructions."
                    )
            else:  # choice
                options = spec.get("options")
                if (
                    not isinstance(options, list)
                    or not 2 <= len(options) <= 20
                    or len(set(options)) != len(options)
                    or not all(isinstance(o, str) and o for o in options)
                ):
                    raise RouterConfigurationError(
                        f"question {name!r}: choice needs 2-20 unique non-empty options."
                    )

    def build_decide_prompt(self, schema: DatasetSchema, questions: Mapping[str, Any]) -> str:
        """Construct the decide-mode prompt: expand a typed-question config for the model."""
        return "\n".join(
            [
                "You are the STROT synthesizer. Expand a typed-question config "
                "for a small local decision model - you configure, you do not "
                "write programs.",
                "",
                "## VERIFIED DATASET SCHEMA",
                schema.to_json(),
                "",
                "## DECISION CONFIG (typed questions)",
                json.dumps(questions, indent=2),
                "",
                "## OUTPUT CONTRACT",
                "Emit exactly one ```python code block containing EXACTLY two "
                "top-level definitions and nothing else:",
                "- `QUESTIONS: Dict[str, Any]` literal with the SAME question "
                "names. Per question: keep \"type\" and \"instructions\" (write "
                "crisp instructions if the config omits them). For 'choice' add "
                "\"criteria\": a dict mapping EVERY option (exact strings) to a "
                "short discriminative description using vocabulary from the "
                "records. For 'score' add \"min\" (copy the config int) and "
                "\"rubric\": a list of exactly (max - min + 1) short level "
                "labels, in order from min to max, grounded in the "
                "instructions' anchors. For 'noul' add \"criteria\": "
                "{\"false\": <short negative-case description>, \"true\": "
                "<short positive-case description>}.",
                "- `def route_context(record: dict) -> str:` projecting ONLY "
                "the fields relevant to answering the questions into a compact "
                "string.",
                "",
                "Rules: short, mutually distinct labels dramatically outperform "
                "long ones; no imports, no other statements.",
            ]
        )

    def _validate_decide_candidate(
        self, code: str, questions: Mapping[str, Any]
    ) -> Optional[str]:
        error = self._validate_candidate(code, ["route_context"])
        if error:
            return error
        tree = ast.parse(code)
        value_node = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "QUESTIONS" for t in targets):
                    value_node = node.value
        if value_node is None:
            return "Missing required module-level literal QUESTIONS."
        try:
            expanded = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return "QUESTIONS must be a literal dict."
        if set(expanded) != set(questions):
            return f"QUESTIONS keys must exactly match the config: {sorted(questions)}"
        for name, spec in questions.items():
            item = expanded[name]
            if not isinstance(item, dict) or item.get("type") != spec["type"]:
                return f"question {name!r}: type must stay {spec['type']!r}"
            if not isinstance(item.get("instructions"), str) or not item["instructions"].strip():
                return f"question {name!r}: instructions must be a non-empty string"
            if spec["type"] == "choice":
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or set(criteria) != set(spec["options"]):
                    return (
                        f"question {name!r}: criteria keys must exactly match "
                        f"the options {spec['options']}"
                    )
                if not all(isinstance(v, str) and v.strip() for v in criteria.values()):
                    return f"question {name!r}: criteria descriptions must be non-empty strings"
            elif spec["type"] == "score":
                expected_len = spec["max"] - spec["min"] + 1
                rubric = item.get("rubric")
                if item.get("min") != spec["min"]:
                    return f"question {name!r}: min must equal {spec['min']}"
                if (
                    not isinstance(rubric, list)
                    or len(rubric) != expected_len
                    or not all(isinstance(v, str) and v.strip() for v in rubric)
                ):
                    return (
                        f"question {name!r}: rubric must be a list of exactly "
                        f"{expected_len} non-empty level labels"
                    )
            else:  # noul
                criteria = item.get("criteria")
                if not isinstance(criteria, dict) or set(criteria) != {"false", "true"}:
                    return (
                        f"question {name!r}: criteria must map exactly "
                        "'false' and 'true' to descriptions"
                    )
        return None

    @staticmethod
    def validate_golden_decide(golden: Any, questions: Mapping[str, Any]) -> None:
        if not isinstance(golden, list) or not golden:
            raise RouterConfigurationError("golden must be a non-empty list.")
        for i, case in enumerate(golden):
            if (
                not isinstance(case, Mapping)
                or not isinstance(case.get("record"), Mapping)
                or not isinstance(case.get("expect"), Mapping)
                or not case["expect"]
            ):
                raise RouterConfigurationError(
                    f"golden[{i}] must be {{'record': {{...}}, 'expect': "
                    "{<question>: <expected>}}."
                )
            for qname, expected in case["expect"].items():
                spec = questions.get(qname)
                if spec is None:
                    raise RouterConfigurationError(
                        f"golden[{i}]: unknown question {qname!r}."
                    )
                if spec["type"] == "choice" and expected not in spec["options"]:
                    raise RouterConfigurationError(
                        f"golden[{i}].{qname}: expected must be one of {spec['options']}."
                    )
                if spec["type"] == "noul" and not isinstance(expected, bool):
                    raise RouterConfigurationError(
                        f"golden[{i}].{qname}: noul expectation must be true/false."
                    )
                if spec["type"] == "score":
                    is_range = (
                        isinstance(expected, (list, tuple))
                        and len(expected) == 2
                        and all(isinstance(v, (int, float)) for v in expected)
                    )
                    if not is_range and not isinstance(expected, (int, float)):
                        raise RouterConfigurationError(
                            f"golden[{i}].{qname}: score expectation must be a "
                            "number or [low, high] range."
                        )

    def _test_golden_decider(
        self,
        source: str,
        golden: List[Mapping[str, Any]],
        questions: Mapping[str, Any],
    ) -> Optional[str]:
        import sys

        module = self._load_candidate_module(source)
        try:
            decider = module.CompiledDecider()
        except Exception as exc:
            raise RouterConfigurationError(
                f"Golden-record gate: decision runtime failed to initialize: {exc}"
            ) from exc
        failures = []
        for i, case in enumerate(golden):
            decision = decider.process(dict(case["record"]))
            for qname, expected in case["expect"].items():
                qtype = questions[qname]["type"]
                if qtype == "choice":
                    got = decision["choices"][qname]["selected"]
                    if got != expected:
                        failures.append(
                            f"golden[{i}].{qname}: selected {got!r}, expected {expected!r}"
                        )
                elif qtype == "noul":
                    p = decision["nouls"][qname]["noul"]
                    if (p >= 0.5) != expected:
                        failures.append(
                            f"golden[{i}].{qname}: noul={p:.2f}, expected "
                            f"{'true' if expected else 'false'}"
                        )
                else:  # score
                    value = decision["scores"][qname]["value"]
                    low, high = (
                        expected
                        if isinstance(expected, (list, tuple))
                        else (expected - 0.5, expected + 0.5)
                    )
                    if not low <= value <= high:
                        failures.append(
                            f"golden[{i}].{qname}: value={value}, expected "
                            f"within [{low}, {high}]"
                        )
        sys.modules.pop("strot_golden_gate", None)
        if failures:
            return (
                "Golden-record promotion gate failed - adjust the criteria, "
                "rubrics, and route_context so these known answers hold:\n"
                + "\n".join(failures)
            )
        return None

    def compile_decide(
        self,
        sample: Any,
        questions: Mapping[str, Any],
        golden: Optional[List[Mapping[str, Any]]] = None,
    ) -> CompiledAsset:
        """Decide mode: compile a typed-question decision config.

        The synthesis agent expands the config (criteria descriptions, score
        rubrics, noul case descriptions) against the verified schema and
        writes the state projection; the local decision model answers all
        questions per record at runtime in one batched pass. With ``golden``,
        candidates are executed against known-answer records inside the
        repair loop and only promoted when every expectation holds.
        """
        if self.runtime_kind != "local":
            raise RouterConfigurationError(
                "decide mode currently requires a local decision model "
                f"(got {self.nlu_model!r})."
            )
        self.validate_questions_config(questions)
        if golden is not None:
            self.validate_golden_decide(golden, questions)
        schema = introspect_schema(sample)
        prompt = self.build_decide_prompt(schema, questions)
        header = templates.DECIDE_HEADER_TEMPLATE.format(
            nlu_model=self.nlu_model,
            schema_literal=repr(schema.to_dict()),
            config_literal=repr(dict(questions)),
        )
        promoted: Dict[str, str] = {}

        def validate(code: str) -> Optional[str]:
            error = self._validate_decide_candidate(code, questions)
            if error:
                return error
            source = (
                "\n\n".join(
                    [header, templates.DECIDE_FOOTER_TEMPLATE.format(generated_paths=code)]
                )
                + "\n"
            )
            try:
                ast.parse(source)
            except SyntaxError as exc:  # template regression guard
                return f"Assembled decision asset failed syntax validation: {exc}"
            if golden:
                error = self._test_golden_decider(source, golden, questions)
                if error:
                    return error
            promoted["source"] = source
            return None

        _, attempts = self._synthesize_with(prompt, validate)
        source = promoted["source"]
        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind="decide",
            goal=json.dumps(questions, sort_keys=True),
            schema=schema,
            attempts=attempts,
        )

    def _assemble(
        self,
        schema: DatasetSchema,
        goal: str,
        action_registry: Dict[str, str],
        generated_paths: str,
        question: Optional[str],
    ) -> str:
        header = templates.ASSET_HEADER_TEMPLATE.format(
            nlu_model=self.nlu_model,
            runtime_kind=self.runtime_kind,
            goal=goal,
            threshold=self.confidence_threshold,
            schema_literal=repr(schema.to_dict()),
        )
        runtime_template = (
            templates.HOSTED_RUNTIME_TEMPLATE
            if self.runtime_kind == "hosted"
            else templates.LOCAL_RUNTIME_TEMPLATE
        )
        runtime = runtime_template.format(nlu_model=self.nlu_model)
        handlers_literal = (
            "{"
            + ", ".join(f"{key!r}: {handler_name(key)}" for key in action_registry)
            + "}"
        )
        footer = templates.ASSET_FOOTER_TEMPLATE.format(
            generated_paths=generated_paths,
            registry_literal=repr(action_registry),
            handlers_literal=handlers_literal,
            default_question=question
            or f"Which execution path best achieves the goal: {goal}?",
        )
        return "\n\n".join([header, runtime, footer]) + "\n"
