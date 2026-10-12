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
import re
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
                "network calls, no imports of torch/transformers/openai; "
                "return a JSON-serializable dict.",
            ]
        )

    # ------------------------------------------------------------------ #
    # Phase 3: feedback-driven synthesis with self-correction             #
    # ------------------------------------------------------------------ #

    def _synthesize_paths(self, prompt: str, expected_functions: List[str]) -> "tuple[str, int]":
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
            error = self._validate_candidate(candidate, expected_functions)
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
    def _validate_candidate(code: str, expected_functions: List[str]) -> Optional[str]:
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
        forbidden = {"torch", "transformers", "openai"}
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

    def compile(
        self,
        sample: Any,
        goal: str,
        action_registry: Mapping[str, str],
        question: Optional[str] = None,
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
        """
        if not action_registry:
            raise CompilationError("action_registry must contain at least one action.", attempts=0)
        schema = introspect_schema(sample)
        expected = [handler_name(key) for key in action_registry]
        if len(set(expected)) != len(expected):
            raise CompilationError(
                "Action keys collide after sanitization; use distinct keys.", attempts=0
            )
        prompt = self.build_synthesis_prompt(schema, goal, action_registry)
        generated_paths, attempts = self._synthesize_paths(prompt, expected)
        source = self._assemble(schema, goal, dict(action_registry), generated_paths, question)
        try:
            ast.parse(source)
        except SyntaxError as exc:  # template regression guard
            raise CompilationError(
                f"Assembled asset failed final syntax validation: {exc}",
                attempts=attempts,
                last_error=str(exc),
            ) from exc
        return CompiledAsset(
            source=source,
            nlu_model=self.nlu_model,
            runtime_kind=self.runtime_kind,
            goal=goal,
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
