# Project STROT (`strot-ai`)

[![PyPI](https://img.shields.io/pypi/v/strot-ai)](https://pypi.org/project/strot-ai/)
[![Python](https://img.shields.io/pypi/pyversions/strot-ai)](https://pypi.org/project/strot-ai/)
[![License](https://img.shields.io/pypi/l/strot-ai)](LICENSE)

**Compile-time LLM code synthesis, deterministic runtime routing.**

An open-source implementation of the [STROT framework](https://arxiv.org/abs/2505.01636)
(*Structured Task Reasoning and Output Transformation*) extended with an NLU
strategy-pattern runtime: a large generative LLM scaffolds your execution
paths **once at compile time**, and a small, fast NLU model routes every
record **deterministically at runtime** — baked into a single standalone
Python file.

```
┌──────────────────────── COMPILE TIME ────────────────────────┐
│ dataset sample ─▶ schema introspection (Phase 1)             │
│                ─▶ prompt scaffolding   (Phase 2)             │
│                ─▶ LLM code synthesis + self-correction loop  │
│                   (Phase 3, bounded retries on error traces) │
│                ─▶ compiled_pipeline.py  (standalone asset)   │
└──────────────────────────────────────────────────────────────┘
┌───────────────────────── RUNTIME ────────────────────────────┐
│ record ─▶ NLU router (local CPU transformer OR hosted API)   │
│        ─▶ confidence >= 0.40 ? dispatch : FallbackException  │
│        ─▶ statically compiled execution path                 │
└──────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install strot-ai            # core (pure stdlib)
pip install strot-ai[local]     # + torch/transformers for local NLU
pip install strot-ai[hosted]    # + openai for hosted NLU / synthesis
```

## Quick start

```python
from strot import StrotCompiler
from strot.compiler import openai_llm_client

tickets = [
    {"ticket_id": 101, "channel": "email", "body": "I was double charged"},
    {"ticket_id": 102, "channel": "chat",  "body": "App crashes on export"},
]

compiler = StrotCompiler(
    nlu_model="SupersonicLabs/Julia-1",     # default: local, CPU, Apache-2.0
    llm_client=openai_llm_client("gpt-4o"), # the compile-time synthesizer
)

asset = compiler.compile(
    sample=tickets,
    goal="Triage incoming support tickets into the correct workflow.",
    action_registry={
        "billing_dispute":  "Refunds, invoice corrections, double charges",
        "technical_defect": "Crashes, bugs, broken behavior",
    },
)
asset.save("compiled_pipeline.py")          # completely self-contained
```

Then, anywhere in production (no `strot` dependency required):

```python
from compiled_pipeline import CompiledPipeline, FallbackException

pipeline = CompiledPipeline()
try:
    outcome = pipeline.process({"ticket_id": 7, "body": "charged twice"})
    print(outcome)  # {"action": "billing_dispute", "confidence": 0.93, "result": {...}}
except FallbackException as exc:
    quarantine(exc)  # confidence < 0.40 — never execute a random path
```

See [`examples/end_to_end.py`](examples/end_to_end.py) for the full
walkthrough (it also runs offline with a stub synthesizer).

## The two runtime architectures

The `nlu_model` string selects which runtime gets compiled into the asset:

**Implementation Alpha — local encoder** (default `SupersonicLabs/Julia-1`).
Loads via `transformers`, forced onto `torch.device("cpu")`. Options are
rendered in the Julia-1 token format (`Context: ...\nQuestion: ...\nOptions:
...\nAnswer:`) and scored from the logits of a **single forward pass**
(<40 ms on CPU). If the action registry exceeds 20 keys, an **Option Tree
Splitter** decomposes the decision into a multi-tier hierarchy, since Julia-1
accuracy degrades beyond 20 choices.

**Implementation Beta — hosted API** (e.g. `openai/gpt-4o-mini`).
An ultra-lightweight block: API keys from `os.environ`, **zero
torch/transformers imports** (tiny container footprint), and Structured
Outputs (strict JSON Schema) forcing the provider to return only the integer
index of the winning execution path.

Both runtimes enforce the same guard: if the calibrated confidence of the
winning path is below **0.40**, the asset raises `FallbackException` instead
of executing blindly.

## API surface

| Symbol | Role |
| --- | --- |
| `StrotCompiler` | Phase 1–3 orchestration: introspect → scaffold → synthesize → assemble |
| `CompiledAsset` | The standalone artifact; `.save("compiled_pipeline.py")` |
| `introspect_schema` | Schema profiler: semantic types, cardinality, null rates, samples |
| `create_router` / `Router` | Library-side routing with confidence enforcement |
| `LocalEncoderStrategy` / `HostedApiStrategy` | The two runtime strategies |
| `FallbackException` | Raised when `max_probability < 0.40` |
| `CompilationError` | Synthesis failed after the self-correction retry budget |

## How the self-correction loop works

Following the STROT paper's feedback-driven refinement phase, the compiler
validates every LLM response before accepting it — syntax (`ast.parse`), the
output contract (one `execute_<key>` function per registry entry), and an
import blocklist (execution paths must stay stdlib-pure). On failure, the
error trace and the rejected code are appended to the original prompt and the
synthesizer retries, up to `max_retries` attempts, after which
`CompilationError` is raised.

## Development

```bash
poetry install --extras all
poetry run pytest
```

## License

MIT — see [LICENSE](LICENSE).
