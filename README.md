# STROT

**The open decision compiler.** Compile your intent once with an LLM. Then decide on every record forever — classification, scoring, routing — as type-safe JSON with probabilities, confidence, and a fitted statistical guarantee, on your own hardware, at zero marginal cost.

```
STROT = Agentic Schema Intent + Compiler Creation & State + Code + Execution + NLU
      ⟹ JSON Decisions (classification · scoring · routing)
```

Agents at compile time. **Never at runtime.**

```bash
pip install strot-ai
```

## How it works

```
                    COMPILE (once per intent + schema)
  intent + sample ──▶ agent synthesizes config ──▶ golden gate ──▶ teacher labels
                                                   (your known      (LLM annotates
                                                    answers must     your real sample,
                                                    pass)            once)
                      ──▶ calibration fitted ──▶ frozen artifact (.py)
                          (temperature +
                           conformal threshold)

                    RUNTIME (every record, forever)
  record ──▶ projection ──▶ local decision model ──▶ conformal gate ──▶ typed JSON
             (compiled)     (144M, CPU, ~60ms)       (dispatch or        decision
                                                      abstain)
```

The LLM runs **once**, at compile time. Production is a frozen, self-contained Python artifact: a small local decision model arbitrates every record, nothing is ever generated, every output is bounded, and low-confidence records abstain instead of guessing.

## Decide: typed questions over records

```python
import strot

result = strot.decide(
    records,
    questions={
        "is_bug_report": {"type": "boolean",
            "instructions": "Does this mention malfunctions, errors, or crashes?"},
        "customer_sentiment": {"type": "score", "min": 1, "max": 5,
            "instructions": "1 is furious, 5 is delighted."},
        "routing_bucket": {"type": "choice",
            "options": ["billing_team", "engineering_team", "product_feedback", "spam"]},
    },
    llm="anthropic/claude-opus-4-8",   # used once, on cache miss only
)

result.outcomes[0]["decision"]
# {"booleans": {"is_bug_report": {"type": "boolean", "probability": 0.9998}},
#  "scores":   {"customer_sentiment": {"value": 1.79, "confidence": 0.899}},
#  "choices":  {"routing_bucket": {"selected": "engineering_team",
#               "probabilities": {...}, "confidence": 0.9992}}}
```

## Run: route and dispatch, with a guarantee

```python
result = strot.run(
    tickets,
    query="Triage incoming support tickets into the correct resolution workflow.",
    action_registry={
        "billing_dispute":  "Refunds, invoice corrections, double charges",
        "technical_defect": "Crashes, bugs, broken behavior",
        "account_lifecycle": "Deletion, data export, GDPR requests",
    },
    llm="anthropic/claude-opus-4-8",
    golden=[  # promotion gate: artifacts must pass your known answers
        {"record": {"body": "I was double charged"}, "expect": "billing_dispute"},
    ],
    target_risk=0.1,  # fitted guarantee: wrong dispatches ≤ 10% of all records (marginal); ambiguous records abstain
)

result.summary()
# {"processed": 13, "fallbacks": 1, "actions": {...}, "cache_hit": False, ...}
```

The first call compiles (seconds, cents). Every later call — and every restart sharing the `state` directory — is a cache hit: no LLM, no API key, pure execution.

## Measured

Real model (SupersonicLabs/Julia-1, 144M, CPU), real datasets, clean Docker containers:

| Metric | Result |
|---|---|
| Grounded routing accuracy (398 records, ground truth) | **92.5%** compiled (74.8% naive — the compiler is worth +18pp on the same model) |
| Hand-tuned ceiling, same model | 100% |
| Typed decisions vs the leading hosted decision API's own example | near-identical (score 1.79 vs 1.8) |
| Calibrated dispatch (α=0.15, 14 records) | 13 dispatched, all correct; 1 ambiguous record abstained (guarantee is marginal — see [DEVELOPERS.md](DEVELOPERS.md)) |
| Latency | ~60 ms/record, CPU, in-process |
| Marginal cost per decision | $0 |
| LLM calls per runtime record | 0 |

## Install

```bash
pip install strot-ai            # core: pure stdlib
pip install strot-ai[local]     # + torch/transformers for the local decision model
pip install strot-ai[hosted]    # + openai for a hosted decision runtime
pip install strot-ai[anthropic] # + anthropic for the synthesizer
```

The default local decision model (`SupersonicLabs/Julia-1`, Apache-2.0) ships its own inference package inside its Hugging Face repository. Bake it into your image at build time:

```dockerfile
RUN pip install strot-ai[local] \
 && python -c "import strot; strot.prefetch_nlu(local_dir='/opt/julia-1', install=True)"
ENV STROT_NLU_CHECKPOINT=/opt/julia-1
```

After that, production runs fully offline. (Never `pip install julia` — that's an unrelated package.)

## What makes it different

- **Compile once, decide forever** — content-addressed artifact cache; schema drift automatically triggers exactly one recompile
- **Golden-record promotion** — a candidate artifact is executed against your known answers inside the compile loop; only passing artifacts are ever served
- **Fitted guarantees** — temperature scaling + split conformal prediction (fitted on disjoint folds to preserve validity), baked into the artifact. The guarantee is marginal: under exchangeability, wrong dispatches are ≤ α of all records; non-singleton prediction sets abstain. The conditional-on-dispatch bound (α / dispatch rate) is reported in the artifact's calibration metadata
- **0% structural hallucination** — output is selection + computation over your registry; nothing is generated at runtime
- **Model-independent** — the decision model is a slot (`nlu=`): local Julia-1 by default, hosted APIs, or any drop-in selector; the compiler, gates, and calibration keep their value across all of them
- **Every compile is logged** (`episodes.jsonl`) — the dataset for future compile-policy training

Reproduce these numbers: [benchmarks/](benchmarks/).

## Documentation

See [DEVELOPERS.md](DEVELOPERS.md) for the developer guide: architecture, artifact anatomy, calibration semantics and caveats, deployment patterns, model swapping, and the API reference.

## License

MIT. Default decision model: Apache-2.0 ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)). Implements ideas from STROT (arXiv:2505.01636).
