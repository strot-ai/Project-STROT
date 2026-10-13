# Benchmarks

Reproducible scripts behind the numbers reported in the main README.

## Routing accuracy: `routing_mpg.py`

The 398-record grounded-routing benchmark (the 74.8% → 92.5% result).

**Dataset:** [seaborn's `mpg.csv`](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv) — 398 cars with a ground-truth `origin` column (usa 249 / japan 79 / europe 70).

```bash
curl -sLO https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
pip install strot-ai[local,anthropic]
python -c "import strot; strot.prefetch_nlu(local_dir='./julia-1', install=True)"
export STROT_NLU_CHECKPOINT=./julia-1 ANTHROPIC_API_KEY=...
python routing_mpg.py
```

**Task:** route each car to one of three market workflows by origin. Two passes:

- **Pass A** — `origin` present in the record (grounded choice; ground truth known per record)
- **Pass B** — `origin` stripped (semantic inference; documents the small decision model's known weakness — expect ~50%, do not use this mode in production)

**Methodology notes (read before quoting numbers):**

- The compiler **never sees the ground-truth labels**. Routing labels and the state projection are synthesized from the action-registry descriptions and the data schema alone; accuracy is then measured against the held-out `origin` column. No goldens were used in the headline run.
- The separately reported **100% "ceiling"** came from hand-tuning option wording *against this metric* — it demonstrates the headroom of the same model, not an unbiased result. The honest headline is the compiled figure.
- **Compile variance is real**: synthesis is stochastic, so repeat runs produce different artifacts with different accuracy (we have observed meaningful spread). The golden-record gate (`golden=`) exists precisely to bound this — production use should always pin known answers.
- Reported environment: `python:3.12-slim` Docker container, CPU only (Apple-silicon host via Docker Desktop), `SupersonicLabs/Julia-1` checkpoint, synthesizer `claude-opus-4-8`, strot-ai v0.3.0+ (labels/projection contract). Latency figures are per-record wall-clock over the full pass after model load.
- The naive-integration baseline (74.8%) and the input-formatting ablation behind the compiled/naive comparison are in `ablation_formats.py`.

## Semantic decisions over unstructured text: `semantic_text.py`

Does compilation help when the answer is **not** a structured field? Two public
datasets with their own ground-truth labels (AG News, 4-class news topic;
DAIR Emotion, 6-class tweets); 300 held-out test records each; the compiler
never sees any test labels. Three conditions: naive (raw text + bare class
names into the decision model), STROT-compiled, and compiled + one labeled
train example per class as goldens.

Measured 2026-10-13 (same environment as above):

| Dataset | Naive | STROT-compiled | + train-goldens |
|---|---|---|---|
| AG News | 72.7% | **86.7%** (+14.0) | 80.3% |
| Emotion | **93.3%** | 85.0% (−8.3) | 70.0% |

**Honest findings, including where STROT loses:**

1. Compilation delivers large gains when integration is the bottleneck
   (AG News: the synthesized labels/projection beat bare class names by 14
   points on text the model must genuinely read).
2. Compilation can *hurt* when the naive interface is already optimal
   (Emotion tweets literally contain the class words, so bare class names are
   near-perfect labels and synthesized wording diluted them). The compiler
   does not yet *measure* candidate labelings against the sample — a known
   roadmap item these numbers motivate directly.
3. Using goldens as an accuracy optimizer backfired in both datasets:
   gate-driven retries selected wording that satisfied a handful of examples
   but generalized worse. Goldens are a **regression gate** against broken or
   variant compiles — not an optimization signal. Use clear, unambiguous
   goldens; don't tune accuracy with them.

## Input-formatting ablation: `ablation_formats.py`

Same model, same records, same question — only state/option formatting varies. This isolates how much of routing accuracy is integration rather than model capability (answer: ~25 points). Runs the decision model directly; no LLM required.
