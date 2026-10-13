"""Semantic decision benchmark: unstructured text, answer NOT a structured field.

Datasets with published ground-truth labels (no human labeling on our side):
- AG News (4 classes, news topic)
- DAIR Emotion (6 classes, tweets)

Conditions: naive (raw text + class names into the decision model) vs
STROT-compiled (synthesized labels/projection; compiler never sees test
labels) vs compiled + train-goldens (one labeled train example per class).
"""

import json
import os
import statistics
import time
import urllib.request

import strot
from julia import load_model

HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset={d}&config={c}&split={s}&offset={o}&length=100"


def fetch(dataset, config, split, n):
    rows = []
    for offset in range(0, n, 100):
        url = HF_ROWS.format(d=dataset, c=config, s=split, o=offset)
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.load(resp)
        rows += [r["row"] for r in payload["rows"]]
    return rows[:n]


BENCHES = {
    "ag_news": {
        "dataset": "fancyzhx/ag_news", "config": "default",
        "classes": ["world", "sports", "business", "sci_tech"],
        "registry": {
            "world": "World news, politics, international affairs, conflict",
            "sports": "Sports, games, athletes, matches, tournaments",
            "business": "Business, markets, economy, companies, finance",
            "sci_tech": "Science and technology, research, software, space",
        },
        "query": "Classify each news snippet into its topic category.",
    },
    "emotion": {
        "dataset": "dair-ai/emotion", "config": "split",
        "classes": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "registry": {
            "sadness": "Sadness, grief, disappointment, loneliness",
            "joy": "Joy, happiness, contentment, excitement",
            "love": "Love, affection, caring, tenderness",
            "anger": "Anger, rage, annoyance, frustration",
            "fear": "Fear, anxiety, worry, terror",
            "surprise": "Surprise, amazement, shock",
        },
        "query": "Classify the emotion expressed in each message.",
    },
}

N_TEST = 300
engine = load_model(os.environ.get("STROT_NLU_CHECKPOINT", "./julia-1"), device="cpu")


def naive_accuracy(texts, truths, classes, question):
    correct = 0
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        rows = [{"state": t, "question": question, "options": classes} for t in batch]
        for res, truth in zip(engine.predict(rows=rows), truths[i:i + 50]):
            if int(res["index"]) == truth:
                correct += 1
    return correct / len(texts)


for name, bench in BENCHES.items():
    print(f"\n================ {name} ================")
    train = fetch(bench["dataset"], bench["config"], "train", 600)
    test = fetch(bench["dataset"], bench["config"], "test", N_TEST)
    classes = bench["classes"]

    test_texts = [r["text"] for r in test]
    test_truth = [int(r["label"]) for r in test]

    # --- Condition A: naive (no STROT) ---
    t0 = time.time()
    naive = naive_accuracy(test_texts, test_truth,
                           [c.replace("_", " ") for c in classes], bench["query"])
    print(f"A naive (raw text + class names):      {naive*100:5.1f}%  ({time.time()-t0:.0f}s)")

    # --- Condition B: STROT-compiled, no goldens ---
    sample = [{"text": r["text"]} for r in train[:25]]
    records = [{"text": t} for t in test_texts]
    result = strot.run(
        sample + records, bench["query"], bench["registry"],
        llm="anthropic/claude-opus-4-8", state=f".strot-bench/sb_{name}_b",
        sample_size=25,
    )
    routed = {o["index"]: o["action"] for o in result.outcomes}
    idx = {c: i for i, c in enumerate(classes)}
    correct = sum(
        1 for i, truth in enumerate(test_truth)
        if routed.get(i + 25) is not None and idx[routed[i + 25]] == truth
    )
    fb = sum(1 for f in result.fallbacks if f["index"] >= 25)
    print(f"B STROT-compiled (attempts={result.attempts}):        "
          f"{correct/N_TEST*100:5.1f}%  fallbacks={fb}")

    # --- Condition C: compiled + one train-golden per class ---
    golden = []
    for class_index, key in enumerate(classes):
        example = next((r for r in train if int(r["label"]) == class_index), None)
        if example is not None:
            golden.append({"record": {"text": example["text"]}, "expect": key})
    print(f"  (goldens found for {len(golden)}/{len(classes)} classes)")
    result_g = strot.run(
        sample + records, bench["query"], bench["registry"],
        llm="anthropic/claude-opus-4-8", state=f".strot-bench/sb_{name}_c",
        sample_size=25, golden=golden,
    )
    routed_g = {o["index"]: o["action"] for o in result_g.outcomes}
    correct_g = sum(
        1 for i, truth in enumerate(test_truth)
        if routed_g.get(i + 25) is not None and idx[routed_g[i + 25]] == truth
    )
    fb_g = sum(1 for f in result_g.fallbacks if f["index"] >= 25)
    print(f"C compiled + train-goldens (attempts={result_g.attempts}): "
          f"{correct_g/N_TEST*100:5.1f}%  fallbacks={fb_g}")
