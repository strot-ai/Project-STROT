"""Real-model benchmark: strot.run() with the actual Julia-1 router.

Runs inside the strot-julia image (real supersonic-julia + 550MB checkpoint,
STROT_NLU_CHECKPOINT baked in). Compile phase uses the Anthropic API.

Pass A — origin field present in each record: measures routing mechanics and
confidence calibration on a grounded choice (ground truth = origin).
Pass B — origin field stripped: measures semantic inference (brand -> market)
where the answer is NOT in the state; Julia's model card predicts weakness here.
"""

import csv
import json
import statistics
import time

import strot

QUERY = (
    "Evaluate each car record for a market analysis, routed by the car's "
    "market of origin."
)

ACTIONS = {
    "usa_market_review": "Cars from usa domestic American manufacturers",
    "europe_market_review": "Cars from europe European import manufacturers",
    "japan_market_review": "Cars from japan Japanese import manufacturers",
}

KEY_TO_ORIGIN = {
    "usa_market_review": "usa",
    "europe_market_review": "europe",
    "japan_market_review": "japan",
}


def load_cars():
    cars = []
    with open("mpg.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            cars.append(
                {
                    "mpg": float(row["mpg"]) if row["mpg"] else None,
                    "cylinders": int(row["cylinders"]),
                    "horsepower": float(row["horsepower"]) if row["horsepower"] else None,
                    "weight": int(row["weight"]),
                    "model_year": int(row["model_year"]),
                    "origin": row["origin"],
                    "name": row["name"],
                }
            )
    return cars


def report(label, result, truths):
    routed = correct = 0
    confidences = []
    for outcome, truth in zip(result.outcomes, (truths[o["index"]] for o in result.outcomes)):
        routed += 1
        confidences.append(outcome["confidence"])
        if KEY_TO_ORIGIN[outcome["action"]] == truth:
            correct += 1
    total = len(truths)
    print(f"--- {label} ---")
    print(f"  routed: {routed}/{total}  fallbacks: {len(result.fallbacks)}")
    if routed:
        print(f"  accuracy (on routed): {correct}/{routed} = {100*correct/routed:.1f}%")
        print(f"  confidence: mean {statistics.mean(confidences):.3f}  "
              f"min {min(confidences):.3f}  max {max(confidences):.3f}")
    wrong_but_confident = sum(
        1 for o in result.outcomes
        if KEY_TO_ORIGIN[o["action"]] != truths[o["index"]] and o["confidence"] >= 0.8
    )
    print(f"  misroutes at confidence >= 0.8: {wrong_but_confident}")


def run_pass(label, records, truths):
    start = time.time()
    result = strot.run(
        records,
        query=QUERY,
        action_registry=ACTIONS,
        llm="anthropic/claude-opus-4-8",
        state=".strot-bench",
    )
    compile_time = time.time() - start
    print(f"{label}: cache_hit={result.cache_hit} attempts={result.attempts} "
          f"total {compile_time:.1f}s for {len(records)} records")
    # Warm pass to time pure routing (artifact already cached)
    start = time.time()
    strot.run(records, query=QUERY, action_registry=ACTIONS,
              llm="anthropic/claude-opus-4-8", state=".strot-bench")
    per_record = (time.time() - start) / len(records) * 1000
    print(f"  warm routing: {per_record:.1f} ms/record on CPU")
    report(label, result, truths)
    print()


def main():
    cars = load_cars()
    truths = [c["origin"] for c in cars]
    print(f"dataset: {len(cars)} cars; ground truth origins: "
          f"{ {o: truths.count(o) for o in set(truths)} }\n")

    run_pass("PASS A (origin in state - grounded choice)", cars, truths)

    stripped = [{k: v for k, v in c.items() if k != "origin"} for c in cars]
    run_pass("PASS B (origin stripped - semantic inference)", stripped, truths)


if __name__ == "__main__":
    main()
