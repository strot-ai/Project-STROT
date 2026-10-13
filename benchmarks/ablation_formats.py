"""Ablation: is the 71% ceiling Julia's capability or STROT's input formatting?

Runs the real Julia-1 engine directly (no STROT compile, no LLM) over the same
398 cars with ground truth, varying only state and option formatting.
"""

import csv
import json
import os
import statistics

from julia import load_model

QUESTION = "Which market of origin does this car belong to?"

VARIANTS = {
    "A: current STROT format (full JSON state, verbose 'key -- desc' options)": {
        "state": lambda c: json.dumps(c, default=str, sort_keys=True),
        "options": [
            "usa_market_review -- Cars from usa domestic American manufacturers",
            "europe_market_review -- Cars from europe European import manufacturers",
            "japan_market_review -- Cars from japan Japanese import manufacturers",
        ],
    },
    "B: full JSON state, plain labels": {
        "state": lambda c: json.dumps(c, default=str, sort_keys=True),
        "options": ["usa", "europe", "japan"],
    },
    "C: projected state (origin+name only), verbose options": {
        "state": lambda c: json.dumps({"origin": c["origin"], "name": c["name"]}),
        "options": [
            "usa_market_review -- Cars from usa domestic American manufacturers",
            "europe_market_review -- Cars from europe European import manufacturers",
            "japan_market_review -- Cars from japan Japanese import manufacturers",
        ],
    },
    "D: projected state, plain labels": {
        "state": lambda c: json.dumps({"origin": c["origin"], "name": c["name"]}),
        "options": ["usa", "europe", "japan"],
    },
}

ANSWER_INDEX = {"usa": 0, "europe": 1, "japan": 2}


def main():
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
    truths = [ANSWER_INDEX[c["origin"]] for c in cars]

    engine = load_model(os.environ.get("STROT_NLU_CHECKPOINT", "./julia-1"), device="cpu")
    print(f"{len(cars)} cars, real Julia-1, question: {QUESTION!r}\n")

    for label, spec in VARIANTS.items():
        rows = [
            {"state": spec["state"](c), "question": QUESTION, "options": spec["options"]}
            for c in cars
        ]
        results = engine.predict(rows=rows)
        correct = confident_wrong = fallbacks = 0
        confs = []
        for res, truth in zip(results, truths):
            conf = float(res["probabilities"][res["index"]])
            confs.append(conf)
            if conf < 0.40:
                fallbacks += 1
                continue
            if int(res["index"]) == truth:
                correct += 1
            elif conf >= 0.8:
                confident_wrong += 1
        routed = len(cars) - fallbacks
        print(label)
        print(
            f"  accuracy {correct}/{routed} = {100*correct/routed:.1f}%  "
            f"fallbacks {fallbacks}  conf-wrong>=0.8: {confident_wrong}  "
            f"mean conf {statistics.mean(confs):.3f}\n"
        )


if __name__ == "__main__":
    main()
