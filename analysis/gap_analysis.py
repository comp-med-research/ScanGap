"""
Stratified gap analysis across datasets and models.

Computes NED/CER/WER per dataset tier, per model, and produces
the degradation spectrum table for the paper.

Expected results layout:
  results/
    {dataset}/
      {model}/
        predictions.json   # output of run_models.py
    {dataset}/
      ground_truth.json    # {filename: gt_text}

Real5-OmniDocBench scanning tier scores are loaded from:
  data/real5_baseline.json  (pre-published, no inference needed)
"""

import json
from pathlib import Path

import pandas as pd
from evaluation.metrics import score_page

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")

# Ordered from least to most degraded — defines the spectrum axis in the paper
DATASET_ORDER = [
    ("omnidocbench_digital",  "OmniDocBench (digital-native)"),
    ("omnidocbench_scanned",  "OmniDocBench (fuzzy_scan, n=28)"),
    ("real5_scan",            "Real5-OmniDocBench (scan tier)"),
    ("funsd",                 "FUNSD"),
    ("wildscans",             "ScanGap (wild historical)"),
]


def load_ground_truth(dataset: str) -> dict:
    gt_path = RESULTS_DIR / dataset / "ground_truth.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        return json.load(f)


def load_predictions(dataset: str, model: str) -> dict:
    pred_path = RESULTS_DIR / dataset / model / "predictions.json"
    if not pred_path.exists():
        return {}
    with open(pred_path) as f:
        return json.load(f)


def score_dataset(dataset: str, model: str) -> dict | None:
    gt    = load_ground_truth(dataset)
    preds = load_predictions(dataset, model)
    if not gt or not preds:
        return None

    scores = []
    for filename, gt_text in gt.items():
        entry = preds.get(filename)
        if entry is None or entry["prediction"] is None:
            continue
        scores.append(score_page(entry["prediction"], gt_text))

    if not scores:
        return None

    return {
        "ned": sum(s["ned"] for s in scores) / len(scores),
        "cer": sum(s["cer"] for s in scores) / len(scores),
        "wer": sum(s["wer"] for s in scores) / len(scores),
        "n":   len(scores),
    }


def load_real5_baselines() -> dict:
    """Return {model_key: {text_ned, overall, ...}} from pre-published results."""
    path = DATA_DIR / "real5_baseline.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_spectrum_table() -> pd.DataFrame:
    from evaluation.run_models import MODELS

    real5 = load_real5_baselines()
    rows  = []

    for dataset_key, dataset_label in DATASET_ORDER:

        # Real5 scores come from the published baseline, not our inference
        if dataset_key == "real5_scan":
            for model_key, meta in MODELS.items():
                baseline = real5.get(model_key)
                if baseline is None:
                    continue
                rows.append({
                    "dataset":  dataset_label,
                    "model":    meta["label"],
                    "category": meta["category"],
                    "params":   meta["params"],
                    "n":        1355,
                    "NED":      round(baseline["text_ned"], 4),
                    "CER":      None,
                    "WER":      None,
                    "overall_real5": round(baseline["overall"], 2),
                })
            continue

        for model_key, meta in MODELS.items():
            result = score_dataset(dataset_key, model_key)
            if result is None:
                continue
            rows.append({
                "dataset":       dataset_label,
                "model":         meta["label"],
                "category":      meta["category"],
                "params":        meta["params"],
                "n":             result["n"],
                "NED":           round(result["ned"], 4),
                "CER":           round(result["cer"], 4),
                "WER":           round(result["wer"], 4),
                "overall_real5": None,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_spectrum_table()
    if df.empty:
        print("No results found. Run evaluation/run_models.py first.")
    else:
        print(df.to_string(index=False))
        out = RESULTS_DIR / "gap_analysis.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved to {out}")
