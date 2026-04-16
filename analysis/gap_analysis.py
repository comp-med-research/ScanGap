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
"""

import json
from pathlib import Path

import pandas as pd
from evaluation.metrics import score_page


# Ordered from least to most degraded — defines the spectrum axis
DATASET_ORDER = [
    ("omnidocbench_digital", "OmniDocBench (digital)"),
    ("omnidocbench_scanned", "OmniDocBench (fuzzy_scan)"),
    ("real5",                "Real5-OmniDocBench (scan tier)"),
    ("funsd",                "FUNSD"),
    ("wildscans",            "ScanGap (wild historical)"),
]

MODELS = ["tesseract", "paddleocr", "surya", "gpt4o", "gemini", "qwen"]


def load_ground_truth(results_dir: Path, dataset: str) -> dict:
    gt_path = results_dir / dataset / "ground_truth.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        return json.load(f)


def load_predictions(results_dir: Path, dataset: str, model: str) -> dict:
    pred_path = results_dir / dataset / model / "predictions.json"
    if not pred_path.exists():
        return {}
    with open(pred_path) as f:
        return json.load(f)


def score_dataset(results_dir: Path, dataset: str, model: str) -> dict | None:
    gt = load_ground_truth(results_dir, dataset)
    preds = load_predictions(results_dir, dataset, model)
    if not gt or not preds:
        return None

    scores = []
    for filename, gt_text in gt.items():
        pred_entry = preds.get(filename)
        if pred_entry is None or pred_entry["prediction"] is None:
            continue
        scores.append(score_page(pred_entry["prediction"], gt_text))

    if not scores:
        return None

    return {
        "ned":   sum(s["ned"] for s in scores) / len(scores),
        "cer":   sum(s["cer"] for s in scores) / len(scores),
        "wer":   sum(s["wer"] for s in scores) / len(scores),
        "n":     len(scores),
    }


def build_spectrum_table(results_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset_key, dataset_label in DATASET_ORDER:
        for model in MODELS:
            result = score_dataset(results_dir, dataset_key, model)
            if result is None:
                continue
            rows.append({
                "dataset":  dataset_label,
                "model":    model,
                "n":        result["n"],
                "NED":      round(result["ned"], 4),
                "CER":      round(result["cer"], 4),
                "WER":      round(result["wer"], 4),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results_dir = Path("results")
    df = build_spectrum_table(results_dir)
    if df.empty:
        print("No results found. Run evaluation/run_models.py first.")
    else:
        print(df.to_string(index=False))
        df.to_csv(results_dir / "gap_analysis.csv", index=False)
        print(f"\nSaved to {results_dir / 'gap_analysis.csv'}")
