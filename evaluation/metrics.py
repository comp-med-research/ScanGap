"""
Metrics for evaluating OCR output against ground truth.
"""

import unicodedata
import editdistance


def normalise(text: str) -> str:
    """Lowercase, unicode-normalise, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


def ned(prediction: str, ground_truth: str) -> float:
    """Normalised Edit Distance (NED). Lower is better. Range [0, 1]."""
    pred = normalise(prediction)
    gt = normalise(ground_truth)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / max(len(pred), len(gt))


def cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate (CER). Lower is better."""
    pred = normalise(prediction)
    gt = normalise(ground_truth)
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)


def wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate (WER). Lower is better."""
    pred_tokens = normalise(prediction).split()
    gt_tokens = normalise(ground_truth).split()
    if len(gt_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0
    return editdistance.eval(pred_tokens, gt_tokens) / len(gt_tokens)


def score_page(prediction: str, ground_truth: str) -> dict:
    """Return all three metrics for a single page."""
    return {
        "ned": ned(prediction, ground_truth),
        "cer": cer(prediction, ground_truth),
        "wer": wer(prediction, ground_truth),
    }
