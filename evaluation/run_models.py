"""
Run OCR models across datasets and save raw predictions.

Usage:
  python -m evaluation.run_models --dataset funsd --model paddleocr_vl_1_5 --input_dir data/funsd/images

Datasets:
  omnidocbench_digital  : OmniDocBench (fuzzy_scan=false)
  omnidocbench_scanned  : OmniDocBench (fuzzy_scan=true, 28 pages)
  funsd                 : FUNSD scanned noisy forms
  wildscans             : ScanGap wild historical pages (IA, LOC, Court Listener)
  (real5 scores are pre-published — stored in data/real5_baseline.json)
"""

import argparse
import json
from pathlib import Path


SUPPORTED_DATASETS = [
    "omnidocbench_digital",
    "omnidocbench_scanned",
    "funsd",
    "wildscans",
]

# Model registry — maps CLI key to (label, params, category, loader)
# Loader is a zero-arg callable that returns an OCRModel instance
MODELS = {
    # ── Pipeline tools ────────────────────────────────────────────────────────
    "tesseract": {
        "label": "Tesseract 5", "params": None, "category": "pipeline",
        "loader": lambda: __import__("models.pipeline.tesseract_model", fromlist=["TesseractModel"]).TesseractModel(),
    },
    "easyocr": {
        "label": "EasyOCR", "params": None, "category": "pipeline",
        "loader": lambda: __import__("models.pipeline.easyocr_model", fromlist=["EasyOCRModel"]).EasyOCRModel(),
    },
    "doctr": {
        "label": "DocTR", "params": None, "category": "pipeline",
        "loader": lambda: __import__("models.pipeline.doctr_model", fromlist=["DocTRModel"]).DocTRModel(),
    },
    "pp_structure_v3": {
        "label": "PP-StructureV3", "params": None, "category": "pipeline",
        "loader": lambda: __import__("models.pipeline.pp_structure_v3", fromlist=["PPStructureV3"]).PPStructureV3(),
    },
    "docling": {
        "label": "Docling", "params": None, "category": "pipeline",
        "loader": lambda: __import__("models.pipeline.docling_model", fromlist=["DoclingModel"]).DoclingModel(),
    },
    # ── General VLMs ─────────────────────────────────────────────────────────
    "gpt_5_2": {
        "label": "GPT-5.2", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.gpt", fromlist=["GPT52"]).GPT52(),
    },
    "gpt_5_4": {
        "label": "GPT-5.4", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.gpt", fromlist=["GPT54"]).GPT54(),
    },
    "claude_opus_46": {
        "label": "Claude Opus 4.6", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.claude", fromlist=["ClaudeOpus46"]).ClaudeOpus46(),
    },
    "claude_sonnet_46": {
        "label": "Claude Sonnet 4.6", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.claude", fromlist=["ClaudeSonnet46"]).ClaudeSonnet46(),
    },
    "gemini_25_pro": {
        "label": "Gemini-2.5 Pro", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.gemini", fromlist=["Gemini25Pro"]).Gemini25Pro(),
    },
    "gemini_31_pro": {
        "label": "Gemini-3.1 Pro", "params": None, "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.gemini", fromlist=["Gemini31Pro"]).Gemini31Pro(),
    },
    "qwen_vl_7b": {
        "label": "Qwen2.5-VL-7B", "params": "7B", "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.qwen_vl", fromlist=["QwenVL7B"]).QwenVL7B(),
    },
    "qwen25_vl_72b": {
        "label": "Qwen2.5-VL-72B", "params": "72B", "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.qwen_vl", fromlist=["QwenVL72B"]).QwenVL72B(),
    },
    "qwen3_vl_235b": {
        "label": "Qwen3-VL-235B-A22B", "params": "235B", "category": "general_vlm",
        "loader": lambda: __import__("models.general_vlm.qwen_vl", fromlist=["Qwen3VL235B"]).Qwen3VL235B(),
    },
    # ── Specialized VLMs ─────────────────────────────────────────────────────
    "dolphin": {
        "label": "Dolphin", "params": "322M", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.dolphin", fromlist=["Dolphin"]).Dolphin(),
    },
    "dolphin_1_5": {
        "label": "Dolphin-1.5", "params": "0.3B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.dolphin", fromlist=["Dolphin15"]).Dolphin15(),
    },
    "mineru2_vlm": {
        "label": "MinerU2-VLM", "params": "0.9B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.mineru", fromlist=["MinerU2VLM"]).MinerU2VLM(),
    },
    "mineru2_5": {
        "label": "MinerU2.5", "params": "1.2B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.mineru", fromlist=["MinerU25"]).MinerU25(),
    },
    "monkeyocr_pro_1b": {
        "label": "MonkeyOCR-pro-1.2B", "params": "1.9B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.monkeyocr", fromlist=["MonkeyOCRPro1B"]).MonkeyOCRPro1B(),
    },
    "monkeyocr_3b": {
        "label": "MonkeyOCR-3B", "params": "3.7B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.monkeyocr", fromlist=["MonkeyOCR3B"]).MonkeyOCR3B(),
    },
    "monkeyocr_pro_3b": {
        "label": "MonkeyOCR-pro-3B", "params": "3.7B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.monkeyocr", fromlist=["MonkeyOCRPro3B"]).MonkeyOCRPro3B(),
    },
    "nanonets_ocr_s": {
        "label": "Nanonets-OCR-s", "params": "3B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.nanonets_ocr", fromlist=["NanonetsOCRModel"]).NanonetsOCRModel(),
    },
    "deepseek_ocr": {
        "label": "Deepseek-OCR", "params": "3B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.deepseek_ocr", fromlist=["DeepSeekOCR"]).DeepSeekOCR(),
    },
    "deepseek_ocr_2": {
        "label": "Deepseek-OCR 2", "params": "3B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.deepseek_ocr", fromlist=["DeepSeekOCR2"]).DeepSeekOCR2(),
    },
    "dots_ocr": {
        "label": "dots.ocr", "params": "3B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.dots_ocr", fromlist=["DotsOCRModel"]).DotsOCRModel(),
    },
    "glm_ocr": {
        "label": "GLM-OCR", "params": "0.9B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.glm_ocr", fromlist=["GLMOCRModel"]).GLMOCRModel(),
    },
    "paddleocr_vl": {
        "label": "PaddleOCR-VL", "params": "0.9B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.paddleocr_vl", fromlist=["PaddleOCRVL"]).PaddleOCRVL(),
    },
    "paddleocr_vl_1_5": {
        "label": "PaddleOCR-VL-1.5", "params": "0.9B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.paddleocr_vl", fromlist=["PaddleOCRVL15"]).PaddleOCRVL15(),
    },
    "internvl35_8b": {
        "label": "InternVL3.5-8B", "params": "8B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.internvl", fromlist=["InternVL35_8B"]).InternVL35_8B(),
    },
    "internvl3_8b": {
        "label": "InternVL3-8B", "params": "8B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.internvl", fromlist=["InternVL3_8B"]).InternVL3_8B(),
    },
    "internvl3_78b": {
        "label": "InternVL3-78B", "params": "78B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.internvl", fromlist=["InternVL3_78B"]).InternVL3_78B(),
    },
    "olmocr2": {
        "label": "olmOCR-2-7B", "params": "7B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.olmocr", fromlist=["OlmOCR2"]).OlmOCR2(),
    },
    "olmocr2_fp8": {
        "label": "olmOCR-2-7B-FP8", "params": "7B", "category": "specialized_vlm",
        "loader": lambda: __import__("models.specialized_vlm.olmocr", fromlist=["OlmOCR2FP8"]).OlmOCR2FP8(),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR models across ScanGap datasets")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--model",   choices=list(MODELS.keys()), required=True)
    parser.add_argument("--input_dir",  type=Path, required=True,
                        help="Directory of page images (.png / .jpg)")
    parser.add_argument("--output_dir", type=Path, default=Path("results"),
                        help="Root results directory (default: results/)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg"))
    print(f"Dataset : {args.dataset}")
    print(f"Model   : {MODELS[args.model]['label']}")
    print(f"Images  : {len(image_paths)}")

    print("Loading model ...")
    model = MODELS[args.model]["loader"]()

    predictions = {}
    for img_path in image_paths:
        print(f"  {img_path.name} ...", end=" ", flush=True)
        try:
            text = model.run(img_path)
            predictions[img_path.name] = {"prediction": text, "error": None}
            print("ok")
        except Exception as e:
            predictions[img_path.name] = {"prediction": None, "error": str(e)}
            print(f"ERROR: {e}")

    out_file = out_dir / "predictions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(predictions)} predictions → {out_file}")


if __name__ == "__main__":
    main()
