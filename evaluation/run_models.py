"""
Run OCR models across datasets and save raw predictions.

Datasets:
  - omnidocbench      : OmniDocBench (fuzzy_scan=true/false split)
  - real5             : Real5-OmniDocBench scanning tier
  - funsd             : FUNSD scanned noisy forms
  - wildscans         : ScanGap wild historical pages (IA, LOC, Court Listener)

Models:
  - tesseract         : Tesseract 5
  - paddleocr         : PaddleOCR
  - surya             : Surya
  - gpt4o             : GPT-4o (vision)
  - gemini            : Gemini 2.0 Flash
  - qwen              : Qwen2.5-VL
"""

import argparse
import json
import os
from pathlib import Path


SUPPORTED_DATASETS = ["omnidocbench", "real5", "funsd", "wildscans"]
SUPPORTED_MODELS = ["tesseract", "paddleocr", "surya", "gpt4o", "gemini", "qwen"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run OCR models across datasets")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of page images")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="Where to save predictions")
    return parser.parse_args()


def run_tesseract(image_path: Path) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(image_path))


def run_paddleocr(image_path: Path) -> str:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(str(image_path), cls=True)
    lines = [word_info[1][0] for line in result for word_info in line]
    return "\n".join(lines)


def run_surya(image_path: Path) -> str:
    raise NotImplementedError("Surya integration: see https://github.com/VikParuchuri/surya")


def run_gpt4o(image_path: Path) -> str:
    raise NotImplementedError("Set OPENAI_API_KEY and implement vision call")


def run_gemini(image_path: Path) -> str:
    raise NotImplementedError("Set GOOGLE_API_KEY and implement vision call")


def run_qwen(image_path: Path) -> str:
    raise NotImplementedError("Qwen2.5-VL integration: see https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct")


MODEL_FNS = {
    "tesseract": run_tesseract,
    "paddleocr": run_paddleocr,
    "surya": run_surya,
    "gpt4o": run_gpt4o,
    "gemini": run_gemini,
    "qwen": run_qwen,
}


def main():
    args = parse_args()
    out_dir = args.output_dir / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {args.input_dir}")

    model_fn = MODEL_FNS[args.model]
    predictions = {}

    for img_path in image_paths:
        print(f"  Processing {img_path.name} ...")
        try:
            text = model_fn(img_path)
            predictions[img_path.name] = {"prediction": text, "error": None}
        except Exception as e:
            predictions[img_path.name] = {"prediction": None, "error": str(e)}

    out_file = out_dir / "predictions.json"
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to {out_file}")


if __name__ == "__main__":
    main()
