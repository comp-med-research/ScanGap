"""
Tesseract 5 (Google / open source).
pip install pytesseract Pillow
System dep: sudo apt install tesseract-ocr tesseract-ocr-eng

Uses LSTM engine (OEM 1) with automatic page segmentation (PSM 3).
"""

from pathlib import Path
from models.base import OCRModel


class TesseractModel(OCRModel):
    def __init__(self, lang: str = "eng", psm: int = 3):
        import pytesseract  # noqa: F401 — fail fast if not installed
        self.lang = lang
        self.config = f"--oem 1 --psm {psm}"

    def run(self, image_path: Path) -> str:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return pytesseract.image_to_string(image, lang=self.lang, config=self.config)
