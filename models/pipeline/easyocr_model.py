"""
EasyOCR (JaidedAI).
pip install easyocr

Supports 80+ languages. Uses CRAFT for detection and CRNN for recognition.
GPU is used automatically if available.
"""

from pathlib import Path
from models.base import OCRModel


class EasyOCRModel(OCRModel):
    def __init__(self, langs: list[str] = None):
        import easyocr
        self.reader = easyocr.Reader(
            langs or ["en"],
            gpu=True,
            verbose=False,
        )

    def run(self, image_path: Path) -> str:
        results = self.reader.readtext(str(image_path), detail=0, paragraph=True)
        return "\n".join(results)
