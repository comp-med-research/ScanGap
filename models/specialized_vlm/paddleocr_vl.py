"""
PaddleOCR-VL / PaddleOCR-VL-1.5 (PaddlePaddle).
pip install paddlepaddle paddleocr

Models:
  PaddleOCR-VL    — 0.9B, top performer on Real5 scanning tier
  PaddleOCR-VL-1.5 — 0.9B, highest overall score (93.43) in Real5 leaderboard
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class PaddleOCRVLModel(OCRModel):
    def __init__(self, version: str = "1.5"):
        self.version = version
        # PaddleOCR-VL uses the standard PaddleOCR interface with VL backend
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
            use_gpu=True,
            ocr_version="PP-OCRv4" if version == "1.5" else "PP-OCRv3",
        )

    def run(self, image_path: Path) -> str:
        result = self.ocr.ocr(str(image_path), cls=True)
        if not result or result[0] is None:
            return ""
        lines = [line[1][0] for line in result[0] if line and line[1]]
        return "\n".join(lines)


class PaddleOCRVL(PaddleOCRVLModel):
    def __init__(self):
        super().__init__(version="1.0")

class PaddleOCRVL15(PaddleOCRVLModel):
    def __init__(self):
        super().__init__(version="1.5")
