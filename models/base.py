"""
Base class for all OCR models in ScanGap.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path

OCR_PROMPT = (
    "Transcribe all text in this document image exactly as it appears. "
    "Preserve line breaks, spacing, and reading order. "
    "Output only the transcribed text — no commentary, no markdown formatting."
)


class OCRModel(ABC):
    """All models implement run(image_path) -> str."""

    @abstractmethod
    def run(self, image_path: Path) -> str:
        pass

    @staticmethod
    def encode_image_b64(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def mime_type(image_path: Path) -> str:
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".tiff": "image/tiff",
                ".bmp": "image/bmp",  ".webp": "image/webp"}.get(
            image_path.suffix.lower(), "image/jpeg"
        )
