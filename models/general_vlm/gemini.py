"""
Google Gemini vision models.
pip install google-genai
Requires: GOOGLE_API_KEY env var

Models:
  gemini-3.1-pro-preview  — latest (April 2026); Gemini 3 Pro was deprecated
                             March 2026, gemini-3-pro-preview now aliases this
  gemini-2.5-pro          — used in Real5-OmniDocBench leaderboard; still available
"""

import os
from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class GeminiModel(OCRModel):
    def __init__(self, model: str = "gemini-3.1-pro-preview"):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model

    def run(self, image_path: Path) -> str:
        from google.genai import types
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=self.mime_type(image_path)),
                OCR_PROMPT,
            ],
        )
        return response.text


class Gemini31Pro(GeminiModel):
    def __init__(self):
        super().__init__(model="gemini-3.1-pro-preview")

class Gemini25Pro(GeminiModel):
    def __init__(self):
        super().__init__(model="gemini-2.5-pro")
