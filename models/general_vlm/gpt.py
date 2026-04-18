"""
OpenAI GPT vision models.
pip install openai
Requires: OPENAI_API_KEY env var

Models:
  gpt-5.4       — latest (April 2026), strongest reasoning + vision
  gpt-5.2       — used in Real5-OmniDocBench leaderboard
"""

import os
from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class GPTModel(OCRModel):
    def __init__(self, model: str = "gpt-5.4"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def run(self, image_path: Path) -> str:
        b64 = self.encode_image_b64(image_path)
        mime = self.mime_type(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }],
            max_tokens=4096,
        )
        return response.choices[0].message.content


# Convenience aliases matching the leaderboard entries
class GPT52(GPTModel):
    def __init__(self):
        super().__init__(model="gpt-5.2")

class GPT54(GPTModel):
    def __init__(self):
        super().__init__(model="gpt-5.4")
