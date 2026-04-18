"""
Anthropic Claude vision models.
pip install anthropic
Requires: ANTHROPIC_API_KEY env var

Models:
  claude-opus-4-6    — latest flagship (April 2026), strongest reasoning + vision
  claude-sonnet-4-6  — balanced speed/quality
"""

import os
from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class ClaudeModel(OCRModel):
    def __init__(self, model: str = "claude-opus-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def run(self, image_path: Path) -> str:
        b64 = self.encode_image_b64(image_path)
        mime = self.mime_type(image_path)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }],
        )
        return response.content[0].text


class ClaudeOpus46(ClaudeModel):
    def __init__(self):
        super().__init__(model="claude-opus-4-6")

class ClaudeSonnet46(ClaudeModel):
    def __init__(self):
        super().__init__(model="claude-sonnet-4-6")
