"""
Deepseek-OCR / Deepseek-OCR 2 (DeepSeek).
pip install transformers accelerate torch

Models:
  deepseek-ai/deepseek-ocr    — 3B, original
  deepseek-ai/deepseek-ocr-2  — 3B, improved (highest text NED in Real5 leaderboard)
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class DeepSeekOCRModel(OCRModel):
    def __init__(self, model_id: str = "deepseek-ai/deepseek-ocr-2"):
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def run(self, image_path: Path) -> str:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=OCR_PROMPT, images=image, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0], skip_special_tokens=True)


class DeepSeekOCR(DeepSeekOCRModel):
    def __init__(self):
        super().__init__("deepseek-ai/deepseek-ocr")

class DeepSeekOCR2(DeepSeekOCRModel):
    def __init__(self):
        super().__init__("deepseek-ai/deepseek-ocr-2")
