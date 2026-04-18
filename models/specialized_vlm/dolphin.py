"""
Dolphin / Dolphin-1.5 (ByteDance).
pip install transformers accelerate torch

Models:
  ByteDance/Dolphin      — 322M, original
  ByteDance/Dolphin-1.5  — 0.3B, improved version used in Real5 leaderboard
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class DolphinModel(OCRModel):
    def __init__(self, model_id: str = "ByteDance/Dolphin-1.5"):
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


class Dolphin(DolphinModel):
    def __init__(self):
        super().__init__("ByteDance/Dolphin")

class Dolphin15(DolphinModel):
    def __init__(self):
        super().__init__("ByteDance/Dolphin-1.5")
