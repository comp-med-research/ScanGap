"""
GLM-OCR (Tsinghua / Zhipu AI).
pip install transformers accelerate torch

Model:
  THUDM/glm-ocr  — 0.9B, highest TEDS (92.28) in Real5 leaderboard
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class GLMOCRModel(OCRModel):
    def __init__(self, model_id: str = "zai-org/GLM-OCR"):
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
