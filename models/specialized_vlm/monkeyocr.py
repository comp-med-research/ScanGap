"""
MonkeyOCR (Monkey team, HuggingFace).
pip install transformers accelerate torch

Models:
  echo840/MonkeyOCR-pro-1.2B  — 1.9B params
  echo840/MonkeyOCR-3B        — 3.7B params
  echo840/MonkeyOCR-pro-3B    — 3.7B params, stronger version
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class MonkeyOCRModel(OCRModel):
    def __init__(self, model_id: str = "echo840/MonkeyOCR-pro-3B"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
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


class MonkeyOCRPro1B(MonkeyOCRModel):
    def __init__(self):
        super().__init__("echo840/MonkeyOCR-pro-1.2B")

class MonkeyOCR3B(MonkeyOCRModel):
    def __init__(self):
        super().__init__("echo840/MonkeyOCR-3B")

class MonkeyOCRPro3B(MonkeyOCRModel):
    def __init__(self):
        super().__init__("echo840/MonkeyOCR-pro-3B")
