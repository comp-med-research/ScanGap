"""
Qwen VL models (Alibaba).
pip install transformers accelerate torch torchvision

Models:
  Qwen/Qwen2.5-VL-7B-Instruct    — smaller, faster
  Qwen/Qwen2.5-VL-72B-Instruct   — used in Real5-OmniDocBench leaderboard
  Qwen/Qwen3-VL-235B-A22B        — largest, MoE architecture (235B total / 22B active)

Note: 72B and 235B models require multi-GPU or quantisation.
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class QwenVLModel(OCRModel):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def run(self, image_path: Path) -> str:
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True)


class QwenVL7B(QwenVLModel):
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-VL-7B-Instruct")

class QwenVL72B(QwenVLModel):
    def __init__(self):
        super().__init__("Qwen/Qwen2.5-VL-72B-Instruct")

class Qwen3VL235B(QwenVLModel):
    def __init__(self):
        # Qwen3-VL uses the same processor/model class
        super().__init__("Qwen/Qwen3-VL-235B-A22B")
