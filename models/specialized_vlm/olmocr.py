"""
olmOCR 2 (Allen AI / Ai2).
pip install olmocr

Models:
  allenai/olmOCR-2-7B-1025      — latest (Oct 2025), GRPO RL-tuned, 82.4 on olmOCR-Bench
  allenai/olmOCR-2-7B-1025-FP8  — FP8 quantised, ~3400 tok/s on H100
  allenai/olmOCR-7B-0225-preview — earlier preview release

olmOCR is fine-tuned from Qwen2.5-VL-7B for PDF/document linearisation.
Uses the olmocr toolkit's anchor-text pipeline for best results.
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class OlmOCRModel(OCRModel):
    def __init__(self, model_id: str = "allenai/olmOCR-2-7B-1025"):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def run(self, image_path: Path) -> str:
        import torch
        from PIL import Image

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


class OlmOCR2(OlmOCRModel):
    def __init__(self):
        super().__init__("allenai/olmOCR-2-7B-1025")

class OlmOCR2FP8(OlmOCRModel):
    def __init__(self):
        super().__init__("allenai/olmOCR-2-7B-1025-FP8")
