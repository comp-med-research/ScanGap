"""
InternVL3.5 / InternVL3 (Shanghai AI Lab).
pip install transformers accelerate torch torchvision timm einops

Models:
  OpenGVLab/InternVL3_5-8B   — latest series, best efficiency/accuracy tradeoff
  OpenGVLab/InternVL3-78B    — largest open-source InternVL, highest accuracy
  OpenGVLab/InternVL3-8B     — good baseline, fits on single A100
"""

from pathlib import Path
from models.base import OCRModel, OCR_PROMPT


class InternVLModel(OCRModel):
    def __init__(self, model_id: str = "OpenGVLab/InternVL3_5-8B"):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def run(self, image_path: Path) -> str:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)

        def build_transform(input_size=448):
            return T.Compose([
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        transform = build_transform()
        pixel_values = transform(Image.open(image_path)).unsqueeze(0).to(
            next(self.model.parameters()).device, dtype=torch.bfloat16
        )
        generation_config = {"max_new_tokens": 2048, "do_sample": False}
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            OCR_PROMPT,
            generation_config,
        )
        return response


class InternVL35_8B(InternVLModel):
    def __init__(self):
        super().__init__("OpenGVLab/InternVL3_5-8B")

class InternVL3_8B(InternVLModel):
    def __init__(self):
        super().__init__("OpenGVLab/InternVL3-8B")

class InternVL3_78B(InternVLModel):
    def __init__(self):
        super().__init__("OpenGVLab/InternVL3-78B")
