"""
DocTR — Document Text Recognition (Mindee).
pip install "python-doctr[torch,viz]"

Two-stage pipeline: DBNet++ for detection, ViT-S for recognition.
Returns words in reading order with bounding boxes; we join to plain text.
"""

from pathlib import Path
from models.base import OCRModel


class DocTRModel(OCRModel):
    def __init__(
        self,
        det_arch: str  = "db_resnet50",
        reco_arch: str = "vitstr_small",
        pretrained: bool = True,
    ):
        from doctr.models import ocr_predictor
        self.predictor = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=pretrained,
        )

    def run(self, image_path: Path) -> str:
        from doctr.io import DocumentFile

        doc = DocumentFile.from_images([str(image_path)])
        result = self.predictor(doc)

        lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    lines.append(" ".join(w.value for w in line.words))
        return "\n".join(lines)
