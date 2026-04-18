"""
PP-StructureV3 (PaddleOCR)
pip install paddlepaddle paddleocr

Layout-aware pipeline: detects regions (text, table, figure) then runs OCR
per region and reconstructs reading order.
"""

from pathlib import Path
from models.base import OCRModel


class PPStructureV3(OCRModel):
    def __init__(self, lang: str = "en"):
        from paddleocr import PPStructure
        self.engine = PPStructure(
            table=True,
            ocr=True,
            lang=lang,
            show_log=False,
            recovery=True,   # reconstruct reading order
        )

    def run(self, image_path: Path) -> str:
        from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
        result = self.engine(str(image_path))
        h, w = result[0]["img_idx"] if result else (0, 0)

        lines = []
        for region in sorted(result, key=lambda r: (r["bbox"][1], r["bbox"][0])):
            rtype = region.get("type", "").lower()
            if rtype == "table":
                lines.append(region.get("res", {}).get("html", ""))
            else:
                res = region.get("res", [])
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, dict):
                            lines.append(item.get("text", ""))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            lines.append(item[1][0] if item[1] else "")
        return "\n".join(lines)
