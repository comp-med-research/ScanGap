"""
Docling (IBM) — v2.88.0+
pip install docling

Supports PDF, images, DOCX, and more. Uses Granite-Docling-258M internally
for layout and table extraction.
"""

from pathlib import Path
from models.base import OCRModel


class DoclingModel(OCRModel):
    def __init__(self):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()

    def run(self, image_path: Path) -> str:
        result = self.converter.convert(str(image_path))
        # Export as plain text (strips markdown tables/headings to raw text)
        return result.document.export_to_text()
