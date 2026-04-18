"""
MinerU2-VLM / MinerU2.5 (OpenDataLab).
pip install magic-pdf[full]

MinerU is a pipeline tool — it extracts structured content from PDFs/images
and returns markdown. We strip markdown to plain text for NED comparison.
"""

from pathlib import Path
from models.base import OCRModel
import re


def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keeping plain text content."""
    text = re.sub(r"#{1,6}\s?", "", text)          # headings
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)  # code
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)     # images
    text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text) # links
    text = re.sub(r"[-\*]{3,}", "", text)            # hr
    text = re.sub(r"\|.*?\|", lambda m: m.group().replace("|", " "), text)  # tables
    return text.strip()


class MinerUModel(OCRModel):
    def __init__(self, version: str = "2.5"):
        self.version = version

    def run(self, image_path: Path) -> str:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            writer = FileBasedDataWriter(tmp)
            ds = PymuDocDataset(str(image_path))
            if ds.classify() == SupportedPdfParseMethod.OCR:
                ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(writer)
            else:
                ds.apply(doc_analyze, ocr=False).pipe_txt_mode(writer)

            md_files = list(Path(tmp).glob("*.md"))
            if not md_files:
                return ""
            return strip_markdown(md_files[0].read_text(encoding="utf-8"))


class MinerU25(MinerUModel):
    def __init__(self):
        super().__init__(version="2.5")

class MinerU2VLM(MinerUModel):
    def __init__(self):
        super().__init__(version="2-vlm")
