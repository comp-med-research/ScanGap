# ScanGap

**Benchmarking the performance gap between digital-native and wild historical scanned documents in OCR evaluation.**

Existing document OCR benchmarks are dominated by digital-native PDFs. Even benchmarks that introduce scanning (e.g. Real5-OmniDocBench) do so by re-scanning clean, modern documents under controlled conditions. ScanGap quantifies how much larger the performance gap becomes when OCR systems face *genuinely* historical, archival documents with organic degradation.

## Datasets

| Dataset | Type | Pages | Source |
|---|---|---|---|
| OmniDocBench (digital) | Digital-native | ~950 | [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) |
| OmniDocBench (fuzzy_scan) | Mild scan | 28 | Same, filtered by `fuzzy_scan=true` |
| Real5-OmniDocBench (scan tier) | Controlled re-scan | 1,355 | [PaddlePaddle/Real5-OmniDocBench](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench) |
| FUNSD | Genuinely scanned noisy forms | 199 | [FUNSD](https://guillaumejaume.github.io/FUNSD/) |
| ScanGap wild scans | Wild historical | ~100 | Internet Archive, LOC, Court Listener |

## Models evaluated

- Tesseract 5
- PaddleOCR
- Surya
- GPT-4o
- Gemini 2.0 Flash
- Qwen2.5-VL

## Repo structure

```
ScanGap/
├── data/
│   └── wildscans/          # ~100 wild historical pages + ground_truth.json
├── evaluation/
│   ├── metrics.py          # NED, CER, WER
│   └── run_models.py       # run any model on any dataset
├── analysis/
│   └── gap_analysis.py     # build degradation spectrum table
└── results/                # model predictions and scores (gitignored if large)
```

## Quickstart

```bash
pip install editdistance pytesseract paddleocr pillow pandas

# Run a model on a dataset
python -m evaluation.run_models \
  --dataset funsd \
  --model tesseract \
  --input_dir data/funsd/images \
  --output_dir results

# Build the gap analysis table
python -m analysis.gap_analysis
```

## Citation

Coming soon.
