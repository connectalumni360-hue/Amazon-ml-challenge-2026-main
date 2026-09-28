# Kaggle Deployment Guide — Amazon ML Challenge 2026

Step-by-step instructions to run the full entity resolution pipeline on Kaggle. Processes all **1,732,544** test entities in approximately **15–20 minutes**.

---

## What You Need

| File | What it does |
|:---|:---|
| `kaggle_pipeline.py` | High-throughput parallel predictor optimized for 4 vCPUs |
| `run_kaggle.ipynb` | Ready-to-upload 3-cell Jupyter notebook |
| `model_v3.pkl` | Pre-trained LightGBM model + TF-IDF vectorizers (~6 MB) |
| `validate_submission.py` | Checks output format before submission |

---

## Setup Instructions

### 1. Create the Notebook

- Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
- Upload `run_kaggle.ipynb` via **File > Upload Notebook**
- In the settings panel (right side):
  - **Accelerator**: None (this gives you 4 CPUs + 30 GB RAM)
  - **Internet**: On (needed for `pip install rapidfuzz anyascii`)

### 2. Attach Your Data

Two things to attach:

1. **Pipeline files** — drag `kaggle_pipeline.py` and `model_v3.pkl` into the file explorer (or upload as a private dataset)
2. **Test data** — click **+ Add Input** and select your dataset containing `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`

> The pipeline auto-scans `/kaggle/input/**` recursively — it will find your files regardless of folder structure.

### 3. Execute

Hit **Run All**. Expected timings:

| Country | Entities | Time |
|:---|---:|---:|
| France | 259k | ~2.5 min |
| India | 603k | ~4.5 min |
| US | 870k | ~8.0 min |
| **Total** | **1.73M** | **~15–18 min** |

---

## Getting Your Results

After execution completes, find `submission_archive.zip` under **Data > Output > /kaggle/working/output** in the sidebar.

This archive contains both `matching_results.tsv` and `candidate_pairs.tsv`, pre-validated and ready for upload to the challenge portal.
