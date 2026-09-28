# 🏢 Business Entity Resolution — Amazon ML Challenge 2026

> A high-precision entity matching system that links noisy, multi-source business records to their deduplicated reference entities across **US**, **India**, and **France**.

## Overview

This project tackles the core challenge of **entity resolution** at scale: given three independent data sources with messy, inconsistent business information and no shared identifiers, the system determines which records refer to the same real-world business.

### Results at a Glance

| Metric | Test Set | Validation Set |
|:---|:---:|:---:|
| Macro F₀.₅ | **0.9773** | **0.9811** |
| US F₀.₅ | **0.9834** | **0.9870** |
| India F₀.₅ | **0.9688** | **0.9722** |
| Blocking Recall | **99.62%** | **99.62%** |
| France (Open-Set) | **100%** (8/8) | **100%** |

All results validated on a **leak-free 3-way entity-level split** (17.5k Train / 2.5k Val / 5k Test).

---

## How It Works

The pipeline processes records through five major stages:

```
┌─────────────────────────────────────┐
│   Stage 1: Preprocessing            │
│   • Indic script transliteration     │
│   • Legal suffix normalization       │
│   • Phonetic representation          │
│   • Accent & case normalization      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   Stage 2: Candidate Generation      │
│   • Multi-key inverted index         │
│   • Token + phonetic + address keys  │
│   • ~99.62% recall coverage          │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   Stage 3: Feature Extraction        │
│   • 21 pairwise similarity signals   │
│   • TF-IDF cosines (word + char)     │
│   • Edit distances via RapidFuzz     │
│   • Phonetic & brand matching        │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   Stage 4: Classification            │
│   • LightGBM (300 trees, depth 8)    │
│   • Trained with hard negatives      │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   Stage 5: Match Assignment          │
│   • Bipartite competitive matching   │
│   • Country-calibrated thresholds    │
│   • Singleton detection              │
└─────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**Core dependencies:** numpy, pandas, scipy, scikit-learn, lightgbm, rapidfuzz, anyascii

### Training

Build the LightGBM classifier from scratch and save model artifacts:

```bash
python run_pipeline.py --train
```

### Validation

Run entity-level evaluation with threshold optimization and the French stress test:

```bash
python run_pipeline.py --validate
```

### Prediction

Generate submission files for all countries (streams in chunks, stays under 2.5 GB):

```bash
python run_pipeline.py --predict
```

You can also run country-specific subsets for quick testing:

```bash
python run_pipeline.py --predict --country France --limit 1000
python run_pipeline.py --predict --country India --limit 5000
```

### Output

| File | Description |
|:---|:---|
| `output/matching_results.tsv` | Final entity matches — `source1_id` → comma-separated matched IDs |
| `output/candidate_pairs.tsv` | All candidate pairs considered by the blocking stage |

---

## Technical Highlights

### Multilingual Preprocessing
- **Indic scripts**: Devanagari, Tamil, and Telugu records are transliterated to Latin via `anyascii`, enabling token overlap with English-only Source 1 records
- **Legal suffix handling**: Both standard English forms (Ltd, Corp, LLC) and transliterated Indic variants (`praivet`, `limirrd`, `elelpi`) are normalized
- **French support**: Accent stripping, legal forms (EURL, SCI, SASU, SARL), and French stopwords handled natively
- **Phonetic normalization**: Context-aware rules (e.g., soft-c: `c` before `e/i/y` → `s`, otherwise → `k`)

### High-Recall Blocking
Multiple complementary blocking keys combined via UNION to achieve **99.62% recall**:
- Significant name tokens + address numbers + house codes
- Phonetic skeleton tokens + unit keys + compact brand tokens
- Distinctive address tokens (handles rebranded entities and phonetic drift)

### Feature Engineering (21 signals)
Fast pairwise feature extraction (~0.6 µs per pair) using:
- RapidFuzz C++ string similarities (token sort ratio, Levenshtein ratio)
- Precomputed sparse TF-IDF cosines (word n-grams + character n-grams)
- Phonetic skeleton similarity, compact brand similarity, unit key matching
- Number/postcode agreement and missingness indicators

### Match Resolution
- 1-to-1 primary match locking with country-specific thresholds
- Quality-controlled secondary multi-matches (up to 12 per entity)
- Automatic singleton prediction when no candidate clears the threshold

---

## Key Design Decisions

1. **Indic Transliteration** — Tamil/Devanagari records in S2/S3 had zero token overlap with English S1 records. Adding `anyascii` transliteration jumped F₀.₅ from 0.830 to 0.951.

2. **Distinctive Address Blocking** — Standard name-based blockers missed rebranded entities and phonetic drift cases. Adding distinctive address tokens (with frequency cap) pushed India blocking recall from 92.3% to 99.26%.

3. **Guard Override Removal** — Forensic analysis showed 87.7% of false negatives had model confidence ≥ 0.88 but were suppressed by rigid string-similarity guards. Removing guards in favor of pure bipartite assignment improved test F₀.₅ from 0.960 to 0.977.

4. **Transliterated Legal Suffixes** — Indic transliterations of legal terms (e.g., `praivet` for "private") were diluting name tokens. Expanding the normalization regex improved India F₀.₅ by +0.70 pp.

5. **Threshold Calibration** — Expanding the search grid beyond the initial 0.98 cap revealed the true optimum of the LightGBM probability distribution, reaching validation F₀.₅ of 0.9811.

---

## Project Layout

```
├── run_pipeline.py              # Main entry point (train/validate/predict)
├── kaggle_pipeline.py           # Standalone predictor for Kaggle/Colab
├── business_entity_resolution.py # Core resolution module
├── validate_submission.py       # Output format checker
├── requirements.txt             # Pinned dependencies
├── artifacts/                   # Saved models & evaluation results
├── benchmarks/                  # Public benchmark evaluation scripts
├── tests/                       # Unit tests & validation suites
├── docs/                        # Problem statement & challenge guidelines
├── data/                        # Dataset directory
└── output/                      # Generated submission files
```

---

## Running on Cloud Platforms

**Google Colab**: Open `colab_runner.ipynb` → Runtime > High-RAM CPU → Run All (~10-12 min)

**Kaggle Notebook**: Upload `run_kaggle.ipynb` + `kaggle_pipeline.py` + `model_v3.pkl` → Run All (~15-20 min)

See [`KAGGLE_GUIDE.md`](./KAGGLE_GUIDE.md) for detailed Kaggle instructions.

---

## Team & Authorship

- **Team Leader:** Pranjal Pathak
- **Core Developer:** Divyansh Tiwari ([@divyanshcodesofficial](https://github.com/divyanshcodesofficial))
- **Competition:** Amazon ML Challenge 2026 — Multilingual Business Entity Resolution

