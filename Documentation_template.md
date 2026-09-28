# Amazon ML Challenge 2026 - Methodology Write-up

## 1. Team Information
- **Team Name:** Pegasus
- **Members:** Pranjal Pathak (Team Leader), Divyansh Tiwari

## 2. Approach / Methodology
### 2.1 Data Preprocessing & Normalization
We built a custom multi-script normalization pipeline to handle the noisy data across US, India, and France.
- **Indic Transliteration:** Used nyascii to convert Devanagari, Tamil, and Telugu texts into standard English.
- **Phonetic Normalization:** Fixed common phonetic variations in Indian names (like sh/s, ph/f, ch/k, z/s, v/w/b).
- **Brand Cleaning:** Cleaned up business names by removing common honorifics (M/s, shri, dr) and domain tails (.com).
- **French Legal Forms:** Handled specific French legal entities like EURL, SCI, SNC, GIE, SASU, SARL, SAS, etc.
- **OCR Fixes:** Fixed common leetspeak and OCR typos (e.g., changing 'de1hi' to 'delhi').
- **Address Decomposition:** Instead of treating addresses as plain text, we extracted specific unit keys (plot numbers, shop numbers, flat numbers, PIN/postal codes, and localities) for better comparison.

### 2.2 Candidate Generation (Blocking)
To avoid comparing every record with every other record, we used a high-recall inverted index multi-key blocking strategy.
- We created blocking keys using name tokens, numbers, house codes, acronyms, phonetic tokens, and extracted unit keys.
- This allowed us to quickly fetch a pool of candidate pairs from the sources without missing potential matches, keeping the total candidate size manageable.

### 2.3 Feature Engineering
For the generated candidates, we extracted 21 specific features to help the model decide:
- **String Distances:** Used Rapidfuzz for fast calculations of token sort ratio, token set ratio, and levenshtein distance.
- **TF-IDF Similarities:** Computed sparse cosine similarities using word and character n-grams.
- **Custom Features:** Added phonetic similarity scores, compact brand matching, and address component congruence (checking if the extracted PIN codes and plot numbers match exactly).

### 2.4 Matching Model
- **Model:** We trained a LightGBM classifier (LGBMClassifier).
- **Training Strategy:** The model was trained specifically on hard negatives (true blocking negatives) so it learns to differentiate closely looking but different businesses.
- **Thresholds:** We used country-specific matching thresholds because naming conventions in France differ slightly from India or the US.

## 3. Post-Processing & Constraints
After the model scored the candidates, we applied a competitive bipartite match assignment to follow the rules:
- Handled primary matches with competitive ranking.
- Prevented number conflicts when assigning multiple matches.
- Ensured that exactly one row per Source 1 entity is present in the final output, leaving unmatched entities as empty singletons.
- Executed the inference partitioned by country to keep the memory footprint well under 2 GB RAM.

## 4. Reproducibility Instructions
To run the pipeline end-to-end and generate the output files:
\\\ash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run the pipeline
python code/business_entity_resolution/src/run_pipeline.py
\\\

## 5. Model License & Parameter Count
- **License:** MIT (Standard LightGBM License)
- **Parameter Count:** Well under 8 Billion parameters (LightGBM is a tree-based model with a tiny memory footprint).
