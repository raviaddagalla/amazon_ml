# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An enterprise-grade, memory-efficient, and highly scalable machine learning pipeline for cross-source Business Entity Resolution (ER) across 11.7+ million business entities.

---

## 1. Executive Summary

This repository contains the complete, self-contained, end-to-end reproducible pipeline for the Business Entity Resolution Challenge. 

### Key Technical Contributions:
1. **Dynamic Country Partitioning**: Strictly enforces intra-country isolation with zero cross-country false positives, dynamically supporting open sets of countries (including France, US, India).
2. **Universal Multi-Script Transliteration & Normalization**: Offline normalization using `anyascii` converting multilingual Indic scripts (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, etc.) and accented European characters into standardized phonetic ASCII representations, complemented by a bidirectional abbreviation dictionary (US, India, France) and corporate stopword pruning.
3. **Multi-Key Weighted Inverted Index Blocking**: Captures primary brand tokens, phonetic metaphone keys (`jellyfish`), sorted-token signatures for word-order invariance, PIN/postal codes, and address number-locality composites.
4. **LightGBM Matcher (34 Discriminative Features)**: Featurizes candidate pairs across 14 name features (Jaro-Winkler, Damerau-Levenshtein, LCS ratio, sorted-token ratio, token metaphones, initials match), 10 address features, 5 structured address component signals (street number match/mismatch, postal code match/mismatch), and number overlaps.
5. **Realistic Distractor Mining & Threshold Calibration**: Fixed the fundamental training bug by mining hard negative distractors directly from the real 10.3M production candidate pool. Calibrated threshold sweep achieves **0.6862 validation Macro $F_{0.5}$** at threshold 0.84 (+0.0724 over the 0.6138 baseline and +0.1082 over the 0.578 leaderboard score).
6. **Bipartite 1-to-1 Mutual Exclusivity Resolver**: Enforces that each candidate from Source 2 or Source 3 is assigned to at most one Source 1 reference entity based on maximum ML confidence, eliminating false merges and maximizing precision on singletons.
7. **Ultra-Low Memory Footprint (<2 GB RAM)**: Uses DuckDB columnar streaming and batch-oriented execution, easily running on standard machines without out-of-memory errors.

---

## 2. Directory Structure

```
business_entity_resolution/
├── data/
│   └── train_val_pairs_34feats.npz    # Pre-extracted 508k train & 95k val pairs (instant retraining)
├── models/
│   └── lgbm_entity_resolver.txt       # Trained 34-feature LightGBM model artifact
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Paths, abbreviations, threshold (0.84), hyperparameter configs
│   ├── preprocessing.py               # Normalization, anyascii transliteration, abbreviation expansion
│   ├── blocking.py                    # Multi-key inverted index blocking (phonetic, sorted, pin, brand)
│   ├── features.py                    # 34 RapidFuzz, phonetic metaphone, and numeric feature extractors
│   ├── train.py                       # Training pipeline with realistic distractor mining & threshold sweep
│   ├── inference.py                   # High-performance streaming country-partitioned inference
│   └── postprocessing.py              # Mutual exclusivity resolver and submission TSV exporters
├── run_pipeline.py                    # Master CLI script (reproduces outputs end-to-end)
├── README.md                          # Reproduction documentation (this file)
└── requirements.txt                   # Pinned dependency versions
```

---

## 3. Environment Setup & Installation

The solution is developed in pure Python 3.10+ without proprietary dependencies or external network lookups.

Install the required pinned dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce End-to-End

### Quick Run (Using Pre-trained Model)
To generate the final submission files (`output/matching_results.tsv` and `output/candidate_pairs.tsv`) and validate them against the test set:
```bash
python run_pipeline.py --infer --validate
```

### Instant Retraining (< 1 Minute)
Retrain the LightGBM model directly from the saved 34-feature training pairs archive and run validation:
```bash
python run_pipeline.py --train --use-saved-pairs
```

### Full Retraining from Scratch
To retrain the LightGBM model from raw TSV data with full distractor mining, run inference, and validate:
```bash
python run_pipeline.py --train --infer --validate
```

Or simply run:
```bash
python run_pipeline.py --all
```

---

## 5. Output Verification

The pipeline generates two files in the `output/` directory:
1. `output/matching_results.tsv`: Tab-separated final entity matches (`source1_entity_id\tmatched_entity_ids`).
2. `output/candidate_pairs.tsv`: Tab-separated candidate pairs from blocking (`source1_entity_id\tcandidate_entity_ids`).

To independently validate the generated output files with the official submission validator:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
A status of `PASS` confirms complete compliance with all challenge formatting and structural requirements.
