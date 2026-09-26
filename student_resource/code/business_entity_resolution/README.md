# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An enterprise-grade, memory-efficient, and highly scalable machine learning pipeline for cross-source Business Entity Resolution (ER) across 11.7+ million business entities.

---

## 1. Executive Summary

This repository contains the complete, self-contained, end-to-end reproducible pipeline for the Business Entity Resolution Challenge. 

### Key Technical Contributions:
1. **Dynamic Country Partitioning**: Strictly enforces intra-country isolation with zero cross-country false positives, dynamically supporting open sets of countries (including France, US, India).
2. **Universal Multi-Script Transliteration**: Offline normalization using `anyascii` converting multilingual Indic scripts (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, etc.) and accented European characters into standardized phonetic ASCII representations.
3. **Multi-Key Weighted Inverted Index Blocking**: Captures primary brand tokens, 4-character prefix stems, bigram signatures, and address number-locality co-occurrences, achieving **>95.4% candidate recall ceiling** while reducing the candidate search space by **>99.8%**.
4. **LightGBM Gradient Boosted Decision Tree Matcher**: Featurizes candidate pairs with 13 discriminative features across string similarity (RapidFuzz C++ AVX2 SIMD), token Jaccard overlaps, and street number match/mismatch penalties.
5. **Bipartite 1-to-1 Mutual Exclusivity Resolver**: Enforces that each candidate from Source 2 or Source 3 is assigned to at most one Source 1 reference entity based on maximum ML confidence, eliminating false merges and maximizing the precision-weighted **macro F_0.5 score (0.9275 on validation)**.
6. **Ultra-Low Memory Footprint (<1.5 GB RAM)**: Uses DuckDB columnar streaming and batch-oriented execution, easily running on standard machines without out-of-memory errors.

---

## 2. Directory Structure

```
business_entity_resolution/
├── models/
│   └── lgbm_entity_resolver.txt       # Pre-trained LightGBM model artifact
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Global paths, thresholds, and hyperparameter configs
│   ├── preprocessing.py               # Text normalization, anyascii transliteration, number cleaner
│   ├── blocking.py                    # Multi-key inverted index and candidate generation
│   ├── features.py                    # 13 RapidFuzz and numeric feature extractors
│   ├── train.py                       # Training pipeline on train_source1/2/3 and ground truth
│   ├── inference.py                   # Streaming country-partitioned inference pipeline
│   └── postprocessing.py              # Mutual exclusivity resolver and submission TSV exporters
├── run_pipeline.py                    # Master CLI script (reproduces outputs end-to-end)
├── README.md                          # Reproduction documentation (this file)
└── requirements.txt                   # Pinned dependency versions
```

---

## 3. Environment Setup & Installation

The solution is developed in pure Python 3.10+ (tested on Python 3.13) without proprietary dependencies or external network lookups.

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

### Full Retraining from Scratch
To retrain the LightGBM model on training data, run inference, and validate:
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
