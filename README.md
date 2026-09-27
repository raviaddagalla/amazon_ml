# Amazon ML Challenge 2026 — Business Entity Resolution

An enterprise-grade, memory-efficient, and highly scalable machine learning pipeline for cross-source Business Entity Resolution (ER) across 11.7+ million business entities.

Developed for the **Amazon ML Challenge 2026**.

---

## 1. Challenge Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. These fragments share no common identifiers. 

- **Source 1**: The deduplicated reference source (~1.73M entities in test).
- **Source 2 & Source 3**: Noisy fragments (~9.97M entities in test) that map to zero, one, or many Source 1 entities.
- **Evaluation Metric**: Macro $F_{0.5}$ score per Source 1 entity (precision weighted 2× over recall, $\beta=0.5$). Singletons score 1.0 if empty and 0.0 upon any false merge.
- **Key Constraints**: Strict open-set country handling (US, India, and unseen France), MIT/Apache 2.0 open-source models only (<8B parameters), and **STRICTLY NO external lookups or APIs**.

---

## 2. Solution Architecture & Key Innovations

```
Raw Multi-Source Records (S1, S2, S3)
                │
                ▼
  [Text Preprocessing, Abbreviation Expansion & Transliteration]
    • Unicode NFKD & AnyAscii Phonetic Transliteration (Indic scripts → Latin)
    • Bidirectional abbreviation expansion (US, India, France)
    • Corporate stopword pruning (pvt, ltd, corp, sarl, sas, etc.)
    • S3 ID suffix stripping (ID: \d+)
                │
                ▼
  [Multi-Strategy Inverted Index Blocking Engine]
    • 6 high-recall, low-collision keys:
        - Primary name tokens (n1:) & bigrams (nb:)
        - Phonetic metaphone keys (m1:) via jellyfish
        - Sorted-token signatures (nsort:) for word-order transposition invariance
        - Postal / PIN codes (pin:) and number-locality composites (na:)
    • C-array (array('i')) posting lists with frequency capping (postings > 500 pruned)
    • Top-30 candidates per S1 entity exported to candidate_pairs.tsv
                │
                ▼
  [LightGBM GBDT Pair Classifier (34 Features)]
    • 34 Dense features via RapidFuzz C++ SIMD and phonetic algorithms:
        - 14 Name similarity & phonetic features (Jaro-Winkler, Damerau-Levenshtein, LCS ratio, sorted ratio, initials, metaphones)
        - 10 Address distance, presence, and ratio features
        - 5 Structured address component signals (street number match/mismatch, postal code match/mismatch)
        - 3 Numeric overlap indicators + 2 metadata features
    • Trained on 508k pairs with realistic hard negatives mined from the full 10.3M production pool
                │
                ▼
  [Bipartite 1-to-1 Mutual Exclusivity Post-Processing]
    • Domain constraint enforcement: candidate records matched at most once across S1 entities
    • Competitive resolution: candidates assigned to highest ML probability argmax
    • Precision-heavy Macro F_0.5 calibrated decision threshold at P >= 0.84
                │
                ▼
  Official Outputs: matching_results.tsv & candidate_pairs.tsv
```

---

## 3. Results & Evaluation Benchmarks

### 3.1 Macro $F_{0.5}$ Progression across Engineering Milestones

| Pipeline Version | Candidate Features | Distractor Mining Pool | Calibrated Threshold | Validation Macro $F_{0.5}$ | Delta vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Old)** | 13 features | 80,000 synthetic | 0.50 (default) | **0.6138** | — |
| **Baseline (Threshold Shift)** | 13 features | 80,000 synthetic | 0.95 | **0.6689** | +0.0551 |
| **Phase 4 (34 Features)** | 34 features | 80,000 synthetic | 0.88 | **0.6720** | +0.0582 |
| **Phase 5A (34 Feats + Real Mining)** | 34 features | **10.3M Full Pool** | 0.84 | **0.6862** | **+0.0724** |
| **Phase 5B (CatBoost Solo)** | 34 features | **10.3M Full Pool** | 0.92 | **0.6819** | +0.0681 |
| **Phase 5B (LGBM 70% + CatBoost 30%)** | 34 features | **10.3M Full Pool** | 0.84 | **0.6864** | **+0.0726** |

### 3.2 Threshold Calibration Breakdown on Upgraded 34-Feature Model

| Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Notes |
|:---:|:---:|:---:|:---:|:---|
| 0.50 | 0.6712 | 0.8340 | 0.5821 | Old default threshold (+0.0574 over old model) |
| 0.60 | 0.6758 | 0.8465 | 0.5794 | |
| 0.70 | 0.6801 | 0.8612 | 0.5742 | |
| 0.80 | 0.6854 | 0.8785 | 0.5684 | |
| **0.84** | **0.6862** | **0.8835** | **0.5661** | **Optimal calibrated threshold** |
| 0.88 | 0.6860 | 0.8904 | 0.5601 | High precision plateau |
| 0.92 | 0.6845 | 0.8980 | 0.5520 | Recall drops slightly |

---

## 4. Repository Structure

```
├── README.md                                 # Executive summary and reproduction guide
├── .gitignore                                # Excludes large datasets and archives (>100MB)
├── student_resource/
│   ├── Documentation_template.md             # Complete methodology write-up
│   ├── output/
│   │   ├── matching_results.tsv              # Leaderboard prediction results (1.73M entities)
│   │   └── candidate_pairs.tsv               # Candidate pairs results
│   ├── utils/
│   │   └── validate_submission.py            # Official validator script
│   └── code/
│       └── business_entity_resolution/
│           ├── data/
│           │   └── train_val_pairs_34feats.npz  # Pre-extracted 508k pairs (instant retraining)
│           ├── docs/
│           │   ├── eda_findings.md           # Exploratory data analysis findings
│           │   └── evaluation_results.md     # Full benchmark tables and sweeps
│           ├── models/
│           │   └── lgbm_entity_resolver.txt  # Trained 34-feature LightGBM model artifact
│           ├── src/
│           │   ├── __init__.py
│           │   ├── config.py                 # Paths, thresholds, legal stopwords, abbreviations
│           │   ├── preprocessing.py          # Transliteration, normalizer, abbreviation expansion
│           │   ├── blocking.py               # 6-strategy multi-key inverted index blocking
│           │   ├── features.py               # 34 RapidFuzz, phonetic, and numeric feature extractors
│           │   ├── train.py                  # Training pipeline with realistic distractor mining
│           │   ├── postprocessing.py         # 1-to-1 mutual exclusivity resolver
│           │   └── inference.py              # Streaming country-partitioned inference engine
│           ├── run_pipeline.py               # Master CLI execution runner
│           ├── README.md                     # Pipeline-specific documentation
│           └── requirements.txt              # Pinned dependencies
├── scratch/                                  # Development testbeds and benchmarks
└── *.pdf                                     # Challenge guidelines and problem statement
```

---

## 5. Getting Started & Reproduction

### Prerequisites
- Python 3.10+
- Windows / Linux / macOS

```bash
cd student_resource/code/business_entity_resolution
pip install -r requirements.txt
```

### Run End-to-End Pipeline
```bash
# 1. Instant Retrain from pre-extracted pairs archive (< 1 minute):
python run_pipeline.py --train --use-saved-pairs

# 2. Run full streaming inference on test set and validate submission:
python run_pipeline.py --infer --validate

# 3. Or run complete pipeline from scratch:
python run_pipeline.py --all
```
