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
  [Text Preprocessing & Transliteration]
    • Unicode NFKD & AnyAscii Phonetic Transliteration (Devanagari, Tamil, Telugu → Latin)
    • Legal suffix normalization (Corp/Corporation, Pvt/Private/Ltd, Inc, LLC)
    • Numerical extraction with zero-padding stripping ("0017560" → "17560")
                │
                ▼
  [Multi-Key Inverted Index Blocking Engine]
    • Composite keys: Name bigrams, salient tokens, composite name-number & address keys
    • C-array (array('i')) posting lists with frequency capping (postings > 500 pruned)
    • High-recall candidate generation: >95.4% ground truth recall ceiling in Top-25 candidates
                │
                ▼
  [LightGBM GBDT Pair Classifier]
    • 13 Dense features via RapidFuzz C++ SIMD:
        - Name token set, token sort, partial, and Levenshtein ratios
        - Word-level Jaccard overlaps and first-word exact match indicators
        - Address partial & token set ratios
        - House number / PIN code exact matches and mismatch penalties
    • Open-source under MIT license (<8B parameter constraint compliant)
                │
                ▼
  [Bipartite 1-to-1 Mutual Exclusivity Post-Processing]
    • Domain constraint enforcement: candidate records matched at most once across S1 entities
    • Competitive resolution: candidates assigned to highest ML probability argmax
    • Precision-heavy Macro F_0.5 threshold optimization at P >= 0.40
                │
                ▼
  Official Outputs: matching_results.tsv & candidate_pairs.tsv
```

---

## 3. Results & Evaluation

| Evaluation Metric | Baseline Blocking | GBDT (Raw P ≥ 0.50) | **Final Model + 1-to-1 Mutual Exclusivity (P ≥ 0.40)** |
|:---|:---:|:---:|:---:|
| **Validation Macro $F_{0.5}$** | 0.1415 | 0.8363 | **0.9275** |
| **Validation Macro Precision** | 0.0682 | 0.8351 | **0.9412** |
| **Validation Macro Recall** | **0.9542** | 0.8410 | **0.8765** |
| **Singletons Correctness** | N/A | 96.2% | **100.0%** (Zero false merges on true singletons) |

### Validation Status
- Passed official submission validator (`utils/validate_submission.py`) with:
  `PASS — no blocking issues found. Safe to submit.`
- Exact row count: **1,732,544 rows** (249,915 singletons, 1,482,629 non-empty matches).

---

## 4. Repository Structure

```
├── README.md                                 # Executive summary and reproduction guide
├── .gitignore                                # Excludes large datasets and archives (>100MB)
├── student_resource/
│   ├── Documentation_template.md             # Complete methodology write-up
│   ├── output/
│   │   └── matching_results.tsv              # Leaderboard prediction results (1.73M entities)
│   ├── utils/
│   │   └── validate_submission.py            # Official validator script
│   └── code/
│       └── business_entity_resolution/
│           ├── src/
│           │   ├── __init__.py
│           │   ├── config.py                 # Paths, thresholds, legal stopwords
│           │   ├── preprocessing.py          # Transliteration & string cleaners
│           │   ├── blocking.py               # Inverted index blocking engine
│           │   ├── features.py               # 13 RapidFuzz & numeric feature extractors
│           │   ├── train.py                  # LightGBM GBDT training pipeline
│           │   ├── postprocessing.py         # 1-to-1 mutual exclusivity resolver
│           │   └── inference.py              # Low-memory streaming inference engine
│           ├── models/
│           │   └── lgbm_entity_resolver.txt  # Pre-trained LightGBM model weights
│           ├── run_pipeline.py               # Master CLI execution runner
│           ├── README.md                     # Pipeline-specific documentation
│           └── requirements.txt              # Pinned dependencies
├── scratch/                                  # Development testbeds and benchmarks
└── *.pdf                                     # Challenge guidelines and problem statement
```

---

## 5. Getting Started & Reproduction

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / Linux / macOS

```bash
cd student_resource/code/business_entity_resolution
pip install -r requirements.txt
```

### Run End-to-End Pipeline
```bash
# Run full streaming inference and validate submission
python run_pipeline.py --infer --validate

# Or train model from scratch, infer, and validate:
python run_pipeline.py --all
```
