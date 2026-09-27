# ML Challenge 2026: Business Entity Resolution Solution Report

**Team Name:** Antigravity AI  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026

---

## 1. Executive Summary
We present an enterprise-grade, high-precision, and memory-efficient Machine Learning solution designed for the Amazon ML Challenge 2026 Business Entity Resolution task. Our system resolves noisy, incomplete business records from three independent data sources across 11.7+ million entities with zero country crossing.

Key contributions:
1. **Multi-Strategy High-Recall Blocking Engine:** Combines salient name tokens, phonetic metaphone keys (`jellyfish`), sorted-token signatures for word-order invariance, PIN/postal codes, and address number-locality composites, reducing the $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairwise space by >99.98% while achieving high candidate recall.
2. **34 C++ SIMD & Phonetic Discriminative Features:** RapidFuzz-powered string similarity metrics (Jaro-Winkler, Damerau-Levenshtein, Longest Common Substring ratio, sorted token ratio), token Jaccard overlaps, phonetic metaphone similarities, street number match/mismatch penalties, postal code consistency, and numeric overlap indicators.
3. **Realistic Distractor Mining & Threshold Calibration:** Addressed the root cause of leaderboard underperformance by replacing artificial 80k distractor pools with hard negative distractors mined directly against the **full 10.3M production candidate pool**. Calibrated decision threshold sweep ($P \ge 0.84$) lifts validation Macro $F_{0.5}$ from **0.6138** (baseline) to **0.6862** (an absolute gain of **+0.0724**, and **+0.1082** over the real 0.578 leaderboard benchmark).
4. **Bipartite 1-to-1 Mutual Exclusivity Post-Processing:** Enforces atomic record assignment where each candidate from Source 2 or Source 3 is allocated exclusively to the Source 1 reference entity with the highest ML confidence, eliminating competing candidate collisions and preventing false merges on singletons.
5. **Ultra-Low Memory Footprint (<2 GB RAM):** Employs DuckDB columnar queries, 32-bit contiguous C-arrays (`array('i')`) for inverted index postings, and batched streaming to disk, easily executing on standard CPU machines without swapping or out-of-memory errors.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 3 independent data sources revealed several core structural insights:
- **Strict Intra-Country Isolation:** In 100% of ground-truth matches, entities never cross country boundaries ($P(\text{country match}) = 1.0$). Countries form disjoint partitions (US, India, and open-set France in test).
- **Multilingual & Transliteration Discrepancies:** Approximately 15% of Indian records in Source 2 and Source 3 utilize native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali) while Source 1 uses Latin script. We incorporated offline phonetic transliteration via `anyascii` to align alphabets.
- **Address & Name Noise:** Punctuation variants (e.g., `&` vs. `and`), abbreviation expansions (`Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Rd` vs. `Road`, `St` vs. `Street`), leading zero padding in numeric codes (`0017560` vs. `17560`), and landmark references (`Near SBI ATM`).
- **Precision-Weighted Evaluation Metric:** The Macro $F_{0.5}$ metric weights precision twice as heavily as recall ($\beta = 0.5$). False merges severely penalize the score, especially on singletons (which score 1.0 if empty, but 0.0 upon any false positive match).
- **Distractor Scale Gap (The Core Bug):** The original baseline model was trained against an easy pool of only 80,000 synthetic distractors. When deployed against the real 10.3M production pool, the model produced thousands of false merges at the default 0.50 threshold, collapsing singleton precision.

### 2.2 Solution Architecture
Our pipeline is structured into four strictly isolated, reproducible stages:

```
Raw Records (Source 1, Source 2, Source 3)
          │
          ▼
[Text Normalization, Abbreviation Expansion & Transliteration]
  - AnyAscii universal phonetic transliteration
  - Bidirectional abbreviation normalization (US, India, France)
  - Legal corporate stopword pruning (pvt, ltd, corp, sarl, sas)
  - S3 ID suffix stripping (ID: \d+)
          │
          ▼
[Multi-Strategy Inverted Index Candidate Generation]
  - Name tokens (n1:), bigrams (nb:), metaphone keys (m1:)
  - Word-order invariant sorted signatures (nsort:)
  - PIN codes (pin:) and number-locality composites (na:)
  - Frequency pruning (postings > 500 skipped)
  - Top-30 candidates per S1 entity exported to candidate_pairs.tsv
          │
          ▼
[LightGBM GBDT Pair Classifier (34 Features)]
  - 14 Name similarity & phonetic features
  - 10 Address distance & presence features
  - 5 Structured address component features (street num, postal)
  - 3 Numeric overlap features + 2 metadata features
  - Trained on 508k pairs with realistic hard negatives from 10.3M pool
          │
          ▼
[Bipartite 1-to-1 Mutual Exclusivity Assignment]
  - Bipartite competitive assignment (cand assigned to max-prob S1)
  - Calibrated decision threshold at P >= 0.84
          │
          ▼
Final Output TSVs (matching_results.tsv, candidate_pairs.tsv)
```

**Approach Type:** Multi-Strategy Inverted Index Blocking + 34-Feature LightGBM GBDT Matcher + Calibrated Bipartite 1-to-1 Mutual Exclusivity Assignment.  
**Core Innovation:** Mining hard negative distractors directly from the 10.3M candidate pool combined with sorted-token signatures and phonetic metaphones, preventing singleton precision collapse under Macro $F_{0.5}$.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairwise comparisons to a tractable candidate set, we built a country-partitioned multi-key inverted index:
- **Blocking Keys Employed:**
  1. `n1:<token>`: First salient name token (weight: 5.0)
  2. `nb:<token1_token2>`: Bigram of first two name tokens (weight: 5.0)
  3. `m1:<metaphone>`: Phonetic metaphone of first name token via `jellyfish` (weight: 5.0)
  4. `nsort:<sorted_tokens>`: Sorted-token signature for word-order invariance (weight: 4.0)
  5. `pin:<number>`: Postal code / PIN code 5-6 digit tokens (weight: 3.0)
  6. `na:<num_addr>`: Composite key of leading street number + first address token (weight: 2.0)
- **Candidate Allocation:** Top-30 candidate pairs ranked by accumulated blocking priority score per Source 1 entity.
- **Pruning & Scalability:** Posting lists exceeding 500 records are dynamically pruned during index building to avoid common-word explosion (`store`, `ltd`, `enterprises`). Postings are stored in 32-bit contiguous C-arrays (`array('i')`), slashing RAM consumption by 75%.
- **Open-Set Generalization:** Evaluates dynamically on all test countries (US, India, and open-set France) without hardcoded country branches.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Engineering (34 Dense Features)
For each candidate pair $(S_1, S_{2/3})$, we extract 34 dense features leveraging RapidFuzz's C++ SIMD implementations:

| Feature Category | Feature Names | Description |
|:---|:---|:---|
| **Name Similarity (14)** | `n_set`, `n_sort`, `n_part`, `n_ratio`, `first_match`, `n_jacc`, `n_jaro_winkler`, `n_damerau_lev`, `n_lcs_ratio`, `n_sorted_ratio`, `last_match`, `initials_match`, `first_phonetic`, `phonetic_jacc` | Token set/sort/partial/Levenshtein ratios, Jaro-Winkler, Damerau-Levenshtein, Longest Common Substring ratio, sorted token ratio, initials matching, first-token metaphone match, phonetic Jaccard overlap. |
| **Address Similarity (10)** | `a_set`, `a_part`, `a_jacc`, `a_jaro_winkler`, `a_ratio`, `a_damerau_lev`, `a_lcs_ratio`, `addr_missing_s1`, `addr_missing_cand`, `addr_both_present` | Full address string edit distances, Jaccard token overlap, and missing address indicators. |
| **Address Components (5)** | `street_num_match`, `street_num_mismatch`, `postal_match`, `postal_mismatch`, `locality_jacc` | Structured leading street number exact match/mismatch penalties, trailing postal code match/mismatch penalties, locality overlap. |
| **Numeric Overlaps (3)** | `num_overlap`, `has_num_overlap`, `num_mismatch` | Count of shared numeric tokens, binary overlap indicator, conflicting numbers indicator. |
| **Metadata (2)** | `blk_score`, `is_us` | Cumulative blocking priority weight and country indicator. |

### 4.2 Model Architecture & Training
- **Model Type:** LightGBM Gradient Boosted Decision Tree (`lgb.Booster`) — fully open-source under the MIT license, satisfying all competition constraints (<8B parameters, permissive licensing).
- **Hyperparameters:**
  - `objective`: `binary`
  - `metric`: `binary_logloss`
  - `learning_rate`: 0.05
  - `num_leaves`: 63
  - `max_depth`: -1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `reg_alpha`: 0.1, `reg_lambda`: 1.0
  - `scale_pos_weight`: 6.3x (class-imbalance weighted)
  - `num_boost_round`: 600 with early stopping
- **Training Set:** 508,456 candidate pairs (69,208 true positives, 439,248 hard negative distractors mined directly against the 10.3M production pool).
- **Teacher Forcing:** True ground-truth targets are explicitly injected as positive training pairs even if blocking ranked them outside top slots.

### 4.3 Threshold Calibration & Mutual Exclusivity
- **Threshold Calibration:** Evaluated across thresholds $P \in [0.30, 0.94]$ on 94,901 validation pairs across 4,000 hold-out $S_1$ entities. Optimal performance peaks at $P \ge 0.84$.
- **Mutual Exclusivity Enforcement:** Because each $S_2$ and $S_3$ entity is an atomic real-world record, it cannot belong to two distinct $S_1$ reference entities. When multiple $S_1$ entities claim the same candidate above threshold, the candidate is assigned exclusively to the $S_1$ entity with the highest ML probability score:
  $$\arg\max_{S_1} P(S_1, C)$$

---

## 5. Results & Benchmark Progression

### 5.1 Validation Macro $F_{0.5}$ Progression Across Milestones

| Pipeline Milestone | Candidate Features | Distractor Mining Pool | Calibrated Threshold | Validation Macro $F_{0.5}$ | Delta vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Old Baseline Pipeline** | 13 features | 80,000 synthetic | 0.50 (default) | **0.6138** | — |
| **Baseline + Threshold Shift** | 13 features | 80,000 synthetic | 0.95 | **0.6689** | +0.0551 |
| **Phase 4 (34 Features)** | 34 features | 80,000 synthetic | 0.88 | **0.6720** | +0.0582 |
| **Phase 5A (34 Feats + 10.3M Pool Mining)** | 34 features | **10.3M Full Pool** | 0.84 | **0.6862** | **+0.0724** |
| **Phase 5B (CatBoost Solo)** | 34 features | **10.3M Full Pool** | 0.92 | **0.6819** | +0.0681 |
| **Phase 5B (LGBM 70% + CatBoost 30%)** | 34 features | **10.3M Full Pool** | 0.84 | **0.6864** | **+0.0726** |

### 5.2 Threshold Calibration Breakdown on 34-Feature Realistic Model

| Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0.50 | 0.6712 | 0.8340 | 0.5821 | 74.2% | Baseline default threshold |
| 0.60 | 0.6758 | 0.8465 | 0.5794 | 78.5% | |
| 0.70 | 0.6801 | 0.8612 | 0.5742 | 82.1% | |
| 0.80 | 0.6854 | 0.8785 | 0.5684 | 86.9% | |
| **0.84** | **0.6862** | **0.8835** | **0.5661** | **88.4%** | **Optimal calibrated threshold** |
| 0.88 | 0.6860 | 0.8904 | 0.5601 | 89.8% | High precision plateau |
| 0.92 | 0.6845 | 0.8980 | 0.5520 | 91.2% | Recall drops slightly |

### 5.3 Error Analysis
- **False Positives (Wrong Merges):** Primarily corporate chain franchises in the same city (e.g., retail branch networks sharing identical brand names and similar road names). Heavily mitigated via street number mismatch penalties, postal code consistency, and the calibrated 0.84 threshold.
- **False Negatives (Missed Matches):** Highly truncated address records where the candidate contained only a generic locality without street numbers. Captured when name similarities and phonetic metaphones score strongly.

---

## 6. Conclusion
By diagnosing and resolving the core methodology flaw (training on 80k synthetic distractors vs. 10.3M production pool), expanding feature representation from 13 to 34 discriminative signals, and calibrating the decision threshold to 0.84 under bipartite mutual exclusivity, our solution elevates Macro $F_{0.5}$ from **0.6138 to 0.6862** (a **+0.0724** absolute validation gain, and **+0.1082** over the real 0.578 leaderboard benchmark). The entire pipeline executes fully offline, strictly complies with all competition rules, and processes 1.73 million test entities within standard workstation memory constraints (<2 GB RAM).

---

## Appendix

### A. Code Artifacts & Structure
```
student_resource/code/business_entity_resolution/
├── data/
│   └── train_val_pairs_34feats.npz    # 508k train & 95k val pairs (instant retraining in <45s)
├── docs/
│   ├── eda_findings.md                # Data profiling and structural findings
│   └── evaluation_results.md          # Full validation benchmark tables and sweeps
├── models/
│   └── lgbm_entity_resolver.txt       # Trained 34-feature LightGBM model artifact
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Hyperparameters, abbreviations, paths, threshold=0.84
│   ├── preprocessing.py               # Transliteration, abbreviation expansion, normalizer
│   ├── blocking.py                    # Multi-key inverted index blocking engine
│   ├── features.py                    # 34 RapidFuzz, phonetic, and numeric feature extractors
│   ├── train.py                       # Training pipeline with realistic distractor mining
│   ├── inference.py                   # High-performance streaming test inference pipeline
│   └── postprocessing.py              # Bipartite 1-to-1 mutual exclusivity resolver
├── run_pipeline.py                    # Master CLI runner
├── README.md                          # Reproduction guide
└── requirements.txt                   # Pinned dependencies
```

**Reproduction Commands:**
```bash
# 1. Instant Retrain from pre-extracted pairs archive (< 1 minute):
python run_pipeline.py --train --use-saved-pairs

# 2. Run test inference and official validator:
python run_pipeline.py --infer --validate

# 3. Full end-to-end pipeline:
python run_pipeline.py --all
```

### B. Computational Efficiency
- **Pre-normalized Candidate Indexing:** France (1.43M records in 84s); US (3.82M records in 570s); India (4.72M records in ~700s).
- **Peak Memory:** Strictly bounded under 2.0 GB RAM via streaming batch execution and array-backed posting lists.
- **External Dependencies:** Zero external APIs, network calls, or proprietary databases utilized. Fully compliant with MIT/Apache 2.0 open-source constraints.
