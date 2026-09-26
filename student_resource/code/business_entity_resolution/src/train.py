"""
Model Training Script for LightGBM Business Entity Resolution Classifier
"""

import os
import time
from collections import Counter, defaultdict
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.blocking import extract_blocking_keys
from src.config import (
    MODEL_PATH,
    MODELS_DIR,
    PROBABILITY_THRESHOLD,
    TOP_CANDIDATES_PER_S1,
    TRAIN_GT,
    TRAIN_S1,
    TRAIN_S2,
    TRAIN_S3,
)
from src.features import compute_pair_features
from src.postprocessing import resolve_mutual_exclusivity
from src.preprocessing import (
    extract_first_word,
    extract_numbers,
    extract_tokens,
    normalize_text,
)


def train_matching_model(sample_size: int = 30000) -> lgb.Booster:
  """Train LightGBM binary classifier on candidate pairs from training data."""
  os.makedirs(MODELS_DIR, exist_ok=True)
  con = duckdb.connect()

  print(
      f"Loading {sample_size:,} S1 training records from {TRAIN_S1}...",
      flush=True,
  )
  s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country 
        FROM read_csv('{TRAIN_S1}', delim='\t', header=true, quote='', escape='') 
        LIMIT {sample_size};
    """).df()

  # Split 80% train, 20% validation
  n_train = int(len(s1_df) * 0.8)
  s1_train = s1_df.iloc[:n_train].copy()
  s1_val = s1_df.iloc[n_train:].copy()

  print(f"Train S1: {len(s1_train):,}, Validation S1: {len(s1_val):,}", flush=True)

  gt_df = con.execute(f"""
        SELECT source1_entity_id, matched_entity_ids 
        FROM read_csv('{TRAIN_GT}', delim='\t', header=true, quote='', escape='') 
        WHERE source1_entity_id IN (SELECT entity_id FROM s1_df);
    """).df()

  gt_dict = {}
  all_target_ids = set()
  for _, r in gt_df.iterrows():
    if pd.notna(r["matched_entity_ids"]) and r["matched_entity_ids"]:
      m = set(r["matched_entity_ids"].split(","))
      gt_dict[r["source1_entity_id"]] = m
      all_target_ids.update(m)
    else:
      gt_dict[r["source1_entity_id"]] = set()

  # Load targets + distractors from S2 and S3
  print(f"Loading candidate pool for {len(all_target_ids):,} targets...", flush=True)
  con.register("targets", pd.DataFrame({"mid": list(all_target_ids)}))

  s2 = con.execute(f"""
        SELECT * FROM (
            SELECT s2.* FROM read_csv('{TRAIN_S2}', delim='\t', header=true, quote='', escape='') s2
            SEMI JOIN targets ON entity_id = mid
            UNION ALL
            SELECT * FROM read_csv('{TRAIN_S2}', delim='\t', header=true, quote='', escape='') LIMIT 80000
        )
    """).df().drop_duplicates(subset=["entity_id"])

  s3 = con.execute(f"""
        SELECT * FROM (
            SELECT s3.* FROM read_csv('{TRAIN_S3}', delim='\t', header=true, quote='', escape='') s3
            SEMI JOIN targets ON entity_id = mid
            UNION ALL
            SELECT * FROM read_csv('{TRAIN_S3}', delim='\t', header=true, quote='', escape='') LIMIT 80000
        )
    """).df().drop_duplicates(subset=["entity_id"])

  pool = pd.concat([s2, s3], ignore_index=True).drop_duplicates(
      subset=["entity_id"]
  )
  print(f"Candidate pool size: {len(pool):,} records.", flush=True)

  # Pre-normalize pool
  print("Pre-normalizing and indexing candidate pool...", flush=True)
  t0 = time.time()
  pool_data = {}
  inv_index = defaultdict(list)
  key_counts = Counter()

  for row in pool.itertuples(index=False):
    eid = row.entity_id
    n_norm = normalize_text(row.business_name)
    a_norm = normalize_text(row.business_address)
    nums = extract_numbers(row.business_address)
    n_words = set(extract_tokens(row.business_name))
    a_words = set(extract_tokens(row.business_address))
    first_word = extract_first_word(row.business_name)
    country = row.country

    pool_data[eid] = (n_norm, a_norm, nums, n_words, a_words, first_word)
    keys = extract_blocking_keys(row.business_name, row.business_address)
    for k in keys:
      inv_index[(country, k)].append(eid)
      key_counts[(country, k)] += 1

  print(f"Pool indexed in {time.time()-t0:.2f}s", flush=True)

  def featurize_split(s1_subset):
    X, y, meta = [], [], []
    for row in s1_subset.itertuples(index=False):
      s1_id = row.entity_id
      s1_name_norm = normalize_text(row.business_name)
      s1_addr_norm = normalize_text(row.business_address)
      s1_nums = extract_numbers(row.business_address)
      s1_n_words = set(extract_tokens(row.business_name))
      s1_a_words = set(extract_tokens(row.business_address))
      s1_first_word = extract_first_word(row.business_name)
      country = row.country

      true_set = gt_dict.get(s1_id, set())

      s1_keys = extract_blocking_keys(row.business_name, row.business_address)
      cand_weights = Counter()
      for k in s1_keys:
        ck = (country, k)
        df = key_counts.get(ck, 0)
        if df == 0 or df > 500:
          continue
        w = (
            5.0
            if k.startswith("n1:") or k.startswith("nb:")
            else (
                3.0
                if k.startswith("pin:")
                else (2.0 if k.startswith("na:") else 1.0)
            )
        )
        for cand_id in inv_index.get(ck, ()):
          cand_weights[cand_id] += w

      if not cand_weights:
        continue

      top_cands = cand_weights.most_common(TOP_CANDIDATES_PER_S1)
      for cand_id, blk_score in top_cands:
        c_n, c_a, c_nums, c_nw, c_aw, c_first = pool_data[cand_id]
        feat = compute_pair_features(
            s1_name_norm,
            s1_addr_norm,
            s1_nums,
            s1_n_words,
            s1_a_words,
            s1_first_word,
            c_n,
            c_a,
            c_nums,
            c_nw,
            c_aw,
            c_first,
            blk_score,
        )
        is_match = 1 if cand_id in true_set else 0
        X.append(feat)
        y.append(is_match)
        meta.append((s1_id, cand_id))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), meta

  print("Featurizing training split...", flush=True)
  X_train, y_train, _ = featurize_split(s1_train)
  print(
      f"Train set: {len(X_train):,} pairs (Positives: {y_train.sum():,})",
      flush=True,
  )

  print("Featurizing validation split...", flush=True)
  X_val, y_val, meta_val = featurize_split(s1_val)
  print(
      f"Val set: {len(X_val):,} pairs (Positives: {y_val.sum():,})", flush=True
  )

  print("Training LightGBM GBDT model...", flush=True)
  clf = lgb.LGBMClassifier(
      n_estimators=200,
      learning_rate=0.07,
      num_leaves=31,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      n_jobs=4,
      verbosity=-1,
  )
  clf.fit(X_train, y_train)

  clf.booster_.save_model(MODEL_PATH)
  print(f"Model saved to {MODEL_PATH}", flush=True)

  # Evaluate on validation split
  val_probs = clf.predict_proba(X_val)[:, 1]
  val_gt_dict = {
      row.entity_id: gt_dict[row.entity_id]
      for row in s1_val.itertuples(index=False)
  }

  scored_cands = [
      (meta_val[i][0], meta_val[i][1], val_probs[i])
      for i in range(len(meta_val))
  ]
  exclusive_matches = resolve_mutual_exclusivity(
      scored_cands, threshold=PROBABILITY_THRESHOLD
  )

  scores = []
  for s1_id, truth in val_gt_dict.items():
    preds = set(exclusive_matches.get(s1_id, []))
    if len(truth) == 0:
      scores.append(1.0 if len(preds) == 0 else 0.0)
    else:
      if len(preds) == 0:
        scores.append(0.0)
      else:
        tp = len(truth & preds)
        precision = tp / len(preds)
        recall = tp / len(truth)
        if precision + recall == 0:
          scores.append(0.0)
        else:
          scores.append(
              (1.25 * precision * recall) / (0.25 * precision + recall)
          )

  val_f05 = np.mean(scores)
  print(
      f"Validation Macro F_0.5 Score at threshold {PROBABILITY_THRESHOLD}:"
      f" {val_f05:.4f}",
      flush=True,
  )
  return clf.booster_


if __name__ == "__main__":
  train_matching_model()
