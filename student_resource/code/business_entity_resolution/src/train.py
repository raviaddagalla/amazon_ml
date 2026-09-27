"""
Model Training Script for LightGBM Business Entity Resolution Classifier
(Phase 5 upgrade: realistic full pool distractors, 34 features, teacher-forcing positives, scale_pos_weight)
"""

import os
import sys
import time
import gc
from collections import Counter
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import get_blocking_keys_from_tokens
from src.config import (
    SRC_DIR,
    MODEL_PATH,
    MODELS_DIR,
    PROBABILITY_THRESHOLD,
    TOP_CANDIDATES_PER_S1,
    MAX_BLOCKING_KEY_SIZE,
    TRAIN_GT,
    TRAIN_S1,
)
from src.features import FEATURE_NAMES, compute_pair_features
from src.postprocessing import resolve_mutual_exclusivity
from src.pool_cache import build_candidate_pool
from src.preprocessing import process_record_text


def train_matching_model(
    sample_size: int = 20000,
    val_split: float = 0.15,
    seed: int = 42,
    use_saved_pairs: bool = False,
) -> lgb.Booster:
    """Train LightGBM binary classifier on realistic candidate pairs.
    
    Phase 5A upgrade: Uses realistic blocking against the ENTIRE S2/S3 pool,
    so the model learns against the true distractor density it encounters at inference time.
    
    Args:
        sample_size: Number of S1 training entities (default 20,000 entities = ~500k pairs).
        val_split: Fraction to hold out for validation (by entity).
        seed: Random seed for reproducibility.
        use_saved_pairs: If True and dataset exists, load pre-extracted pairs from disk to retrain instantly.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    DATA_DIR = os.path.join(SRC_DIR, "..", "data")
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset_path = os.path.join(DATA_DIR, "train_val_pairs_34feats.npz")
    t_total = time.time()

    print("=" * 70, flush=True)
    print("PHASE 5A/5B: TRAINING WITH REALISTIC DISTRACTORS + 34 FEATURES", flush=True)
    print("=" * 70, flush=True)

    if use_saved_pairs and os.path.exists(dataset_path):
        print(f"\nLoading pre-extracted training pairs from {dataset_path}...", flush=True)
        t0 = time.time()
        npz_data = np.load(dataset_path, allow_pickle=True)
        X_train = npz_data['X_train']
        y_train = npz_data['y_train']
        X_val = npz_data['X_val']
        y_val = npz_data['y_val']
        all_meta_val = [tuple(m) for m in npz_data['meta_val']]
        val_s1_ids = npz_data['val_s1_ids']
        val_gt_mids = npz_data['val_gt_mids']
        val_gt_dict = {
            sid: set(mids) for sid, mids in zip(val_s1_ids, val_gt_mids)
        }
        print(f"  Loaded {len(X_train):,} train pairs and {len(X_val):,} val pairs in {time.time()-t0:.1f}s", flush=True)
    else:
        con = duckdb.connect()
        
        # 1. Load S1 records
        print("\nLoading Source 1 training records...", flush=True)
        t0 = time.time()
        s1_df = con.execute(f"""
            SELECT entity_id, business_name, business_address, country 
            FROM read_csv('{TRAIN_S1}', delim='\t', header=true, quote='', escape='')
        """).df()
        print(f"  Loaded {len(s1_df):,} S1 entities in {time.time()-t0:.1f}s", flush=True)
        
        # Random entity-level split
        rng = np.random.RandomState(seed)
        n_total = len(s1_df)
        perm = rng.permutation(n_total)
        n_val = int(n_total * val_split)
        
        n_train_sample = min(sample_size, n_total - n_val)
        n_val_sample = min(int(n_train_sample * 0.20), n_val)
        
        train_indices = perm[n_val : n_val + n_train_sample]
        val_indices = perm[:n_val_sample]
        
        s1_train = s1_df.iloc[train_indices].copy().reset_index(drop=True)
        s1_val = s1_df.iloc[val_indices].copy().reset_index(drop=True)
        del s1_df
        
        print(f"  Training S1 entities:   {len(s1_train):,}", flush=True)
        print(f"  Validation S1 entities: {len(s1_val):,}", flush=True)
        
        # Load ground truth
        print("\nLoading ground truth...", flush=True)
        t0 = time.time()
        gt_raw = con.execute(f"""
            SELECT source1_entity_id, matched_entity_ids 
            FROM read_csv('{TRAIN_GT}', delim='\t', header=true, quote='', escape='')
        """).df()
        
        gt_dict = {}
        s1_ids_gt = gt_raw['source1_entity_id'].values
        match_ids_gt = gt_raw['matched_entity_ids'].values
        for i in range(len(gt_raw)):
            s1_id = s1_ids_gt[i]
            mid = match_ids_gt[i]
            if isinstance(mid, str) and mid:
                gt_dict[s1_id] = set(mid.split(','))
            else:
                gt_dict[s1_id] = set()
        del gt_raw
        print(f"  GT loaded in {time.time()-t0:.1f}s", flush=True)
        
        # 2. Process Country by Country
        countries = sorted(set(s1_train['country'].unique()))
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val, all_meta_val = [], [], []
        
        for country in countries:
            print(f"\n{'='*50}", flush=True)
            print(f"Processing country: {country}", flush=True)
            print(f"{'='*50}", flush=True)
            
            s1_train_c = s1_train[s1_train['country'] == country].reset_index(drop=True)
            s1_val_c = s1_val[s1_val['country'] == country].reset_index(drop=True)
            print(f"  Train S1: {len(s1_train_c):,}, Val S1: {len(s1_val_c):,}", flush=True)
            
            # Build in-memory candidate pool
            (pool_eids, pool_n_norm, pool_a_norm, pool_nums,
             pool_first, inv_index) = build_candidate_pool(country)
            
            print(f"  Building index lookup for {len(pool_eids):,} pool entities...", flush=True)
            eid_to_idx = {eid: idx for idx, eid in enumerate(pool_eids)}
            
            def featurize_split(s1_subset, is_val=False):
                X, y, meta = [], [], []
                t_start = time.time()
                n_processed = 0
                
                for row in s1_subset.itertuples(index=False):
                    s1_id = row.entity_id
                    true_set = gt_dict.get(s1_id, set())
                    
                    s1_name_norm, s1_addr_norm, n_toks, a_toks, s1_first_word, nums = process_record_text(
                        row.business_name, row.business_address
                    )
                    s1_nums = set(nums)
                    s1_n_words = set(n_toks)
                    s1_a_words = set(a_toks)
                    
                    s1_keys = get_blocking_keys_from_tokens(n_toks, a_toks, nums)
                    cand_weights = Counter()
                    for k in s1_keys:
                        postings = inv_index.get(k)
                        if postings is None or len(postings) > MAX_BLOCKING_KEY_SIZE:
                            continue
                        w = (5.0 if (k.startswith("n1:") or k.startswith("nb:") or k.startswith("m1:"))
                             else (4.0 if k.startswith("nsort:")
                                   else (3.0 if k.startswith("pin:")
                                         else (2.0 if (k.startswith("na:") or k.startswith("n2:")) else 1.0))))
                        for pool_idx in postings:
                            cand_weights[pool_idx] += w
                    
                    top_cands = cand_weights.most_common(TOP_CANDIDATES_PER_S1)
                    selected_pool_indices = set()
                    
                    for pool_idx, blk_score in top_cands:
                        cand_eid = pool_eids[pool_idx]
                        selected_pool_indices.add(pool_idx)
                        feat = compute_pair_features(
                            s1_name_norm, s1_addr_norm, s1_nums, s1_n_words, s1_a_words, s1_first_word,
                            pool_n_norm[pool_idx], pool_a_norm[pool_idx],
                            set(pool_nums[pool_idx]),
                            set(pool_n_norm[pool_idx].split()),
                            set(pool_a_norm[pool_idx].split()),
                            pool_first[pool_idx],
                            blk_score,
                            country=country,
                        )
                        is_match = 1 if cand_eid in true_set else 0
                        X.append(feat)
                        y.append(is_match)
                        if is_val:
                            meta.append((s1_id, cand_eid))
                    
                    # For training only: Teacher forcing / positive injection
                    if not is_val and true_set:
                        for true_eid in true_set:
                            p_idx = eid_to_idx.get(true_eid)
                            if p_idx is not None and p_idx not in selected_pool_indices:
                                feat = compute_pair_features(
                                    s1_name_norm, s1_addr_norm, s1_nums, s1_n_words, s1_a_words, s1_first_word,
                                    pool_n_norm[p_idx], pool_a_norm[p_idx],
                                    set(pool_nums[p_idx]),
                                    set(pool_n_norm[p_idx].split()),
                                    set(pool_a_norm[p_idx].split()),
                                    pool_first[p_idx],
                                    blk_score=0.0,
                                    country=country,
                                )
                                X.append(feat)
                                y.append(1)
                    
                    n_processed += 1
                    if n_processed % 5000 == 0:
                        elapsed = time.time() - t_start
                        rate = n_processed / max(elapsed, 0.001)
                        print(f"    Featurized {n_processed:,}/{len(s1_subset):,} ({rate:.0f} q/s)", flush=True)
                
                return (
                    np.array(X, dtype=np.float32) if X else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
                    np.array(y, dtype=np.int32) if y else np.empty(0, dtype=np.int32),
                    meta
                )
            
            print("  Featurizing training split...", flush=True)
            t0 = time.time()
            X_t, y_t, _ = featurize_split(s1_train_c, is_val=False)
            print(f"    Train pairs: {len(X_t):,} (Pos: {y_t.sum():,}) in {time.time()-t0:.1f}s", flush=True)
            all_X_train.append(X_t)
            all_y_train.append(y_t)
            
            print("  Featurizing validation split...", flush=True)
            t0 = time.time()
            X_v, y_v, meta_v = featurize_split(s1_val_c, is_val=True)
            print(f"    Val pairs:   {len(X_v):,} (Pos: {y_v.sum():,}) in {time.time()-t0:.1f}s", flush=True)
            all_X_val.append(X_v)
            all_y_val.append(y_v)
            all_meta_val.extend(meta_v)
            
            del pool_eids, pool_n_norm, pool_a_norm, pool_nums, pool_first, inv_index, eid_to_idx
            gc.collect()
        
        # 3. Combine Training Data
        X_train = np.vstack(all_X_train)
        y_train = np.concatenate(all_y_train)
        X_val = np.vstack(all_X_val)
        y_val = np.concatenate(all_y_val)
        del all_X_train, all_y_train, all_X_val, all_y_val
        gc.collect()
        
        # Build val_gt_dict
        val_gt_dict = {}
        val_s1_ids_list = []
        val_gt_mids_list = []
        for row in s1_val.itertuples(index=False):
            mids = list(gt_dict.get(row.entity_id, set()))
            val_gt_dict[row.entity_id] = set(mids)
            val_s1_ids_list.append(row.entity_id)
            val_gt_mids_list.append(mids)
        
        # Save pre-extracted dataset for rapid experimentation
        print(f"\nSaving training pairs dataset to {dataset_path}...", flush=True)
        t0 = time.time()
        np.savez_compressed(
            dataset_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            meta_val=np.array(all_meta_val, dtype=object),
            val_s1_ids=np.array(val_s1_ids_list, dtype=object),
            val_gt_mids=np.array(val_gt_mids_list, dtype=object),
        )
        print(f"  Dataset saved in {time.time()-t0:.1f}s ({os.path.getsize(dataset_path)/(1024*1024):.1f} MB)", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("Combined Training Set", flush=True)
    print(f"{'='*70}", flush=True)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = min(n_neg / max(n_pos, 1), 20.0)
    print(f"  Train: {len(X_train):,} pairs ({n_pos:,} pos, {n_neg:,} neg, weight={scale_pos_weight:.1f}x)", flush=True)
    print(f"  Val:   {len(X_val):,} pairs ({(y_val==1).sum():,} pos, {(y_val==0).sum():,} neg)", flush=True)
    
    # 4. Train LightGBM
    print(f"\nTraining LightGBM model ({len(FEATURE_NAMES)} features)...", flush=True)
    t0 = time.time()
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': seed,
        'n_jobs': -1,
        'verbosity': -1,
    }
    
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]
    
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=600,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=callbacks,
    )
    
    print(f"  Training finished in {time.time()-t0:.1f}s (Best round: {bst.best_iteration})", flush=True)
    bst.save_model(MODEL_PATH)
    print(f"  Model saved to {MODEL_PATH}", flush=True)
    
    # 5. Fast Validation Threshold Sweep
    print(f"\n{'='*70}", flush=True)
    print("VALIDATION THRESHOLD SWEEP", flush=True)
    print(f"{'='*70}", flush=True)
    
    val_probs = bst.predict(X_val)
    scored_cands = [
        (all_meta_val[i][0], all_meta_val[i][1], float(val_probs[i]))
        for i in range(len(all_meta_val))
    ]
    
    best_f05 = 0.0
    best_thresh = PROBABILITY_THRESHOLD
    
    for t in np.arange(0.30, 0.96, 0.02):
        t = round(float(t), 2)
        matches = resolve_mutual_exclusivity(scored_cands, threshold=t)
        
        scores = []
        for s1_id, truth in val_gt_dict.items():
            preds = set(matches.get(s1_id, []))
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
                        scores.append((1.25 * precision * recall) / (0.25 * precision + recall))
        
        f05 = float(np.mean(scores))
        marker = ""
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            marker = " <-- BEST"
        print(f"  Threshold {t:.2f} -> Macro F0.5: {f05:.4f}{marker}", flush=True)
    
    print(f"\nBest threshold: {best_thresh:.2f} -> Macro F0.5: {best_f05:.4f}", flush=True)
    print(f"Total time: {(time.time()-t_total)/60:.1f} minutes", flush=True)
    
    return bst


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=20000,
                        help='Number of S1 training entities (default: 20,000)')
    parser.add_argument('--use-saved-pairs', action='store_true',
                        help='Load pre-extracted candidate pairs from disk')
    args = parser.parse_args()
    train_matching_model(sample_size=args.sample_size, use_saved_pairs=args.use_saved_pairs)
