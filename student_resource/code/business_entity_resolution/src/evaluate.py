"""
Trustworthy Validation Harness for Business Entity Resolution Pipeline

Phase 0 / Updated: Builds a realistic evaluation that mirrors production inference conditions.
- Uses full training source2/source3 as candidate pool (cached for fast iteration)
- Runs actual production blocking from blocking.py
- Computes features with real features.py code (34 features)
- Applies real post-processing/threshold logic
- Reports official macro F_0.5, including singletons

Usage:
    python -m src.evaluate                      # Full validation (15% hold-out)
    python -m src.evaluate --sample-frac 0.01   # Quick smoke-test on 1% of S1
    python -m src.evaluate --threshold 0.60     # Test specific threshold
    python -m src.evaluate --sweep              # Sweep thresholds 0.30-0.95
"""

import argparse
import gc
import os
import sys
import time
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from collections import Counter

from src.config import (
    MODEL_PATH,
    TOP_CANDIDATES_PER_S1,
    PROBABILITY_THRESHOLD,
    TRAIN_GT,
    TRAIN_S1,
    MAX_BLOCKING_KEY_SIZE,
)
from src.preprocessing import process_record_text
from src.blocking import get_blocking_keys_from_tokens
from src.features import FEATURE_NAMES, compute_pair_features
from src.postprocessing import resolve_mutual_exclusivity
from src.pool_cache import get_cached_pool


def compute_macro_f05(gt_dict: dict, pred_dict: dict) -> dict:
    """Compute macro-averaged F_0.5, including singletons."""
    scores = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_total = 0
    matched_correct = matched_total = 0
    
    for s1_id, truth in gt_dict.items():
        preds = set(pred_dict.get(s1_id, []))
        
        if len(truth) == 0:
            singleton_total += 1
            if len(preds) == 0:
                scores.append(1.0)
                singleton_correct += 1
            else:
                scores.append(0.0)
                fp_total += len(preds)
        else:
            matched_total += 1
            if len(preds) == 0:
                scores.append(0.0)
                fn_total += len(truth)
            else:
                tp = len(truth & preds)
                fp = len(preds - truth)
                fn = len(truth - preds)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                
                precision = tp / len(preds) if preds else 0
                recall = tp / len(truth) if truth else 0
                
                if precision + recall == 0:
                    scores.append(0.0)
                else:
                    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                    scores.append(f05)
                    if f05 > 0.5:
                        matched_correct += 1
    
    macro_f05 = float(np.mean(scores)) if scores else 0.0
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return {
        'macro_f05': macro_f05,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'n_entities': len(gt_dict),
        'n_singletons': singleton_total,
        'singleton_accuracy': singleton_correct / singleton_total if singleton_total > 0 else 0.0,
        'n_matched': matched_total,
        'matched_f05_gt_half': matched_correct,
    }


def run_evaluation(
    sample_frac: float = 1.0,
    threshold: float | None = None,
    sweep: bool = False,
    val_split: float = 0.15,
    seed: int = 42,
    top_k: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full evaluation pipeline."""
    if threshold is None:
        threshold = PROBABILITY_THRESHOLD
    if top_k is None:
        top_k = TOP_CANDIDATES_PER_S1
    
    t_total = time.time()
    con = duckdb.connect()
    
    if verbose:
        print("=" * 70, flush=True)
        print("TRUSTWORTHY VALIDATION HARNESS (34 Features)", flush=True)
        print("=" * 70, flush=True)
    
    # 1. Load S1 and Ground Truth
    if verbose:
        print("\nLoading training Source 1...", flush=True)
    t0 = time.time()
    s1_all = con.execute(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_S1}', delim='\t', header=true, quote='', escape='')
    """).df()
    if verbose:
        print(f"  Loaded {len(s1_all):,} S1 entities in {time.time()-t0:.1f}s", flush=True)
    
    if verbose:
        print("Loading ground truth...", flush=True)
    t0 = time.time()
    gt_raw = con.execute(f"""
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{TRAIN_GT}', delim='\t', header=true, quote='', escape='')
    """).df()
    
    gt_dict_all = {}
    s1_ids_gt = gt_raw['source1_entity_id'].values
    match_ids_gt = gt_raw['matched_entity_ids'].values
    for i in range(len(gt_raw)):
        s1_id = s1_ids_gt[i]
        mid = match_ids_gt[i]
        if isinstance(mid, str) and mid:
            gt_dict_all[s1_id] = set(mid.split(','))
        else:
            gt_dict_all[s1_id] = set()
    del gt_raw
    if verbose:
        print(f"  GT loaded in {time.time()-t0:.1f}s ({len(gt_dict_all):,} entries)", flush=True)
    
    # 2. Split S1 into validation
    rng = np.random.RandomState(seed)
    all_entity_ids = s1_all['entity_id'].values
    n_total = len(all_entity_ids)
    perm = rng.permutation(n_total)
    
    n_val = int(n_total * val_split)
    val_indices = set(perm[:n_val])
    
    if sample_frac < 1.0:
        n_sample = max(1, int(n_val * sample_frac))
        val_indices = set(list(val_indices)[:n_sample])
    
    val_mask = np.array([i in val_indices for i in range(n_total)])
    s1_val = s1_all[val_mask].copy().reset_index(drop=True)
    val_gt = {eid: gt_dict_all[eid] for eid in s1_val['entity_id'] if eid in gt_dict_all}
    
    n_val_entities = len(s1_val)
    n_val_singletons = sum(1 for v in val_gt.values() if len(v) == 0)
    n_val_matched = n_val_entities - n_val_singletons
    
    if verbose:
        print(f"\nValidation split: {n_val_entities:,} entities "
              f"({n_val_singletons:,} singletons, {n_val_matched:,} with matches)", flush=True)
    
    del s1_all
    gc.collect()
    
    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}. Train first.", flush=True)
        sys.exit(1)
    bst = lgb.Booster(model_file=MODEL_PATH)
    expected_features = bst.num_feature()
    actual_features = len(FEATURE_NAMES)
    if verbose:
        print(f"\nModel loaded: expects {expected_features} features, "
              f"FEATURE_NAMES has {actual_features}", flush=True)
    if expected_features != actual_features:
        print(f"WARNING: Feature count mismatch! Model expects {expected_features}, "
              f"code provides {actual_features}. Retrain model first.", flush=True)
    
    # 4. Process Country by Country using Cached Pools
    val_countries = s1_val['country'].unique().tolist()
    all_scored_candidates = []
    all_candidate_sets = {}
    blocking_recall_hits = 0
    blocking_recall_total = 0
    
    for country in val_countries:
        if verbose:
            print(f"\n{'='*50}", flush=True)
            print(f"Processing country: {country}", flush=True)
            print(f"{'='*50}", flush=True)
        
        s1_country = s1_val[s1_val['country'] == country].reset_index(drop=True)
        n_s1 = len(s1_country)
        if verbose:
            print(f"  Val S1 entities for {country}: {n_s1:,}", flush=True)
        
        # Load cached pool and index
        (pool_eids, pool_n_norm, pool_a_norm, pool_nums,
         pool_first, inv_index) = get_cached_pool(country)
        
        t0 = time.time()
        batch_features = []
        batch_meta = []
        BATCH_SIZE = 10000
        
        for row_idx, row in enumerate(s1_country.itertuples(index=False)):
            s1_id = row.entity_id
            true_set = val_gt.get(s1_id, set())
            
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
            
            if not cand_weights:
                all_candidate_sets[s1_id] = set()
                if true_set:
                    blocking_recall_total += len(true_set)
                continue
            
            top_cands = cand_weights.most_common(top_k)
            cand_eid_set = set()
            
            for pool_idx, blk_score in top_cands:
                cand_eid = pool_eids[pool_idx]
                cand_eid_set.add(cand_eid)
                
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
                batch_features.append(feat)
                batch_meta.append((s1_id, cand_eid))
            
            all_candidate_sets[s1_id] = cand_eid_set
            
            if true_set:
                blocking_recall_total += len(true_set)
                blocking_recall_hits += len(true_set & cand_eid_set)
            
            if len(batch_features) >= BATCH_SIZE * top_k:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = bst.predict(X_batch)
                for i, (s1_id_b, cand_eid_b) in enumerate(batch_meta):
                    all_scored_candidates.append((s1_id_b, cand_eid_b, float(probs[i])))
                batch_features = []
                batch_meta = []
            
            if verbose and (row_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (row_idx + 1) / max(elapsed, 0.001)
                print(f"    Progress: {row_idx+1:,}/{n_s1:,} ({rate:.0f} q/s)", flush=True)
        
        # Remaining batch
        if batch_features:
            X_batch = np.array(batch_features, dtype=np.float32)
            probs = bst.predict(X_batch)
            for i, (s1_id_b, cand_eid_b) in enumerate(batch_meta):
                all_scored_candidates.append((s1_id_b, cand_eid_b, float(probs[i])))
        
        if verbose:
            elapsed = time.time() - t0
            print(f"  Completed {country} queries in {elapsed:.1f}s ({n_s1/max(elapsed,0.001):.0f} q/s)", flush=True)
        
        del pool_eids, pool_n_norm, pool_a_norm, pool_nums, pool_first, inv_index
        gc.collect()
    
    # 5. Metrics & Threshold Sweep
    blocking_recall = blocking_recall_hits / blocking_recall_total if blocking_recall_total > 0 else 0.0
    
    if sweep:
        if verbose:
            print(f"\n{'='*70}", flush=True)
            print("THRESHOLD SWEEP", flush=True)
            print(f"{'='*70}", flush=True)
        
        best_f05 = 0.0
        best_thresh = threshold
        results_list = []
        
        for t in np.arange(0.30, 0.96, 0.01):
            t = round(float(t), 2)
            matches = resolve_mutual_exclusivity(all_scored_candidates, threshold=t)
            m = compute_macro_f05(val_gt, matches)
            results_list.append((t, m['macro_f05'], m['micro_precision'], m['micro_recall']))
            if m['macro_f05'] > best_f05:
                best_f05 = m['macro_f05']
                best_thresh = t
        
        if verbose:
            print(f"\n{'Threshold':<12} {'Macro F0.5':<12} {'Precision':<12} {'Recall':<12}", flush=True)
            print("-" * 48, flush=True)
            for t, f05, prec, rec in results_list:
                marker = " <-- BEST" if t == best_thresh else ""
                print(f"{t:<12.2f} {f05:<12.4f} {prec:<12.4f} {rec:<12.4f}{marker}", flush=True)
            print(f"\nBest threshold: {best_thresh:.2f} -> Macro F0.5: {best_f05:.4f}", flush=True)
        
        threshold = best_thresh
    
    final_matches = resolve_mutual_exclusivity(all_scored_candidates, threshold=threshold)
    metrics = compute_macro_f05(val_gt, final_matches)
    metrics['blocking_recall'] = blocking_recall
    metrics['threshold_used'] = threshold
    metrics['top_k_used'] = top_k
    metrics['val_size'] = n_val_entities
    metrics['sample_frac'] = sample_frac
    
    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("EVALUATION RESULTS", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Validation entities:    {metrics['n_entities']:,}", flush=True)
        print(f"  - Singletons:           {metrics['n_singletons']:,}", flush=True)
        print(f"  - With matches:         {metrics['n_matched']:,}", flush=True)
        print(f"", flush=True)
        print(f"  Blocking recall ceiling: {metrics['blocking_recall']:.4f} "
              f"({blocking_recall_hits:,}/{blocking_recall_total:,})", flush=True)
        print(f"  Top-K candidates/entity: {top_k}", flush=True)
        print(f"  Probability threshold:   {threshold:.2f}", flush=True)
        print(f"", flush=True)
        print(f"  Macro F_0.5:            {metrics['macro_f05']:.4f}", flush=True)
        print(f"  Micro precision:        {metrics['micro_precision']:.4f}", flush=True)
        print(f"  Micro recall:           {metrics['micro_recall']:.4f}", flush=True)
        print(f"  Singleton accuracy:     {metrics['singleton_accuracy']:.4f}", flush=True)
        print(f"  TP/FP/FN:               {metrics['tp']:,} / {metrics['fp']:,} / {metrics['fn']:,}", flush=True)
        print(f"", flush=True)
        print(f"  Total eval time:        {(time.time()-t_total)/60:.1f} minutes", flush=True)
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trustworthy Validation Harness")
    parser.add_argument('--sample-frac', type=float, default=1.0,
                        help='Fraction of validation entities to evaluate (0-1)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override probability threshold')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep thresholds and report best')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Override TOP_CANDIDATES_PER_S1')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    run_evaluation(
        sample_frac=args.sample_frac,
        threshold=args.threshold,
        sweep=args.sweep,
        val_split=args.val_split,
        seed=args.seed,
        top_k=args.top_k,
    )
