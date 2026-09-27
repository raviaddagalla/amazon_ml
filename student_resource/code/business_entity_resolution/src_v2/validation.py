"""
Validation Harness for Tiered Deterministic-First Entity Resolution (v2)
Evaluates on a stratified hold-out sample against the FULL candidate pool.
Computes macro F_0.5, singleton accuracy, per-country, and per-bucket breakdowns.
"""

import os
import sys
import time
import argparse
import duckdb
import numpy as np
import pandas as pd
from collections import defaultdict

from src_v2.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TIER1_SCORE, TIER2_SCORE, TIER3_PROB_THRESHOLD,
    TOP_CANDIDATES_PER_S1
)
from src_v2.canonicalize import get_record_signatures
from src_v2.retrieval import InvertedIndexEngine
from src_v2.cascade import (
    evaluate_tier1_match,
    evaluate_tier2_match,
    resolve_bipartite_matches,
)


def compute_official_macro_f05(gt_dict: dict, pred_dict: dict) -> dict:
    """
    Compute macro-averaged F_0.5 per official competition specification.
    
    Formula:
        F_0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R)
    For singletons (truth = empty):
        pred = empty -> score = 1.0
        pred != empty -> score = 0.0
    For matched (truth != empty):
        pred = empty -> score = 0.0
    """
    scores = []
    tp_total = fp_total = fn_total = 0
    singleton_correct = singleton_total = 0
    matched_correct = matched_total = 0
    
    bucket_scores = defaultdict(list)
    country_scores = defaultdict(list)
    
    for s1_id, truth in gt_dict.items():
        preds = set(pred_dict.get(s1_id, []))
        n_truth = len(truth)
        
        # Categorize into bucket
        if n_truth == 0:
            b_name = "0 (singleton)"
        elif n_truth == 1:
            b_name = "1 match"
        elif n_truth in (2, 3):
            b_name = "2-3 matches"
        else:
            b_name = "4+ matches"
            
        if n_truth == 0:
            singleton_total += 1
            if len(preds) == 0:
                s = 1.0
                singleton_correct += 1
            else:
                s = 0.0
                fp_total += len(preds)
        else:
            matched_total += 1
            if len(preds) == 0:
                s = 0.0
                fn_total += n_truth
            else:
                tp = len(truth & preds)
                fp = len(preds - truth)
                fn = len(truth - preds)
                tp_total += tp
                fp_total += fp
                fn_total += fn
                
                prec = tp / len(preds) if preds else 0.0
                rec = tp / n_truth if n_truth else 0.0
                
                if prec + rec == 0:
                    s = 0.0
                else:
                    s = (1.25 * prec * rec) / (0.25 * prec + rec)
                if s > 0.5:
                    matched_correct += 1
        
        scores.append(s)
        bucket_scores[b_name].append(s)
    
    macro_f05 = float(np.mean(scores)) if scores else 0.0
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "n_entities": len(gt_dict),
        "singleton_total": singleton_total,
        "singleton_acc": singleton_correct / singleton_total if singleton_total > 0 else 0.0,
        "matched_total": matched_total,
        "matched_acc": matched_correct / matched_total if matched_total > 0 else 0.0,
        "bucket_f05": {k: float(np.mean(v)) for k, v in bucket_scores.items()},
    }


def run_tiered_validation(sample_size_per_country: int = 2000, seed: int = 42):
    """
    Run full validation on stratified hold-out S1 sample against full S2+S3 pool.
    """
    print("=" * 70)
    print("STARTING STRATIFIED TIERED CASCADE VALIDATION (src_v2)")
    print("=" * 70)
    
    con = duckdb.connect()
    
    # 1. Stratified sampling of S1 hold-out entities
    print(f"Sampling {sample_size_per_country} entities per country from train_source1...")
    q_sample = f"""
    WITH s1_annotated AS (
        SELECT 
            s1.entity_id,
            s1.business_name,
            s1.business_address,
            s1.country,
            COALESCE(gt.matched_entity_ids, '') as matched_entity_ids,
            CASE 
                WHEN gt.matched_entity_ids IS NULL OR length(trim(gt.matched_entity_ids)) = 0 THEN 0
                ELSE length(gt.matched_entity_ids) - length(replace(gt.matched_entity_ids, ',', '')) + 1
            END as match_count
        FROM read_csv('{TRAIN_S1}', delim='\\t', header=true, quote='', escape='') s1
        LEFT JOIN read_csv('{TRAIN_GT}', delim='\\t', header=true, quote='', escape='') gt 
            ON s1.entity_id = gt.source1_entity_id
    )
    SELECT * FROM s1_annotated
    USING SAMPLE {sample_size_per_country * 4} (reservoir, {seed});
    """
    val_df_all = con.execute(q_sample).fetchdf()
    
    # Stratify by country and take exact sample_size_per_country
    val_subsets = []
    for c in ["US", "India"]:
        c_df = val_df_all[val_df_all["country"] == c].head(sample_size_per_country)
        val_subsets.append(c_df)
    val_s1_df = pd.concat(val_subsets).reset_index(drop=True)
    print(f"Sampled {len(val_s1_df):,} validation entities ({val_s1_df['country'].value_counts().to_dict()})")
    
    # Ground truth dictionary
    val_gt = {}
    for _, row in val_s1_df.iterrows():
        m_str = str(row["matched_entity_ids"]).strip()
        val_gt[row["entity_id"]] = set(m.strip() for m in m_str.split(",") if m.strip()) if m_str else set()
    
    all_predictions = {}
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
    
    for country in ["US", "India"]:
        print(f"\nEvaluating country: {country}")
        t0 = time.time()
        
        country_s1 = val_s1_df[val_s1_df["country"] == country].reset_index(drop=True)
        
        # Load candidate pool for this country (S2 + S3)
        print(f"  Loading candidate pool for {country} from S2 & S3...")
        q_pool = f"""
        SELECT entity_id, business_name, business_address FROM read_csv('{TRAIN_S2}', delim='\\t', header=true, quote='', escape='') WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, business_name, business_address FROM read_csv('{TRAIN_S3}', delim='\\t', header=true, quote='', escape='') WHERE country = '{country}';
        """
        pool_df = con.execute(q_pool).fetchdf()
        pool_size = len(pool_df)
        print(f"  Candidate pool size: {pool_size:,} records")
        
        # Precompute candidate signatures
        print("  Precomputing signatures for candidate pool...")
        pool_eids = pool_df["entity_id"].tolist()
        pool_sigs = []
        for name, addr in zip(pool_df["business_name"], pool_df["business_address"]):
            pool_sigs.append(get_record_signatures(name, addr))
        del pool_df
        
        # Build inverted index
        print("  Building multi-strategy inverted index...")
        inv_index = InvertedIndexEngine()
        inv_index.build_index(pool_sigs)
        print(f"  Index built with {len(inv_index.index):,} keys in {time.time()-t0:.1f}s")
        
        # Evaluate validation S1 entities
        print(f"  Running Tiered Cascade on {len(country_s1):,} S1 entities...")
        candidate_matches = []  # (s1_id, cand_id, score)
        
        for idx, row in country_s1.iterrows():
            s1_id = row["entity_id"]
            s1_sig = get_record_signatures(row["business_name"], row["business_address"])
            
            # Query inverted index for candidate set
            retrieved = inv_index.query(s1_sig, top_k=TOP_CANDIDATES_PER_S1)
            
            s1_accepted = []
            for pool_idx, r_score in retrieved:
                cand_id = pool_eids[pool_idx]
                cand_sig = pool_sigs[pool_idx]
                
                # Check Tier 1: Exact canonical match
                if evaluate_tier1_match(s1_sig, cand_sig):
                    candidate_matches.append((s1_id, cand_id, TIER1_SCORE))
                    tier_counts["tier1"] += 1
                    s1_accepted.append(cand_id)
                    continue
                
                # Check Tier 2: High-confidence fuzzy match with address anchor
                if evaluate_tier2_match(s1_sig, cand_sig):
                    candidate_matches.append((s1_id, cand_id, TIER2_SCORE))
                    tier_counts["tier2"] += 1
                    s1_accepted.append(cand_id)
                    continue
        
        # Resolve 1-to-1 mutual exclusivity
        country_preds = resolve_bipartite_matches(candidate_matches)
        for s1_id in country_s1["entity_id"]:
            all_predictions[s1_id] = country_preds.get(s1_id, [])
        
        del pool_sigs, pool_eids, inv_index
    
    # Compute official metrics
    metrics = compute_official_macro_f05(val_gt, all_predictions)
    
    print("\n" + "=" * 70)
    print("VALIDATION HARNESS RESULTS (src_v2 Tiered Cascade)")
    print("=" * 70)
    print(f"Macro F_0.5:           {metrics['macro_f05']:.4f}")
    print(f"Micro Precision:       {metrics['micro_precision']:.4f}")
    print(f"Micro Recall:          {metrics['micro_recall']:.4f}")
    print(f"Singleton Accuracy:    {metrics['singleton_acc']*100:.2f}% ({metrics['singleton_total']:,} singletons)")
    print(f"Matched Entity F_0.5:  {metrics['matched_acc']*100:.2f}% ({metrics['matched_total']:,} entities)")
    print("\nPerformance by Match Count Bucket:")
    for b_name, b_score in sorted(metrics["bucket_f05"].items()):
        print(f"  {b_name:<16}: F_0.5 = {b_score:.4f}")
    print("\nTier Acceptance Distribution:")
    for t_name, count in tier_counts.items():
        print(f"  {t_name:<16}: {count:,} candidates accepted")
    print("=" * 70)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Samples per country")
    args = parser.parse_args()
    run_tiered_validation(sample_size_per_country=args.samples)
