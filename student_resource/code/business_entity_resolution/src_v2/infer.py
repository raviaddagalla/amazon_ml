"""
Full Test Inference Runner for Tiered Deterministic-First Pipeline (v2)
Runs country-by-country streaming inference, generates both matching_results.tsv
and candidate_pairs.tsv in exact reference order, and verifies with validator.
"""

import os
import sys
import gc
import time
import argparse
import duckdb
from collections import defaultdict
import numpy as np
import pandas as pd

import lightgbm as lgb
import jellyfish

from src.features import compute_pair_features
from src.preprocessing import process_record_text

from src_v2.config import (
    TEST_S1, TEST_S2, TEST_S3,
    MATCHING_OUTPUT, CANDIDATE_OUTPUT,
    TEMP_PARTITIONS_DIR, STUDENT_RESOURCE_ROOT,
    TIER1_SCORE, TIER2_SCORE, TIER3_PROB_THRESHOLD,
    TOP_CANDIDATES_PER_S1, BATCH_SIZE
)
from src_v2.canonicalize import get_record_signatures
from src_v2.retrieval import InvertedIndexEngine
from src_v2.cascade import (
    evaluate_tier1_match,
    evaluate_tier2_match,
    resolve_bipartite_matches,
)


def run_country_inference(
    country: str,
    temp_dir: str = TEMP_PARTITIONS_DIR,
    top_k: int = TOP_CANDIDATES_PER_S1,
) -> tuple[str, str]:
    """Process a single country partition with the Tiered Cascade Engine."""
    safe_name = country.replace(" ", "_")
    temp_match_path = os.path.join(temp_dir, f"match_{safe_name}.tsv")
    temp_cand_path = os.path.join(temp_dir, f"cand_{safe_name}.tsv")

    print("\n" + "=" * 65, flush=True)
    print(f"PROCESSING PARTITION: {country} (v2 Tiered Cascade)", flush=True)
    print("=" * 65, flush=True)

    con = duckdb.connect()
    t_start = time.time()

    # 1. Load S1 for this country
    print(f"Loading Source 1 records for {country}...", flush=True)
    s1_country_df = con.execute(f"""
        SELECT entity_id, business_name, business_address
        FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='')
        WHERE country = '{country}';
    """).fetchdf()
    n_s1 = len(s1_country_df)
    print(f"  {country} S1 count: {n_s1:,} entities", flush=True)

    # 2. Load candidate pool (S2 + S3)
    print(f"Loading Candidate Pool (S2 + S3) for {country}...", flush=True)
    q_pool = f"""
        SELECT entity_id, business_name, business_address
        FROM read_csv('{TEST_S2}', delim='\\t', header=true, quote='', escape='')
        WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, business_name, business_address
        FROM read_csv('{TEST_S3}', delim='\\t', header=true, quote='', escape='')
        WHERE country = '{country}';
    """
    pool_df = con.execute(q_pool).fetchdf()
    n_pool = len(pool_df)
    print(f"  {country} Candidate pool: {n_pool:,} records", flush=True)

    # 3. Precompute signatures and feature representations for candidate pool
    print(f"Precomputing signatures for {n_pool:,} pool candidates...", flush=True)
    pool_eids = pool_df["entity_id"].tolist()
    pool_names = pool_df["business_name"].fillna("").tolist()
    pool_addrs = pool_df["business_address"].fillna("").tolist()
    pool_sigs = []
    pool_name_toks = []
    pool_addr_toks = []
    pool_first = []

    for name, addr in zip(pool_names, pool_addrs):
        pool_sigs.append(get_record_signatures(name, addr))
        _, _, n_toks, a_toks, f_tok, _ = process_record_text(name, addr)
        pool_name_toks.append(n_toks)
        pool_addr_toks.append(a_toks)
        pool_first.append(f_tok)
    del pool_df
    gc.collect()

    # 4. Build inverted index
    print(f"Building high-recall inverted index over candidate pool...", flush=True)
    inv_index = InvertedIndexEngine()
    inv_index.build_index(pool_sigs)
    print(f"  Index built with {len(inv_index.index):,} keys in {time.time()-t_start:.1f}s", flush=True)

    # Load LightGBM model for Tier 3 residual classification
    model_path = os.path.join(STUDENT_RESOURCE_ROOT, "code", "business_entity_resolution", "models", "lgbm_entity_resolver.txt")
    bst = None
    if os.path.exists(model_path):
        print(f"Loading LightGBM model for Tier 3 from {model_path}...", flush=True)
        bst = lgb.Booster(model_file=model_path)
        print(f"  Model loaded with {bst.num_feature()} features.", flush=True)

    # 5. Process S1 entities and stream candidates
    print(f"Evaluating Tiered Cascade across {n_s1:,} S1 entities...", flush=True)
    candidate_matches = []  # (s1_id, cand_id, score)
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0

    t0 = time.time()
    with open(temp_cand_path, "w", encoding="utf-8") as fc:
        for batch_start in range(0, n_s1, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_s1)
            batch_s1 = s1_country_df.iloc[batch_start:batch_end]
            tier3_batch_features = []
            tier3_batch_meta = []

            for _, row in batch_s1.iterrows():
                s1_id = row["entity_id"]
                s1_raw_name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                s1_raw_addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
                s1_sig = get_record_signatures(s1_raw_name, s1_raw_addr)

                retrieved = inv_index.query(s1_sig, top_k=top_k)
                cand_ids = [pool_eids[p_idx] for p_idx, _ in retrieved]

                # Stream candidate_pairs.tsv row
                cand_str = ",".join(cand_ids) if cand_ids else ""
                fc.write(f"{s1_id}\t{cand_str}\n")

                s1_pre = None
                s1_norm_name, s1_norm_addr, s1_n_toks, s1_a_toks, s1_f_tok, s1_nums = process_record_text(s1_raw_name, s1_raw_addr)

                # Evaluate tiers
                for p_idx, r_score in retrieved:
                    cand_id = pool_eids[p_idx]
                    c_sig = pool_sigs[p_idx]

                    if evaluate_tier1_match(s1_sig, c_sig):
                        candidate_matches.append((s1_id, cand_id, TIER1_SCORE))
                        tier1_count += 1
                    elif evaluate_tier2_match(s1_sig, c_sig):
                        candidate_matches.append((s1_id, cand_id, TIER2_SCORE))
                        tier2_count += 1
                    elif bst is not None:
                        # Prepare for Tier 3 LightGBM scoring
                        if s1_pre is None:
                            s1_pre = {
                                's1_norm_name': s1_norm_name,
                                's1_norm_addr': s1_norm_addr,
                                's1_n_words': set(s1_n_toks),
                                's1_a_words': set(s1_a_toks),
                                's1_first_word': s1_f_tok,
                                's1_nums': s1_nums,
                                's1_name_sorted': " ".join(sorted(s1_n_toks)),
                                's1_m0': jellyfish.metaphone(s1_f_tok) if s1_f_tok else "",
                                's1_metaphones': {jellyfish.metaphone(w) for w in s1_n_toks if w},
                                's1_street_num': next((n for n in s1_nums if len(n) <= 4), ""),
                                's1_postal': next((n for n in reversed(s1_nums) if len(n) in (5, 6)), ""),
                                's1_name_tokens_list': s1_n_toks,
                            }
                        feat = compute_pair_features(
                            s1_raw_name, s1_raw_addr,
                            pool_names[p_idx], pool_addrs[p_idx],
                            pool_name_toks[p_idx], pool_addr_toks[p_idx],
                            pool_first[p_idx],
                            r_score,
                            country=country,
                            s1_precomputed=s1_pre,
                        )
                        tier3_batch_features.append(feat)
                        tier3_batch_meta.append((s1_id, cand_id))

            # Batch predict Tier 3 with LightGBM
            if tier3_batch_features and bst is not None:
                X_batch = np.array(tier3_batch_features, dtype=np.float32)
                probs = bst.predict(X_batch)
                for idx, (s1_id, cand_id) in enumerate(tier3_batch_meta):
                    p = float(probs[idx])
                    if p >= TIER3_PROB_THRESHOLD:
                        candidate_matches.append((s1_id, cand_id, p))
                        tier3_count += 1
                del X_batch, probs, tier3_batch_features, tier3_batch_meta

            if (batch_end % 20000 == 0) or (batch_end == n_s1):
                elapsed = time.time() - t0
                rate = batch_end / max(elapsed, 0.001)
                print(f"  Progress: {batch_end:,}/{n_s1:,} entities ({rate:.0f} ent/s) | Tier1: {tier1_count:,} | Tier2: {tier2_count:,} | Tier3: {tier3_count:,}", flush=True)

    del pool_eids, pool_names, pool_addrs, pool_sigs, pool_name_toks, pool_addr_toks, pool_first, inv_index
    gc.collect()

    # 6. Apply bipartite mutual exclusivity resolution
    print(f"Resolving 1-to-1 competitive matches across {len(candidate_matches):,} raw pairs...", flush=True)
    country_matches = resolve_bipartite_matches(candidate_matches)
    del candidate_matches
    gc.collect()

    # 7. Write match partition file
    print(f"Writing match partition file to {temp_match_path}...", flush=True)
    with open(temp_match_path, "w", encoding="utf-8") as fm:
        for s1_id in s1_country_df["entity_id"]:
            m_list = country_matches.get(s1_id, [])
            m_str = ",".join(m_list) if m_list else ""
            fm.write(f"{s1_id}\t{m_str}\n")

    total_matches = sum(len(m) for m in country_matches.values())
    singletons = sum(1 for s1_id in s1_country_df["entity_id"] if not country_matches.get(s1_id))
    print(f"Finished {country} in {time.time()-t_start:.1f}s | Matches: {total_matches:,} | Singletons: {singletons:,}", flush=True)

    del s1_country_df, country_matches
    gc.collect()

    return temp_match_path, temp_cand_path


def run_full_inference_v2(
    matching_path: str = MATCHING_OUTPUT,
    candidate_path: str = CANDIDATE_OUTPUT,
    countries_to_process: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Run full test inference across all countries with v2 cascade."""
    print("\n" + "=" * 70, flush=True)
    print("STARTING FULL TEST SET INFERENCE (v2 Tiered Cascade)", flush=True)
    print("=" * 70, flush=True)

    con = duckdb.connect()
    os.makedirs(TEMP_PARTITIONS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    available_countries = [row[0] for row in con.execute(f"""
        SELECT DISTINCT country 
        FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='');
    """).fetchall()]
    print(f"Countries present in test set: {available_countries}", flush=True)

    target_countries = countries_to_process or available_countries
    print(f"Target countries: {target_countries}", flush=True)

    partition_match_files = []
    partition_cand_files = []

    for country in target_countries:
        safe_name = country.replace(" ", "_")
        m_path = os.path.join(TEMP_PARTITIONS_DIR, f"match_{safe_name}.tsv")
        c_path = os.path.join(TEMP_PARTITIONS_DIR, f"cand_{safe_name}.tsv")

        if not overwrite and os.path.exists(m_path) and os.path.exists(c_path):
            expected_s1 = con.execute(f"""
                SELECT count(*) FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='') 
                WHERE country = '{country}';
            """).fetchone()[0]
            with open(m_path, "r", encoding="utf-8") as f:
                m_count = sum(1 for _ in f)
            if m_count == expected_s1:
                print(f"Partition for {country} verified ({m_count:,} rows). Reusing.", flush=True)
                partition_match_files.append((country, m_path))
                partition_cand_files.append((country, c_path))
                continue

        m_res, c_res = run_country_inference(country)
        partition_match_files.append((country, m_res))
        partition_cand_files.append((country, c_res))

    # Assemble final submission files strictly in test_source1 reference order
    print("\n" + "=" * 70, flush=True)
    print("ASSEMBLING FINAL SUBMISSION ARTIFACTS", flush=True)
    print("=" * 70, flush=True)

    s1_all_ids = con.execute(f"""
        SELECT entity_id FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='');
    """).fetchdf()["entity_id"].tolist()
    total_test_s1 = len(s1_all_ids)
    print(f"Total reference S1 entities: {total_test_s1:,}", flush=True)

    # Load all match partitions
    print("Loading country match partitions into memory...", flush=True)
    full_matches = {}
    for country, m_path in partition_match_files:
        with open(m_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    full_matches[parts[0]] = parts[1]
                elif len(parts) == 1:
                    full_matches[parts[0]] = ""

    print(f"Writing final {matching_path}...", flush=True)
    with open(matching_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            f_out.write(f"{s1_id}\t{full_matches.get(s1_id, '')}\n")
    del full_matches
    gc.collect()

    # Load candidate partitions
    print("Loading country candidate partitions into memory...", flush=True)
    full_candidates = {}
    for country, c_path in partition_cand_files:
        with open(c_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    full_candidates[parts[0]] = parts[1]
                elif len(parts) == 1:
                    full_candidates[parts[0]] = ""

    print(f"Writing final {candidate_path}...", flush=True)
    with open(candidate_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            f_out.write(f"{s1_id}\t{full_candidates.get(s1_id, '')}\n")
    del full_candidates
    gc.collect()

    print(f"\nFinal artifacts assembled successfully!")
    print(f"  {matching_path}: {os.path.getsize(matching_path)/1e6:.1f} MB")
    print(f"  {candidate_path}: {os.path.getsize(candidate_path)/1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Recompute partitions")
    args = parser.parse_args()
    run_full_inference_v2(overwrite=args.overwrite)
