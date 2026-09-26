"""
High-Performance Streaming Inference Pipeline for Full Test Set Entity Resolution
"""

import os
import time
import gc
from array import array
from collections import Counter, defaultdict
import duckdb
import numpy as np
import lightgbm as lgb

from src.config import (
    TEST_S1, TEST_S2, TEST_S3, MODEL_PATH, OUTPUT_DIR,
    MATCHING_OUTPUT, CANDIDATE_OUTPUT,
    TOP_CANDIDATES_PER_S1, PROBABILITY_THRESHOLD
)
from src.preprocessing import process_record_text
from src.blocking import get_blocking_keys_from_tokens
from src.features import compute_pair_features


def run_country_inference(
    country: str,
    bst: lgb.Booster,
    temp_dir: str
) -> tuple[str, str]:
    """
    Process a single country partition with minimal RAM and stream outputs directly to disk.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"STARTING COUNTRY PARTITION: {country.upper()}", flush=True)
    print(f"{'='*70}", flush=True)
    
    t_start = time.time()
    con = duckdb.connect()

    # 1. Load S1 reference entities for this country
    print(f"Loading Source 1 records for {country}...", flush=True)
    t0 = time.time()
    s1_country_df = con.execute(f"""
        SELECT entity_id, business_name, business_address 
        FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='') 
        WHERE country = '{country}';
    """).df()
    n_s1 = len(s1_country_df)
    print(f"Loaded {n_s1:,} Source 1 entities for {country} in {time.time()-t0:.2f}s.", flush=True)

    if n_s1 == 0:
        return "", ""

    # 2. Load candidate pool (S2 and S3 for this country)
    print(f"Loading Source 2 and Source 3 candidate pool for {country}...", flush=True)
    t0 = time.time()
    pool_df = con.execute(f"""
        SELECT entity_id, business_name, business_address 
        FROM read_csv('{TEST_S2}', delim='\\t', header=true, quote='', escape='') 
        WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, business_name, business_address 
        FROM read_csv('{TEST_S3}', delim='\\t', header=true, quote='', escape='') 
        WHERE country = '{country}';
    """).df()
    n_pool = len(pool_df)
    print(f"Loaded {n_pool:,} candidate pool records in {time.time()-t0:.2f}s.", flush=True)

    # 3. Pre-normalize pool into compact memory structures and build inverted index
    print("Pre-normalizing candidate pool and building multi-key blocking index...", flush=True)
    t0 = time.time()
    
    pool_eids = pool_df["entity_id"].values
    pool_names = pool_df["business_name"].fillna("").values
    pool_addrs = pool_df["business_address"].fillna("").values

    pool_n_norm = []
    pool_a_norm = []
    pool_nums = []
    pool_first = []
    
    inv_index: dict[str, array] = {}

    for idx in range(n_pool):
        c_n, c_a, n_toks, a_toks, first_w, nums = process_record_text(
            pool_names[idx], pool_addrs[idx]
        )
        
        pool_n_norm.append(c_n)
        pool_a_norm.append(c_a)
        pool_nums.append(tuple(nums))
        pool_first.append(first_w)
        
        keys = get_blocking_keys_from_tokens(n_toks, a_toks, nums)
        for k in keys:
            postings = inv_index.get(k)
            if postings is None:
                inv_index[k] = array('i', [idx])
            elif len(postings) <= 500:
                postings.append(idx)

    del pool_df, pool_names, pool_addrs
    gc.collect()
    print(f"Candidate pool pre-normalized & indexed in {time.time()-t0:.2f}s. Unique keys: {len(inv_index):,}", flush=True)

    # 4. Stream S1 entities in batches, write candidates to disk, and update best assignments on-the-fly
    safe_name = country.replace(" ", "_")
    temp_match_path = os.path.join(temp_dir, f"match_{safe_name}.tsv")
    temp_cand_path = os.path.join(temp_dir, f"cand_{safe_name}.tsv")

    print(f"Streaming S1 batches and performing pair inference...", flush=True)
    BATCH_SIZE = 25000
    best_assignment: dict[str, tuple[str, float]] = {}  # cand_eid -> (s1_id, prob)
    total_candidates_emitted = 0

    t0 = time.time()
    with open(temp_cand_path, "w", encoding="utf-8") as fc:
        for batch_start in range(0, n_s1, BATCH_SIZE):
            batch_df = s1_country_df.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_features = []
            batch_meta = []

            for row in batch_df.itertuples(index=False):
                s1_id = row.entity_id
                raw_n = row.business_name
                raw_a = row.business_address

                s1_name_norm, s1_addr_norm, n_toks, a_toks, s1_first_word, nums = process_record_text(raw_n, raw_a)
                s1_nums = set(nums)
                s1_n_words = set(n_toks)
                s1_a_words = set(a_toks)

                s1_keys = get_blocking_keys_from_tokens(n_toks, a_toks, nums)
                cand_weights = Counter()
                for k in s1_keys:
                    postings = inv_index.get(k)
                    if postings is None or len(postings) > 500:
                        continue
                    w = 5.0 if (k.startswith("n1:") or k.startswith("nb:")) else (
                        3.0 if k.startswith("pin:") else (2.0 if k.startswith("na:") else 1.0)
                    )
                    for pool_idx in postings:
                        cand_weights[pool_idx] += w

                if not cand_weights:
                    fc.write(f"{s1_id}\t\n")
                    continue

                top_cands = cand_weights.most_common(TOP_CANDIDATES_PER_S1)
                cand_eids = [pool_eids[idx] for idx, _ in top_cands]
                fc.write(f"{s1_id}\t{','.join(cand_eids)}\n")
                total_candidates_emitted += len(cand_eids)

                for pool_idx, blk_score in top_cands:
                    cand_eid = pool_eids[pool_idx]
                    feat = compute_pair_features(
                        s1_name_norm, s1_addr_norm, s1_nums, s1_n_words, s1_a_words, s1_first_word,
                        pool_n_norm[pool_idx], pool_a_norm[pool_idx],
                        set(pool_nums[pool_idx]),
                        set(pool_n_norm[pool_idx].split()),
                        set(pool_a_norm[pool_idx].split()),
                        pool_first[pool_idx],
                        blk_score
                    )
                    batch_features.append(feat)
                    batch_meta.append((s1_id, cand_eid))

            if batch_features:
                X_batch = np.array(batch_features, dtype=np.float32)
                probs = bst.predict(X_batch)
                for idx, (s1_id, cand_eid) in enumerate(batch_meta):
                    p = float(probs[idx])
                    if p >= PROBABILITY_THRESHOLD:
                        prev = best_assignment.get(cand_eid)
                        if prev is None or p > prev[1]:
                            best_assignment[cand_eid] = (s1_id, p)
                del X_batch, probs, batch_features, batch_meta

            progress = min(batch_start + BATCH_SIZE, n_s1)
            elapsed = time.time() - t0
            rate = progress / max(elapsed, 0.001)
            print(f"  Progress: {progress:,}/{n_s1:,} entities ({rate:.0f} q/s) | Active matches: {len(best_assignment):,}", flush=True)

    # Free pool memory and inverted index
    del pool_eids, pool_n_norm, pool_a_norm, pool_nums, pool_first, inv_index
    gc.collect()

    # Invert best_assignment to get final matches per S1 entity
    country_matches = defaultdict(list)
    for cand_eid, (s1_id, _) in best_assignment.items():
        country_matches[s1_id].append(cand_eid)
    del best_assignment
    gc.collect()

    # Write match partition file
    print(f"Writing match partition file to {temp_match_path}...", flush=True)
    with open(temp_match_path, "w", encoding="utf-8") as fm:
        for s1_id in s1_country_df["entity_id"]:
            matches = country_matches.get(s1_id)
            m_str = ",".join(matches) if matches else ""
            fm.write(f"{s1_id}\t{m_str}\n")

    total_matches = sum(len(m) for m in country_matches.values())
    singletons = sum(1 for s1_id in s1_country_df["entity_id"] if not country_matches.get(s1_id))
    print(f"Completed {country} in {time.time()-t_start:.1f}s | Matches: {total_matches:,} | Singletons: {singletons:,}", flush=True)

    del s1_country_df, country_matches
    gc.collect()

    return temp_match_path, temp_cand_path


def run_full_inference(
    matching_path: str = MATCHING_OUTPUT,
    candidate_path: str = CANDIDATE_OUTPUT,
    countries_to_process: list[str] | None = None
) -> None:
    """
    Run full test inference across all countries and export submission files.
    """
    print("\n" + "=" * 70, flush=True)
    print("STARTING FULL TEST SET INFERENCE PIPELINE", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")

    print(f"Loading LightGBM model from {MODEL_PATH}...", flush=True)
    bst = lgb.Booster(model_file=MODEL_PATH)
    
    con = duckdb.connect()

    temp_dir = os.path.join(OUTPUT_DIR, "temp_partitions")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    # Discover all unique countries in test set dynamically (open set)
    available_countries = [row[0] for row in con.execute(f"""
        SELECT DISTINCT country 
        FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='');
    """).fetchall()]
    print(f"Countries present in test set: {available_countries}", flush=True)

    target_countries = countries_to_process or available_countries
    print(f"Countries to process in this run: {target_countries}", flush=True)

    t_pipeline = time.time()
    partition_match_files = []
    partition_cand_files = []

    for country in target_countries:
        safe_name = country.replace(" ", "_")
        m_path = os.path.join(temp_dir, f"match_{safe_name}.tsv")
        c_path = os.path.join(temp_dir, f"cand_{safe_name}.tsv")
        
        # Check if complete partition files already exist
        if os.path.exists(m_path) and os.path.exists(c_path):
            expected_s1 = con.execute(f"""
                SELECT count(*) FROM read_csv('{TEST_S1}', delim='\\t', header=true, quote='', escape='') 
                WHERE country = '{country}';
            """).fetchone()[0]
            with open(m_path, "r", encoding="utf-8") as f:
                m_count = sum(1 for _ in f)
            if m_count == expected_s1:
                print(f"Partition files for {country} already exist and verified ({m_count:,} rows). Reusing.", flush=True)
                partition_match_files.append(m_path)
                partition_cand_files.append(c_path)
                continue

        m_path, c_path = run_country_inference(country, bst, temp_dir)
        if m_path and c_path:
            partition_match_files.append(m_path)
            partition_cand_files.append(c_path)

    # If all countries were processed, assemble the final ordered submission files
    if set(target_countries) == set(available_countries):
        print(f"\n{'='*70}", flush=True)
        print("ASSEMBLING FINAL SUBMISSION FILES IN EXACT TEST SET REFERENCE ORDER", flush=True)
        print(f"{'='*70}", flush=True)
        
        # 1. Assemble matching_results.tsv
        t0 = time.time()
        print("Consolidating matches in reference order...", flush=True)
        match_lookup = {}
        for p in partition_match_files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    match_lookup[parts[0]] = parts[1] if len(parts) > 1 else ""

        with open(matching_path, "w", encoding="utf-8") as fm, open(TEST_S1, "r", encoding="utf-8") as f_ref:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            f_ref.readline()  # skip header
            n_match_written = 0
            for line in f_ref:
                line_clean = line.strip()
                if not line_clean:
                    continue
                s1_id = line_clean.split("\t", 1)[0]
                fm.write(f"{s1_id}\t{match_lookup.get(s1_id, '')}\n")
                n_match_written += 1

        print(f"Successfully written {n_match_written:,} matching records in {time.time()-t0:.2f}s!", flush=True)
        del match_lookup
        gc.collect()

        # 2. Assemble candidate_pairs.tsv
        t0 = time.time()
        print("Consolidating candidates in reference order...", flush=True)
        cand_lookup = {}
        for p in partition_cand_files:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    cand_lookup[parts[0]] = parts[1] if len(parts) > 1 else ""

        with open(candidate_path, "w", encoding="utf-8") as fc, open(TEST_S1, "r", encoding="utf-8") as f_ref:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            f_ref.readline()  # skip header
            n_cand_written = 0
            for line in f_ref:
                line_clean = line.strip()
                if not line_clean:
                    continue
                s1_id = line_clean.split("\t", 1)[0]
                fc.write(f"{s1_id}\t{cand_lookup.get(s1_id, '')}\n")
                n_cand_written += 1

        print(f"Successfully written {n_cand_written:,} candidate records in {time.time()-t0:.2f}s!", flush=True)
        del cand_lookup
        gc.collect()

    print(f"\nFull end-to-end inference finished in {(time.time()-t_pipeline)/60:.2f} minutes!", flush=True)


if __name__ == "__main__":
    run_full_inference()
