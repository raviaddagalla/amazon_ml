"""
Candidate Pool Builder for Realistic Distractor Mining
Builds pre-normalized pools and inverted indexes in memory with zero disk thrashing.
"""

import os
import time
import gc
from array import array
import duckdb

from src.config import (
    MAX_BLOCKING_KEY_SIZE,
    TRAIN_S2,
    TRAIN_S3,
)
from src.preprocessing import process_record_text
from src.blocking import get_blocking_keys_from_tokens


def build_candidate_pool(country: str):
    """Build the candidate pool and inverted index for a given country in memory.
    
    Returns:
        tuple: (pool_eids, pool_n_norm, pool_a_norm, pool_nums, pool_first, inv_index)
    """
    print(f"Building candidate pool for {country} in memory...", flush=True)
    con = duckdb.connect()
    t0 = time.time()
    pool_df = con.execute(f"""
        SELECT entity_id, business_name, business_address
        FROM read_csv('{TRAIN_S2}', delim='\t', header=true, quote='', escape='')
        WHERE country = '{country}'
        UNION ALL
        SELECT entity_id, business_name, business_address
        FROM read_csv('{TRAIN_S3}', delim='\t', header=true, quote='', escape='')
        WHERE country = '{country}'
    """).df()
    n_pool = len(pool_df)
    print(f"  Loaded {n_pool:,} records in {time.time()-t0:.1f}s", flush=True)
    
    t0 = time.time()
    pool_eids = pool_df["entity_id"].values
    pool_names_raw = pool_df["business_name"].fillna("").values
    pool_addrs_raw = pool_df["business_address"].fillna("").values
    
    pool_n_norm = []
    pool_a_norm = []
    pool_nums = []
    pool_first = []
    
    inv_index: dict[str, array] = {}
    
    for idx in range(n_pool):
        c_n, c_a, n_toks, a_toks, first_w, nums = process_record_text(
            pool_names_raw[idx], pool_addrs_raw[idx]
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
            elif len(postings) <= MAX_BLOCKING_KEY_SIZE:
                postings.append(idx)
        
        if (idx + 1) % 1000000 == 0:
            print(f"    Indexed {idx+1:,}/{n_pool:,} records ({time.time()-t0:.1f}s)...", flush=True)
    
    del pool_df, pool_names_raw, pool_addrs_raw
    gc.collect()
    print(f"  Pool indexed in {time.time()-t0:.1f}s. Unique keys: {len(inv_index):,}", flush=True)
    
    return (
        pool_eids,
        pool_n_norm,
        pool_a_norm,
        pool_nums,
        pool_first,
        inv_index,
    )


# Alias for compatibility with callers
get_cached_pool = lambda country, force_rebuild=False: build_candidate_pool(country)
