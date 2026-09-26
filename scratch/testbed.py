import sys, os, time, re, anyascii, math
import duckdb, psutil
import pandas as pd, numpy as np
from collections import defaultdict, Counter
import rapidfuzz
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("FAST TESTBED BENCHMARK (DuckDB + Fast Inverted Index + RapidFuzz)", flush=True)
print("="*70, flush=True)

STOPWORDS = {
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'holdings', 'holding', 'sarl', 'sasu', 'sas', 'sci', 'llp', 'pllc', 'gmbh',
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'floor', 'fl', 'unit', 'suite', 'ste', 'near', 'opp', 'behind', 'block',
    'sector', 'plot', 'no', 'building', 'bldg', 'tower', 'city', 'state',
    'rue', 'boulevard', 'blvd', 'allee', 'des', 'du', 'de', 'la', 'le'
}

def get_tokens(name, addr):
    tokens = set()
    if isinstance(name, str) and name:
        c_name = anyascii.anyascii(name).lower()
        c_name = re.sub(r'[^a-z0-9\s]', ' ', c_name)
        words = c_name.split()
        for w in words:
            if len(w) >= 3 and w not in STOPWORDS:
                tokens.add('N:' + w)
        if len(words) >= 2:
            w1, w2 = words[0], words[1]
            if w1 not in STOPWORDS and w2 not in STOPWORDS:
                tokens.add('NB:' + w1 + '_' + w2)
    if isinstance(addr, str) and addr:
        c_addr = anyascii.anyascii(addr).lower()
        nums = re.findall(r'\b\d{2,6}\b', c_addr)
        for num in nums:
            tokens.add('NUM:' + num)
        c_addr = re.sub(r'[^a-z0-9\s]', ' ', c_addr)
        for w in c_addr.split():
            if len(w) >= 4 and w not in STOPWORDS:
                tokens.add('A:' + w)
    return tokens

# 1. DuckDB setup to load S1, GT, and targets + 50k distractors
t0 = time.time()
con = duckdb.connect()
con.execute("""
    CREATE TABLE s1 AS 
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 2000;
""")
con.execute("""
    CREATE TABLE gt AS 
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='') 
    WHERE source1_entity_id IN (SELECT entity_id FROM s1);
""")
con.execute("""
    CREATE TABLE target_ids AS 
    SELECT unnest(string_split(matched_entity_ids, ',')) as mid 
    FROM gt 
    WHERE matched_entity_ids IS NOT NULL AND length(matched_entity_ids) > 0;
""")

# Load matching targets + 50k distractor rows each from S2 and S3
s2_df = con.execute("""
    SELECT * FROM (
        SELECT s2.* FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') s2
        SEMI JOIN target_ids t ON s2.entity_id = t.mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    );
""").df().drop_duplicates(subset=['entity_id'])

s3_df = con.execute("""
    SELECT * FROM (
        SELECT s3.* FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') s3
        SEMI JOIN target_ids t ON s3.entity_id = t.mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    );
""").df().drop_duplicates(subset=['entity_id'])

pool = pd.concat([s2_df, s3_df], ignore_index=True)
s1_df = con.execute("SELECT * FROM s1").df()
gt_df = con.execute("SELECT * FROM gt").df()

print(f"Loaded {len(s1_df)} S1 records, pool of {len(pool):,} S2/S3 candidate records in {time.time()-t0:.2f}s", flush=True)

gt_dict = {}
all_matched_ids = set()
for _, r in gt_df.iterrows():
    if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids']:
        targets = set(r['matched_entity_ids'].split(','))
        gt_dict[r['source1_entity_id']] = targets
        all_matched_ids.update(targets)

print(f"Matched targets present in pool: {pool['entity_id'].isin(all_matched_ids).sum()} / {len(all_matched_ids)}", flush=True)

# 2. Build Inverted Index
print("\nBuilding Inverted Index...", flush=True)
t0 = time.time()
inv_idx = defaultdict(list)
doc_freq = Counter()

pool_records = []
for row in pool.itertuples(index=False):
    eid, name, addr, country = row.entity_id, row.business_name, row.business_address, row.country
    toks = get_tokens(name, addr)
    idx = len(pool_records)
    pool_records.append((eid, country, toks, name, addr))
    for t in toks:
        inv_idx[(country, t)].append(idx)
        doc_freq[(country, t)] += 1

t1 = time.time()
print(f"Indexed {len(pool_records):,} records in {t1-t0:.2f}s. Unique keys: {len(inv_idx):,}", flush=True)

# 3. Query S1 entities & evaluate Top-K Recall
for K in [10, 20, 30]:
    total_true = 0
    recalled = 0
    t0 = time.time()
    for row in s1_df.itertuples(index=False):
        s1_id, s1_name, s1_addr, s1_country = row.entity_id, row.business_name, row.business_address, row.country
        true_matches = gt_dict.get(s1_id, set()) & set(pool['entity_id'])
        if not true_matches:
            continue
        total_true += len(true_matches)
        
        s1_toks = get_tokens(s1_name, s1_addr)
        cand_scores = defaultdict(float)
        for t in s1_toks:
            key = (s1_country, t)
            df = doc_freq.get(key, 0)
            if df == 0 or df > 10000:
                continue
            idf = math.log((len(pool_records) + 1) / (df + 1)) + 1.0
            w = 2.5 if t.startswith('N:') or t.startswith('NB:') else (1.5 if t.startswith('NUM:') else 1.0)
            for p_idx in inv_idx[key]:
                cand_scores[p_idx] += idf * w
        
        if cand_scores:
            top_k = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:K]
            cand_eids = {pool_records[idx][0] for idx, _ in top_k}
        else:
            cand_eids = set()
        
        recalled += len(true_matches & cand_eids)
    
    t1 = time.time()
    recall_pct = (recalled / total_true * 100) if total_true > 0 else 0
    print(f"Top-{K:2d} Recall: {recalled}/{total_true} ({recall_pct:.2f}%) in {t1-t0:.2f}s ({len(s1_df)/(t1-t0):.0f} queries/s)", flush=True)

print("="*70, flush=True)
