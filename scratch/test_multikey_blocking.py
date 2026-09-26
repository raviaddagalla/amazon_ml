import sys, time, re, anyascii
import duckdb, pandas as pd, numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("IMPROVED MULTI-KEY BLOCKING RECALL & EFFICIENCY TEST", flush=True)
print("="*70, flush=True)

STOPWORDS = {
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'holdings', 'holding', 'sarl', 'sasu', 'sas', 'sci', 'llp', 'pllc', 'gmbh',
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'floor', 'fl', 'unit', 'suite', 'ste', 'near', 'opp', 'behind', 'block',
    'sector', 'plot', 'no', 'building', 'bldg', 'tower',
    'rue', 'boulevard', 'blvd', 'allee', 'des', 'du', 'de', 'la', 'le'
}

DOMAIN_CLEAN_RE = re.compile(r'(\.com|\.net|\.org|\.in|\.co|\.fr|\.io|\.biz|www\.|https?://|@)')

def clean_tokens(text):
    if not isinstance(text, str): return []
    # Strip web extensions & URLs
    text = DOMAIN_CLEAN_RE.sub(' ', text)
    text = anyascii.anyascii(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if w]

def extract_numbers(addr):
    if not isinstance(addr, str): return []
    nums = []
    for raw in re.findall(r'\b\d+\b', addr):
        n = raw.lstrip('0')
        if n and 1 <= len(n) <= 6:
            nums.append(n)
    return nums

def extract_blocking_keys(name, addr):
    keys = set()
    name_toks = [w for w in clean_tokens(name) if w not in STOPWORDS]
    addr_toks = [w for w in clean_tokens(addr) if w not in STOPWORDS]
    nums = extract_numbers(addr)

    # 1. Primary & Secondary Name Tokens
    if name_toks:
        # First word (if len >= 2)
        if len(name_toks[0]) >= 2:
            keys.add('n1:' + name_toks[0])
            if len(name_toks[0]) >= 4:
                keys.add('npref:' + name_toks[0][:4])
        # Second word
        if len(name_toks) >= 2 and len(name_toks[1]) >= 2:
            keys.add('n2:' + name_toks[1])
            keys.add(f'nb:{name_toks[0]}_{name_toks[1]}')
        # Third word if first 2 are very short
        if len(name_toks) >= 3 and len(name_toks[2]) >= 3:
            keys.add('n3:' + name_toks[2])

    # 2. Address Number + Locality/Street Word
    if nums and addr_toks:
        for num in nums[:3]:
            for at in addr_toks[:4]:
                if len(at) >= 3 and at != num:
                    keys.add(f'na:{num}_{at}')
    
    # 3. Distinctive PIN / Postal Code alone (5-6 digits)
    for num in nums:
        if len(num) in (5, 6):
            keys.add('pin:' + num)

    return keys

# 1. Load 5,000 S1 records and their ground truth
con = duckdb.connect()
print("Loading 5,000 S1 training records and ground truth...", flush=True)
s1_df = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 5000;
""").df()

gt_df = con.execute("""
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='') 
    WHERE source1_entity_id IN (SELECT entity_id FROM s1_df);
""").df()

target_ids = set()
gt_dict = {}
for _, r in gt_df.iterrows():
    if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids']:
        m = set(r['matched_entity_ids'].split(','))
        gt_dict[r['source1_entity_id']] = m
        target_ids.update(m)

print(f"5,000 S1 entities, {len(gt_dict)} have matches, total true target matches: {len(target_ids)}", flush=True)

# 2. Load candidate pool: targets + 100,000 distractors from S2 and S3
print("Loading candidate pool from S2 and S3 (including all targets + distractors)...", flush=True)
con.register('targets', pd.DataFrame({'mid': list(target_ids)}))

s2_targets = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') 
    SEMI JOIN targets ON entity_id = mid;
""").df()

s3_targets = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') 
    SEMI JOIN targets ON entity_id = mid;
""").df()

s2_distractors = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 50000;
""").df()

s3_distractors = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 50000;
""").df()

pool = pd.concat([s2_targets, s3_targets, s2_distractors, s3_distractors], ignore_index=True).drop_duplicates(subset=['entity_id'])
pool_id_set = set(pool['entity_id'])
present_targets = pool['entity_id'].isin(target_ids).sum()
print(f"Pool size: {len(pool):,} records. True targets present: {present_targets} / {len(target_ids)}", flush=True)

# 3. Build Blocking Index
print("\nIndexing candidate pool...", flush=True)
t0 = time.time()
inv_index = defaultdict(list)
key_counts = Counter()

for row in pool.itertuples(index=False):
    eid, name, addr, country = row.entity_id, row.business_name, row.business_address, row.country
    keys = extract_blocking_keys(name, addr)
    for k in keys:
        inv_index[(country, k)].append(eid)
        key_counts[(country, k)] += 1

t1 = time.time()
print(f"Pool indexed in {t1-t0:.2f}s. Unique (country, key) pairs: {len(inv_index):,}", flush=True)

# 4. Query S1 entities & measure recall & candidate count
print("\nQuerying S1 entities...", flush=True)
MAX_KEY_SIZE = 1000  # suppress keys that have more than 1000 records

t0 = time.time()
total_true = 0
recalled_true = 0
candidate_counts = []

for row in s1_df.itertuples(index=False):
    s1_id, s1_name, s1_addr, s1_country = row.entity_id, row.business_name, row.business_address, row.country
    true_set = gt_dict.get(s1_id, set()) & pool_id_set
    if not true_set:
        continue
    total_true += len(true_set)
    
    s1_keys = extract_blocking_keys(s1_name, s1_addr)
    candidates = set()
    for k in s1_keys:
        ck = (s1_country, k)
        if key_counts.get(ck, 0) > MAX_KEY_SIZE:
            continue
        candidates.update(inv_index.get(ck, ()))
            
    candidate_counts.append(len(candidates))
    recalled_true += len(true_set & candidates)

t1 = time.time()
print(f"Querying completed in {t1-t0:.2f}s ({len(s1_df)/(t1-t0):.0f} queries/s)", flush=True)
print(f"Total True Targets: {total_true:,}")
print(f"Recalled Targets  : {recalled_true:,} ({recalled_true / total_true * 100:.2f}%)")
print(f"Avg Candidates per S1: {np.mean(candidate_counts):.1f} (Median: {np.median(candidate_counts):.1f}, 90th percentile: {np.percentile(candidate_counts, 90):.1f})")
print(f"Candidate Space Reduction Ratio: {1.0 - (np.mean(candidate_counts) / len(pool)):.6%}")
print("="*70, flush=True)
