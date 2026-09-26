import sys, time, re, anyascii
import duckdb, pandas as pd, numpy as np
from collections import defaultdict, Counter
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("END-TO-END MATCHING & F_0.5 EVALUATION BENCHMARK", flush=True)
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
    text = DOMAIN_CLEAN_RE.sub(' ', text)
    text = anyascii.anyascii(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [w for w in text.split() if w]

def norm_str(text):
    if not isinstance(text, str): return ''
    text = DOMAIN_CLEAN_RE.sub(' ', text)
    text = anyascii.anyascii(text).lower()
    return re.sub(r'[^a-z0-9\s]', ' ', text).strip()

def extract_numbers(addr):
    if not isinstance(addr, str): return set()
    nums = set()
    for raw in re.findall(r'\b\d+\b', addr):
        n = raw.lstrip('0')
        if n and 1 <= len(n) <= 6:
            nums.add(n)
    return nums

def extract_blocking_keys(name, addr):
    keys = set()
    name_toks = [w for w in clean_tokens(name) if w not in STOPWORDS]
    addr_toks = [w for w in clean_tokens(addr) if w not in STOPWORDS]
    nums = list(extract_numbers(addr))

    if name_toks:
        if len(name_toks[0]) >= 2:
            keys.add('n1:' + name_toks[0])
            if len(name_toks[0]) >= 4:
                keys.add('npref:' + name_toks[0][:4])
        if len(name_toks) >= 2 and len(name_toks[1]) >= 2:
            keys.add('n2:' + name_toks[1])
            keys.add(f'nb:{name_toks[0]}_{name_toks[1]}')
        if len(name_toks) >= 3 and len(name_toks[2]) >= 3:
            keys.add('n3:' + name_toks[2])

    if nums and addr_toks:
        for num in nums[:3]:
            for at in addr_toks[:4]:
                if len(at) >= 3 and at != num:
                    keys.add(f'na:{num}_{at}')
    
    for num in nums:
        if len(num) in (5, 6):
            keys.add('pin:' + num)

    return keys

# 1. Load S1 and Ground Truth
con = duckdb.connect()
print("Loading S1 and GT...", flush=True)
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

gt_dict = {}
all_target_ids = set()
for _, r in gt_df.iterrows():
    if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids']:
        m = set(r['matched_entity_ids'].split(','))
        gt_dict[r['source1_entity_id']] = m
        all_target_ids.update(m)
    else:
        gt_dict[r['source1_entity_id']] = set()

# 2. Load candidate pool
con.register('targets', pd.DataFrame({'mid': list(all_target_ids)}))
s2 = con.execute("""
    SELECT * FROM (
        SELECT s2.* FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') s2
        SEMI JOIN targets ON entity_id = mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    )
""").df().drop_duplicates(subset=['entity_id'])

s3 = con.execute("""
    SELECT * FROM (
        SELECT s3.* FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') s3
        SEMI JOIN targets ON entity_id = mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    )
""").df().drop_duplicates(subset=['entity_id'])

pool = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"Pool size: {len(pool):,} records. True targets present: {pool['entity_id'].isin(all_target_ids).sum()} / {len(all_target_ids)}", flush=True)

# Pre-normalize pool for ultra-fast RapidFuzz scoring
pool_data = {}
inv_index = defaultdict(list)
key_counts = Counter()

t0 = time.time()
for row in pool.itertuples(index=False):
    eid = row.entity_id
    n_norm = norm_str(row.business_name)
    a_norm = norm_str(row.business_address)
    nums = extract_numbers(row.business_address)
    country = row.country
    pool_data[eid] = (n_norm, a_norm, nums)
    
    keys = extract_blocking_keys(row.business_name, row.business_address)
    for k in keys:
        inv_index[(country, k)].append(eid)
        key_counts[(country, k)] += 1

print(f"Candidate pool pre-normalized & indexed in {time.time()-t0:.2f}s", flush=True)

# 3. Blocking & Scoring
print("\nRunning Blocking and Scoring...", flush=True)
t0 = time.time()
MAX_KEY_SIZE = 400

scored_pairs = [] # (s1_id, cand_id, score)

for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    s1_name_norm = norm_str(row.business_name)
    s1_addr_norm = norm_str(row.business_address)
    s1_nums = extract_numbers(row.business_address)
    country = row.country
    
    s1_keys = extract_blocking_keys(row.business_name, row.business_address)
    cands = set()
    for k in s1_keys:
        ck = (country, k)
        if key_counts.get(ck, 0) > MAX_KEY_SIZE:
            continue
        cands.update(inv_index.get(ck, ()))
        
    for cand_id in cands:
        c_n, c_a, c_nums = pool_data[cand_id]
        
        # Fast scoring with RapidFuzz
        n_ratio = fuzz.token_set_ratio(s1_name_norm, c_n)
        a_ratio = fuzz.token_set_ratio(s1_addr_norm, c_a) if s1_addr_norm and c_a else 50
        
        # Number matching bonus / penalty
        if s1_nums and c_nums:
            num_match = 100 if (s1_nums & c_nums) else 20
        else:
            num_match = 50
            
        score = 0.50 * n_ratio + 0.35 * a_ratio + 0.15 * num_match
        scored_pairs.append((s1_id, cand_id, score, n_ratio, a_ratio, num_match))

t1 = time.time()
print(f"Scored {len(scored_pairs):,} pairs in {t1-t0:.2f}s ({len(scored_pairs)/(t1-t0):.0f} pairs/s)", flush=True)

# 4. Threshold & F_0.5 Evaluation
def compute_macro_f05(preds_dict, gt_dict):
    scores = []
    for s1_id, truth in gt_dict.items():
        preds = preds_dict.get(s1_id, set())
        # Singletons (truth is empty)
        if len(truth) == 0:
            if len(preds) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
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
                    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                    scores.append(f05)
    return np.mean(scores)

# Group pairs by s1_id
pairs_by_s1 = defaultdict(list)
for item in scored_pairs:
    s1_id, cand_id, score = item[0], item[1], item[2]
    pairs_by_s1[s1_id].append((cand_id, score))

for thresh in [65, 70, 73, 75, 78, 80, 83, 85]:
    # Strategy A: Independent thresholding per pair
    preds_indep = {}
    for s1_id in s1_df['entity_id']:
        cands = pairs_by_s1.get(s1_id, [])
        matches = {c for c, sc in cands if sc >= thresh}
        preds_indep[s1_id] = matches
    f05_indep = compute_macro_f05(preds_indep, gt_dict)
    
    # Strategy B: Mutual exclusivity (best match per candidate) + threshold
    best_for_cand = {}
    for item in scored_pairs:
        s1_id, cand_id, score = item[0], item[1], item[2]
        if score >= thresh:
            if cand_id not in best_for_cand or score > best_for_cand[cand_id][1]:
                best_for_cand[cand_id] = (s1_id, score)
    
    preds_exclusive = defaultdict(set)
    for cand_id, (s1_id, sc) in best_for_cand.items():
        preds_exclusive[s1_id].add(cand_id)
        
    f05_exclusive = compute_macro_f05(preds_exclusive, gt_dict)
    print(f"Threshold {thresh:2d} | F_0.5 (Independent): {f05_indep:.4f} | F_0.5 (1-to-1 Exclusive): {f05_exclusive:.4f}", flush=True)

print("="*70, flush=True)
