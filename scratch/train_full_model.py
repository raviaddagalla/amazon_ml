import sys, os, time, re, anyascii
import duckdb, pandas as pd, numpy as np
from collections import defaultdict, Counter
from rapidfuzz import fuzz
import lightgbm as lgb

sys.stdout.reconfigure(encoding='utf-8')

print("="*70, flush=True)
print("TRAINING LIGHTGBM MODEL ON 25,000 DIVERSE TRAINING SAMPLES", flush=True)
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

# Load 25,000 S1 records (20k Train, 5k Val)
con = duckdb.connect()
print("Loading 25,000 S1 training records from train_source1.tsv...", flush=True)
s1_df = con.execute("""
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 25000;
""").df()

s1_train = s1_df.iloc[:20000].copy()
s1_val = s1_df.iloc[20000:].copy()

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

print(f"Total S1: {len(s1_df)}, Targets: {len(all_target_ids):,}", flush=True)

# Load candidate pool
con.register('targets', pd.DataFrame({'mid': list(all_target_ids)}))
s2 = con.execute("""
    SELECT * FROM (
        SELECT s2.* FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') s2
        SEMI JOIN targets ON entity_id = mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') LIMIT 80000
    )
""").df().drop_duplicates(subset=['entity_id'])

s3 = con.execute("""
    SELECT * FROM (
        SELECT s3.* FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') s3
        SEMI JOIN targets ON entity_id = mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') LIMIT 80000
    )
""").df().drop_duplicates(subset=['entity_id'])

pool = pd.concat([s2, s3], ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"Pool size: {len(pool):,} records. True targets present: {pool['entity_id'].isin(all_target_ids).sum()} / {len(all_target_ids)}", flush=True)

# Pre-normalize pool
pool_data = {}
inv_index = defaultdict(list)
key_counts = Counter()

t0 = time.time()
for row in pool.itertuples(index=False):
    eid = row.entity_id
    n_norm = norm_str(row.business_name)
    a_norm = norm_str(row.business_address)
    nums = extract_numbers(row.business_address)
    n_words = set(clean_tokens(row.business_name))
    a_words = set(clean_tokens(row.business_address))
    first_word = clean_tokens(row.business_name)[0] if clean_tokens(row.business_name) else ''
    country = row.country
    pool_data[eid] = (n_norm, a_norm, nums, n_words, a_words, first_word)
    
    keys = extract_blocking_keys(row.business_name, row.business_address)
    for k in keys:
        inv_index[(country, k)].append(eid)
        key_counts[(country, k)] += 1

print(f"Pool indexed in {time.time()-t0:.2f}s", flush=True)

def featurize_candidates(s1_subset, top_cands_k=25):
    X = []
    y = []
    meta = []
    
    for row in s1_subset.itertuples(index=False):
        s1_id = row.entity_id
        s1_name_norm = norm_str(row.business_name)
        s1_addr_norm = norm_str(row.business_address)
        s1_nums = extract_numbers(row.business_address)
        s1_n_words = set(clean_tokens(row.business_name))
        s1_a_words = set(clean_tokens(row.business_address))
        s1_first_word = clean_tokens(row.business_name)[0] if clean_tokens(row.business_name) else ''
        country = row.country
        
        true_set = gt_dict.get(s1_id, set())
        
        s1_keys = extract_blocking_keys(row.business_name, row.business_address)
        cand_weights = Counter()
        for k in s1_keys:
            ck = (country, k)
            df = key_counts.get(ck, 0)
            if df == 0 or df > 500:
                continue
            w = 5.0 if k.startswith('n1:') or k.startswith('nb:') else (3.0 if k.startswith('pin:') else (2.0 if k.startswith('na:') else 1.0))
            for cand_id in inv_index.get(ck, ()):
                cand_weights[cand_id] += w
                
        if not cand_weights:
            continue
            
        top_cands = cand_weights.most_common(top_cands_k)
        for cand_id, blk_score in top_cands:
            c_n, c_a, c_nums, c_nw, c_aw, c_first = pool_data[cand_id]
            
            n_set = fuzz.token_set_ratio(s1_name_norm, c_n)
            n_sort = fuzz.token_sort_ratio(s1_name_norm, c_n)
            n_part = fuzz.partial_ratio(s1_name_norm, c_n)
            n_ratio = fuzz.ratio(s1_name_norm, c_n)
            
            a_set = fuzz.token_set_ratio(s1_addr_norm, c_a) if s1_addr_norm and c_a else 50
            a_part = fuzz.partial_ratio(s1_addr_norm, c_a) if s1_addr_norm and c_a else 50
            
            first_match = 1.0 if s1_first_word and c_first and s1_first_word == c_first else 0.0
            
            n_jacc = len(s1_n_words & c_nw) / max(1, len(s1_n_words | c_nw))
            a_jacc = len(s1_a_words & c_aw) / max(1, len(s1_a_words | c_aw))
            
            num_overlap = len(s1_nums & c_nums)
            has_num_overlap = 1.0 if num_overlap > 0 else 0.0
            num_mismatch = 1.0 if (s1_nums and c_nums and num_overlap == 0) else 0.0
            
            is_match = 1 if cand_id in true_set else 0
            
            feat = [
                n_set, n_sort, n_part, n_ratio, first_match, n_jacc,
                a_set, a_part, a_jacc,
                num_overlap, has_num_overlap, num_mismatch,
                blk_score
            ]
            X.append(feat)
            y.append(is_match)
            meta.append((s1_id, cand_id))
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), meta

print("Extracting features for training set (20,000 S1)...", flush=True)
t0 = time.time()
X_train, y_train, meta_train = featurize_candidates(s1_train, top_cands_k=25)
print(f"Train data: {len(X_train):,} pairs (Positives: {y_train.sum():,}) in {time.time()-t0:.2f}s", flush=True)

print("Extracting features for validation set (5,000 S1)...", flush=True)
t0 = time.time()
X_val, y_val, meta_val = featurize_candidates(s1_val, top_cands_k=25)
print(f"Val data: {len(X_val):,} pairs (Positives: {y_val.sum():,}) in {time.time()-t0:.2f}s", flush=True)

# Train LightGBM model
print("\nTraining LightGBM model...", flush=True)
t0 = time.time()
clf = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.07,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4,
    verbosity=-1
)
clf.fit(X_train, y_train)
print(f"LightGBM trained in {time.time()-t0:.2f}s", flush=True)

# Save model
os.makedirs('scratch/models', exist_ok=True)
clf.booster_.save_model('scratch/models/lgbm_entity_resolver.txt')
print("Model saved to scratch/models/lgbm_entity_resolver.txt", flush=True)

# Predict probabilities on Validation set
val_probs = clf.predict_proba(X_val)[:, 1]

# Evaluate F_0.5 at different probability thresholds with 1-to-1 matching
val_gt_dict = {row.entity_id: gt_dict[row.entity_id] for row in s1_val.itertuples(index=False)}

best_thresh = 0.50
best_f05 = 0.0

for prob_thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    best_for_cand = {}
    for idx, (s1_id, cand_id) in enumerate(meta_val):
        prob = val_probs[idx]
        if prob >= prob_thresh:
            if cand_id not in best_for_cand or prob > best_for_cand[cand_id][1]:
                best_for_cand[cand_id] = (s1_id, prob)
                
    preds_exclusive = defaultdict(set)
    for cand_id, (s1_id, p) in best_for_cand.items():
        preds_exclusive[s1_id].add(cand_id)
        
    scores = []
    for s1_id, truth in val_gt_dict.items():
        preds = preds_exclusive.get(s1_id, set())
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
    f05 = np.mean(scores)
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = prob_thresh
    print(f"Prob Threshold {prob_thresh:.2f} | Validation F_0.5 = {f05:.4f}", flush=True)

print(f"\nOPTIMAL DECISION THRESHOLD: {best_thresh:.2f} with Validation F_0.5 = {best_f05:.4f}", flush=True)
print("="*70, flush=True)
