import sys, re, anyascii, math, time
import duckdb
import pandas as pd
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

# DuckDB quick load
con = duckdb.connect()
s1_df = con.execute('''
    SELECT entity_id, business_name, business_address, country 
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='') 
    LIMIT 2000;
''').df()

gt_df = con.execute('''
    SELECT source1_entity_id, matched_entity_ids 
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='') 
    WHERE source1_entity_id IN (SELECT entity_id FROM s1_df);
''').df()

target_ids = set()
gt_dict = {}
for _, r in gt_df.iterrows():
    if pd.notna(r['matched_entity_ids']) and r['matched_entity_ids']:
        m = set(r['matched_entity_ids'].split(','))
        gt_dict[r['source1_entity_id']] = m
        target_ids.update(m)

# Load 50k S2 and 50k S3 distractors plus targets
con.register('target_table', pd.DataFrame({'mid': list(target_ids)}))
s2 = con.execute('''
    SELECT * FROM (
        SELECT s2.* FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') s2
        SEMI JOIN target_table t ON s2.entity_id = t.mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    )
''').df().drop_duplicates(subset=['entity_id'])

s3 = con.execute('''
    SELECT * FROM (
        SELECT s3.* FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') s3
        SEMI JOIN target_table t ON s3.entity_id = t.mid
        UNION ALL
        SELECT * FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') LIMIT 50000
    )
''').df().drop_duplicates(subset=['entity_id'])

pool = pd.concat([s2, s3], ignore_index=True)
present_cnt = pool['entity_id'].isin(target_ids).sum()
print(f'S1: {len(s1_df)}, Pool: {len(pool):,}, Targets present: {present_cnt} / {len(target_ids)}')

STOPWORDS = {
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'holdings', 'holding', 'sarl', 'sasu', 'sas', 'sci', 'llp', 'pllc', 'gmbh',
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'floor', 'fl', 'unit', 'suite', 'ste', 'near', 'opp', 'behind', 'block',
    'sector', 'plot', 'no', 'building', 'bldg', 'tower', 'city', 'state',
    'rue', 'boulevard', 'blvd', 'allee', 'des', 'du', 'de', 'la', 'le', 'east', 'west', 'north', 'south'
}

def extract_keys(name, addr):
    keys = set()
    if isinstance(name, str) and name:
        clean_n = re.sub(r'[^a-z0-9\s]', ' ', anyascii.anyascii(name).lower()).split()
        for w in clean_n:
            if len(w) >= 3 and w not in STOPWORDS:
                keys.add('n_' + w)
        if len(clean_n) >= 2:
            if clean_n[0] not in STOPWORDS and clean_n[1] not in STOPWORDS:
                keys.add(f'nb_{clean_n[0]}_{clean_n[1]}')
    if isinstance(addr, str) and addr:
        clean_a = re.sub(r'[^a-z0-9\s]', ' ', anyascii.anyascii(addr).lower())
        for num in re.findall(r'\b\d{2,6}\b', clean_a):
            keys.add('num_' + num)
        for w in clean_a.split():
            if len(w) >= 4 and w not in STOPWORDS:
                keys.add('a_' + w)
    return keys

# Build index
t0 = time.time()
inv_idx = defaultdict(list)
doc_freq = Counter()
pool_eids = pool['entity_id'].values

for idx, row in enumerate(pool.itertuples(index=False)):
    country = row.country
    keys = extract_keys(row.business_name, row.business_address)
    for k in keys:
        inv_idx[(country, k)].append(idx)
        doc_freq[(country, k)] += 1

print(f'Built index in {time.time()-t0:.2f}s. Total keys: {len(inv_idx):,}')

# Evaluate recall with MAX_DF cap
MAX_DF = 500
t0 = time.time()
total_true = 0
hits_10 = 0
hits_25 = 0

for row in s1_df.itertuples(index=False):
    s1_id = row.entity_id
    true_targets = gt_dict.get(s1_id, set()) & set(pool['entity_id'])
    if not true_targets: continue
    total_true += len(true_targets)
    
    country = row.country
    keys = extract_keys(row.business_name, row.business_address)
    
    # Candidate scoring: IDF weighted
    cand_scores = defaultdict(float)
    for k in keys:
        ck = (country, k)
        df = doc_freq.get(ck, 0)
        if df == 0 or df > MAX_DF: continue
        idf = math.log((len(pool) + 1) / (df + 1))
        weight = 3.0 if k.startswith('n_') or k.startswith('nb_') else (1.5 if k.startswith('num_') else 1.0)
        score_increment = idf * weight
        for p_idx in inv_idx[ck]:
            cand_scores[p_idx] += score_increment
            
    if cand_scores:
        top_candidates = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:25]
        top_eids_25 = {pool_eids[x[0]] for x in top_candidates}
        top_eids_10 = {pool_eids[x[0]] for x in top_candidates[:10]}
        hits_25 += len(true_targets & top_eids_25)
        hits_10 += len(true_targets & top_eids_10)

t1 = time.time()
print(f'Evaluation of {len(s1_df)} queries completed in {t1-t0:.2f}s ({len(s1_df)/(t1-t0):.0f} q/s)!')
print(f'Total true targets: {total_true}')
print(f'Recall @ 10: {hits_10} / {total_true} ({hits_10/total_true*100:.2f}%)')
print(f'Recall @ 25: {hits_25} / {total_true} ({hits_25/total_true*100:.2f}%)')
