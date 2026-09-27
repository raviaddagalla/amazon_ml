import sys
import duckdb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("AUDIT 0a: Entity ID numeric suffixes & row positions")
print("=" * 60)

# Sample 50,000 ground truth rows
q0a = """
WITH gt_sample AS (
    SELECT 
        source1_entity_id, 
        matched_entity_ids,
        row_number() OVER () as s1_row_idx
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND length(matched_entity_ids) > 0
    LIMIT 10000
),
unpacked AS (
    SELECT 
        source1_entity_id,
        s1_row_idx,
        unnest(string_split(matched_entity_ids, ',')) as match_id
    FROM gt_sample
)
SELECT 
    source1_entity_id,
    s1_row_idx,
    match_id,
    TRY_CAST(replace(source1_entity_id, 'S1-', '') AS BIGINT) as s1_num,
    TRY_CAST(replace(replace(match_id, 'S2-', ''), 'S3-', '') AS BIGINT) as match_num
FROM unpacked
LIMIT 10000;
"""
df_0a = con.execute(q0a).fetchdf()
df_0a['num_diff'] = (df_0a['s1_num'] - df_0a['match_num']).abs()
df_0a['num_ratio'] = df_0a['match_num'] / np.maximum(df_0a['s1_num'], 1)

print(f"Sampled {len(df_0a)} pairs from 10,000 S1 entities.")
print(f"S1 ID range: [{df_0a['s1_num'].min()}, {df_0a['s1_num'].max()}]")
print(f"Match ID range: [{df_0a['match_num'].min()}, {df_0a['match_num'].max()}]")
print(f"Correlation between S1 numeric ID and Match numeric ID: {df_0a['s1_num'].corr(df_0a['match_num']):.4f}")
print("Numeric ID difference percentiles:")
print(df_0a['num_diff'].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))

print("\nSample pairs with IDs:")
for _, r in df_0a.head(10).iterrows():
    print(f"  {r['source1_entity_id']} -> {r['match_id']} | s1_num={r['s1_num']}, match_num={r['match_num']}, diff={r['num_diff']}")
