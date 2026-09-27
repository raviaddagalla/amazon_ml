import sys
import duckdb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("AUDIT 0b: Match count predictability per S1 entity")
print("=" * 60)

q0b = """
WITH gt_stats AS (
    SELECT 
        source1_entity_id,
        CASE 
            WHEN matched_entity_ids IS NULL OR length(trim(matched_entity_ids)) = 0 THEN 0
            ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
        END as match_count
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='')
),
s1_stats AS (
    SELECT 
        entity_id,
        country,
        length(business_name) as name_len,
        length(business_address) as addr_len,
        business_address IS NULL OR length(trim(business_address)) = 0 as addr_missing,
        length(regexp_replace(business_address, '[^0-9]', '', 'g')) > 0 as addr_has_digits
    FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='')
)
SELECT 
    s1.country,
    s1.addr_missing,
    s1.addr_has_digits,
    CASE 
        WHEN s1.name_len < 15 THEN 'short_name (<15)'
        WHEN s1.name_len < 30 THEN 'med_name (15-30)'
        ELSE 'long_name (>30)'
    END as name_len_bin,
    CASE 
        WHEN s1.addr_len < 20 THEN 'short_addr (<20)'
        WHEN s1.addr_len < 50 THEN 'med_addr (20-50)'
        ELSE 'long_addr (>50)'
    END as addr_len_bin,
    gt.match_count
FROM s1_stats s1
JOIN gt_stats gt ON s1.entity_id = gt.source1_entity_id
LIMIT 200000;
"""
df = con.execute(q0b).fetchdf()

print(f"Loaded {len(df):,} sample records.")
print("\n--- Match Count by Country ---")
print(df.groupby('country')['match_count'].agg(['count', 'mean', 'median', lambda x: (x == 0).mean()]).rename(columns={'<lambda_0>': 'pct_singleton'}))

print("\n--- Match Count by Address Missing ---")
print(df.groupby('addr_missing')['match_count'].agg(['count', 'mean', 'median', lambda x: (x == 0).mean()]).rename(columns={'<lambda_0>': 'pct_singleton'}))

print("\n--- Match Count by Name Length Bin ---")
print(df.groupby('name_len_bin')['match_count'].agg(['count', 'mean', 'median', lambda x: (x == 0).mean()]).rename(columns={'<lambda_0>': 'pct_singleton'}))

print("\n--- Match Count Distribution (Overall) ---")
print(df['match_count'].value_counts(normalize=True).sort_index().head(12))
