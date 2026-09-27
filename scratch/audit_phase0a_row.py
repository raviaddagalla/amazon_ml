import sys
import duckdb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("AUDIT 0a (Part 2): Row positions correlation in source files")
print("=" * 60)

q = """
CREATE TEMP TABLE s1_idx AS
SELECT entity_id, row_number() OVER () as s1_row
FROM read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='')
LIMIT 100000;

CREATE TEMP TABLE s2_idx AS
SELECT entity_id, row_number() OVER () as s2_row
FROM read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='')
LIMIT 200000;

CREATE TEMP TABLE s3_idx AS
SELECT entity_id, row_number() OVER () as s3_row
FROM read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='')
LIMIT 200000;

WITH gt AS (
    SELECT 
        source1_entity_id,
        unnest(string_split(matched_entity_ids, ',')) as match_id
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL
    LIMIT 10000
)
SELECT 
    gt.source1_entity_id,
    gt.match_id,
    s1.s1_row,
    s2.s2_row,
    s3.s3_row
FROM gt
JOIN s1_idx s1 ON gt.source1_entity_id = s1.entity_id
LEFT JOIN s2_idx s2 ON gt.match_id = s2.entity_id
LEFT JOIN s3_idx s3 ON gt.match_id = s3.entity_id
WHERE s2.s2_row IS NOT NULL OR s3.s3_row IS NOT NULL
LIMIT 5000;
"""
df = con.execute(q).fetchdf()
print(f"Matched {len(df)} sample rows with row indices.")
if len(df) > 0:
    df['match_row'] = df['s2_row'].combine_first(df['s3_row'])
    corr = df['s1_row'].corr(df['match_row'])
    print(f"Correlation between S1 row index and S2/S3 row index: {corr:.4f}")
    print("Sample row comparisons:")
    for _, r in df.head(10).iterrows():
        print(f"  S1 row: {r['s1_row']} -> Match row: {r['match_row']}")
