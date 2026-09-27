import sys
import duckdb
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("ANALYSIS OF: student_resource/output/matching_results.tsv")
print("=" * 60)

q = """
WITH res AS (
    SELECT 
        source1_entity_id,
        matched_entity_ids,
        CASE 
            WHEN matched_entity_ids IS NULL OR length(trim(matched_entity_ids)) = 0 THEN 0
            ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
        END as match_count
    FROM read_csv('student_resource/output/matching_results.tsv', delim='\t', header=true, quote='', escape='')
),
joined AS (
    SELECT 
        r.source1_entity_id,
        r.matched_entity_ids,
        r.match_count,
        s1.business_name,
        s1.business_address,
        s1.country
    FROM res r
    JOIN read_csv('student_resource/dataset/test/test_source1.tsv', delim='\t', header=true, quote='', escape='') s1
        ON r.source1_entity_id = s1.entity_id
)
SELECT 
    country,
    count(*) as total_s1,
    sum(match_count) as total_matches,
    sum(CASE WHEN match_count = 0 THEN 1 ELSE 0 END) as singletons,
    round(100.0 * sum(CASE WHEN match_count = 0 THEN 1 ELSE 0 END) / count(*), 2) as pct_singletons,
    round(avg(match_count), 2) as avg_matches_per_s1,
    round(avg(CASE WHEN match_count > 0 THEN match_count ELSE NULL END), 2) as avg_matches_non_singleton
FROM joined
GROUP BY country
ORDER BY country;
"""
df_country = con.execute(q).fetchdf()
print(df_country)

q_dist = """
WITH res AS (
    SELECT 
        CASE 
            WHEN matched_entity_ids IS NULL OR length(trim(matched_entity_ids)) = 0 THEN 0
            ELSE length(matched_entity_ids) - length(replace(matched_entity_ids, ',', '')) + 1
        END as match_count
    FROM read_csv('student_resource/output/matching_results.tsv', delim='\t', header=true, quote='', escape='')
)
SELECT 
    match_count,
    count(*) as count,
    round(100.0 * count(*) / (SELECT count(*) FROM res), 2) as pct
FROM res
GROUP BY match_count
ORDER BY match_count
LIMIT 12;
"""
df_dist = con.execute(q_dist).fetchdf()
print("\n--- MATCH COUNT DISTRIBUTION ---")
print(df_dist)
