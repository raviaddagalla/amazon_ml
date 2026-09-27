import sys
import duckdb
import pandas as pd
import numpy as np
import re

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("AUDIT 0c: Exact & Near-Duplicate Analysis of True Pairs")
print("=" * 60)

# Extract 5000 true pairs from train
q0c = """
WITH gt_sample AS (
    SELECT 
        source1_entity_id,
        unnest(string_split(matched_entity_ids, ',')) as match_id
    FROM read_csv('student_resource/dataset/train/train_ground_truth.tsv', delim='\t', header=true, quote='', escape='')
    WHERE matched_entity_ids IS NOT NULL AND length(trim(matched_entity_ids)) > 0
    LIMIT 10000
),
pairs AS (
    SELECT 
        gt.source1_entity_id,
        gt.match_id,
        s1.business_name as s1_name,
        s1.business_address as s1_addr,
        COALESCE(s2.business_name, s3.business_name) as match_name,
        COALESCE(s2.business_address, s3.business_address) as match_addr,
        s1.country
    FROM gt_sample gt
    JOIN read_csv('student_resource/dataset/train/train_source1.tsv', delim='\t', header=true, quote='', escape='') s1 
        ON gt.source1_entity_id = s1.entity_id
    LEFT JOIN read_csv('student_resource/dataset/train/train_source2.tsv', delim='\t', header=true, quote='', escape='') s2 
        ON gt.match_id = s2.entity_id
    LEFT JOIN read_csv('student_resource/dataset/train/train_source3.tsv', delim='\t', header=true, quote='', escape='') s3 
        ON gt.match_id = s3.entity_id
    WHERE (s2.business_name IS NOT NULL OR s3.business_name IS NOT NULL)
    LIMIT 5000
)
SELECT * FROM pairs;
"""
df = con.execute(q0c).fetchdf()
print(f"Sampled {len(df):,} true positive pairs.")

def clean_str(s):
    if not s or pd.isna(s):
        return ""
    # Lowercase and keep alphanumeric and spaces
    s = s.lower()
    # Strip S3 ID suffixes e.g. (ID: 12345)
    s = re.sub(r'\(id:\s*\d+\)', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return " ".join(s.split())

def sort_tokens(s):
    return " ".join(sorted(clean_str(s).split()))

# Compute exact matches under various normalization levels
exact_raw_name = (df['s1_name'] == df['match_name']).mean()
exact_raw_addr = (df['s1_addr'] == df['match_addr']).mean()
exact_raw_both = ((df['s1_name'] == df['match_name']) & (df['s1_addr'] == df['match_addr'])).mean()

df['s1_name_clean'] = df['s1_name'].apply(clean_str)
df['match_name_clean'] = df['match_name'].apply(clean_str)
df['s1_addr_clean'] = df['s1_addr'].apply(clean_str)
df['match_addr_clean'] = df['match_addr'].apply(clean_str)

exact_clean_name = (df['s1_name_clean'] == df['match_name_clean']).mean()
exact_clean_addr = (df['s1_addr_clean'] == df['match_addr_clean']).mean()
exact_clean_both = ((df['s1_name_clean'] == df['match_name_clean']) & (df['s1_addr_clean'] == df['match_addr_clean'])).mean()

df['s1_name_sorted'] = df['s1_name'].apply(sort_tokens)
df['match_name_sorted'] = df['match_name'].apply(sort_tokens)
df['s1_addr_sorted'] = df['s1_addr'].apply(sort_tokens)
df['match_addr_sorted'] = df['match_addr'].apply(sort_tokens)

exact_sorted_name = (df['s1_name_sorted'] == df['match_name_sorted']).mean()
exact_sorted_addr = (df['s1_addr_sorted'] == df['match_addr_sorted']).mean()
exact_sorted_both = ((df['s1_name_sorted'] == df['match_name_sorted']) & (df['s1_addr_sorted'] == df['match_addr_sorted'])).mean()

# Check numbers in address (postal code / street numbers)
def extract_digits(s):
    if not s or pd.isna(s):
        return []
    return re.findall(r'\b\d+\b', str(s))

def digit_overlap(row):
    d1 = set(extract_digits(row['s1_addr']))
    d2 = set(extract_digits(row['match_addr']))
    if not d1 and not d2:
        return True # neither has digits
    return len(d1 & d2) > 0

df['addr_digits_overlap'] = df.apply(digit_overlap, axis=1)

print("\n--- EXACT MATCH FRACTIONS ON TRUE POSITIVE PAIRS ---")
print(f"1. Raw strings identical:")
print(f"   Name:    {exact_raw_name*100:.2f}%")
print(f"   Address: {exact_raw_addr*100:.2f}%")
print(f"   Both:    {exact_raw_both*100:.2f}%")

print(f"\n2. Cleaned (lowercased + stripped non-alphanumeric + ID suffix stripped):")
print(f"   Name:    {exact_clean_name*100:.2f}%")
print(f"   Address: {exact_clean_addr*100:.2f}%")
print(f"   Both:    {exact_clean_both*100:.2f}%")

print(f"\n3. Token-Sorted Cleaned:")
print(f"   Name:    {exact_sorted_name*100:.2f}%")
print(f"   Address: {exact_sorted_addr*100:.2f}%")
print(f"   Both:    {exact_sorted_both*100:.2f}%")

print(f"\n4. Address Number Overlap (Street / Pincode):")
print(f"   Any number match: {df['addr_digits_overlap'].mean()*100:.2f}%")

print("\n--- SAMPLE MISMATCHED PAIRS (What does the residual look like?) ---")
mismatches = df[df['s1_name_sorted'] != df['match_name_sorted']]
for _, r in mismatches.head(10).iterrows():
    print("S1:   ", r['s1_name'], "|", r['s1_addr'])
    print("MATCH:", r['match_name'], "|", r['match_addr'])
    print("-" * 50)
