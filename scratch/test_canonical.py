import sys
import duckdb
import pandas as pd
import numpy as np
import re
import anyascii

sys.stdout.reconfigure(encoding='utf-8')
con = duckdb.connect()

print("=" * 60)
print("AUDIT: Measuring Canonicalization Potential on True Pairs")
print("=" * 60)

q = """
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
df = con.execute(q).fetchdf()

# Aggressive corporate suffixes to strip completely
CORP_SUFFIXES = {
    'private', 'limited', 'pvt', 'ltd', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'enterprises', 'enterprise', 'services', 'service',
    'holdings', 'holding', 'group', 'industries', 'industry', 'international', 'intl',
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'gmbh', 'bv', 'nv', 'plc'
}

ABBR_MAP = {
    'st': 'street', 'str': 'street', 'rd': 'road', 'dr': 'drive', 'ave': 'avenue',
    'blvd': 'boulevard', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'hwy': 'highway',
    'ste': 'suite', 'apt': 'apartment', 'fl': 'floor', 'bldg': 'building',
    'pk': 'park', 'pkwy': 'parkway', 'cir': 'circle', 'ter': 'terrace',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    'mgmt': 'management', 'tech': 'technology', 'mfg': 'manufacturing',
    'dist': 'distribution', 'eng': 'engineering', 'assoc': 'associates',
    'soln': 'solutions', 'solns': 'solutions', 'sys': 'systems'
}

def canonicalize_name(name):
    if not name or pd.isna(name):
        return ""
    # Transliterate Unicode (Telugu, Hindi, French diacritics -> ASCII)
    s = anyascii.anyascii(str(name)).lower()
    # Strip ID suffixes like (ID: 12345)
    s = re.sub(r'\(id:\s*\d+\)', '', s)
    # Strip domain extensions (.com, .in, etc.)
    s = re.sub(r'\.(com|in|org|net|co|io|biz|info)\b', '', s)
    # Remove all non-alphanumeric
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # Expand abbreviations
    tokens = [ABBR_MAP.get(w, w) for w in s.split()]
    # Strip corporate suffixes
    tokens = [w for w in tokens if w not in CORP_SUFFIXES]
    # Return sorted tokens signature
    return " ".join(sorted(tokens))

def canonicalize_address(addr):
    if not addr or pd.isna(addr):
        return ""
    s = anyascii.anyascii(str(addr)).lower()
    s = re.sub(r'\.(com|in|org|net|co|io|biz|info)\b', '', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    tokens = [ABBR_MAP.get(w, w) for w in s.split()]
    # Extract numbers
    nums = sorted([re.sub(r'^0+', '', w) for w in tokens if w.isdigit() and len(w) <= 6])
    # Extract non-number tokens
    words = sorted([w for w in tokens if not w.isdigit()])
    return " ".join(nums) + " | " + " ".join(words)

df['s1_canon_name'] = df['s1_name'].apply(canonicalize_name)
df['match_canon_name'] = df['match_name'].apply(canonicalize_name)
df['s1_canon_addr'] = df['s1_addr'].apply(canonicalize_address)
df['match_canon_addr'] = df['match_addr'].apply(canonicalize_address)

canon_name_match = (df['s1_canon_name'] == df['match_canon_name']).mean()
canon_addr_match = (df['s1_canon_addr'] == df['match_canon_addr']).mean()
canon_both_match = ((df['s1_canon_name'] == df['match_canon_name']) & (df['s1_canon_addr'] == df['match_canon_addr'])).mean()

print(f"Canonical Name Exact Match:    {canon_name_match*100:.2f}% (up from 4.88% raw / 21.18% clean)")
print(f"Canonical Address Exact Match: {canon_addr_match*100:.2f}% (up from 0.22% raw / 12.16% clean)")
print(f"Both Exactly Match:            {canon_both_match*100:.2f}%")

# What if name matches and at least one address number matches?
def check_name_plus_number(row):
    if not row['s1_canon_name'] or not row['match_canon_name']:
        return False
    if row['s1_canon_name'] == row['match_canon_name']:
        # Check if numbers overlap or address is missing
        nums1 = set(re.findall(r'\b\d+\b', str(row['s1_addr']) if pd.notna(row['s1_addr']) else ''))
        nums2 = set(re.findall(r'\b\d+\b', str(row['match_addr']) if pd.notna(row['match_addr']) else ''))
        if not nums1 and not nums2:
            return True
        return len(nums1 & nums2) > 0
    return False

df['name_match_and_number'] = df.apply(check_name_plus_number, axis=1)
print(f"Canonical Name Match + Number Overlap: {df['name_match_and_number'].mean()*100:.2f}%")
