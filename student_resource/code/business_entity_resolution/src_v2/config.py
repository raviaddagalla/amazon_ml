"""
Configuration for Tiered Deterministic-First Entity Resolution Pipeline (v2)
Country-agnostic, open-set design for US, India, France, and unseen countries.
"""

import os
import re

# ==============================================================================
# 1. PATH RESOLUTION
# ==============================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
STUDENT_RESOURCE_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(STUDENT_RESOURCE_ROOT, "output")
TEMP_PARTITIONS_DIR = os.path.join(OUTPUT_DIR, "temp_partitions_v2")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_PARTITIONS_DIR, exist_ok=True)

# Dataset paths
TRAIN_DIR = os.path.join(STUDENT_RESOURCE_ROOT, "dataset", "train")
TEST_DIR = os.path.join(STUDENT_RESOURCE_ROOT, "dataset", "test")

TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
RESIDUAL_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_residual_v2.txt")

# ==============================================================================
# 2. CANONICALIZATION & ABBREVIATION DICTIONARIES
# ==============================================================================

# S3 ID suffix pattern e.g., "(ID: 12345)"
S3_ID_SUFFIX_RE = re.compile(r'\(id:\s*\d+\)', re.IGNORECASE)

# Domain extension pattern e.g., ".com", ".in", ".org", ".fr", "www."
DOMAIN_CLEAN_RE = re.compile(r'\b(www\.)|\.(com|in|org|net|co|io|biz|info|fr|gov|edu|mil)\b', re.IGNORECASE)

# Comprehensive corporate/legal suffixes across US, India, France, and international
CORPORATE_LEGAL_SUFFIXES = {
    # India / Commonwealth
    'private', 'limited', 'pvt', 'ltd', 'llp', 'opc', 'proprietorship',
    # US / UK / International
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'co', 'company',
    'enterprises', 'enterprise', 'services', 'service', 'holdings', 'holding',
    'group', 'industries', 'industry', 'international', 'intl', 'plc', 'lp',
    'trust', 'foundation', 'solutions', 'associates', 'consultants',
    # France / European
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'gmbh', 'bv', 'nv',
    'ets', 'cie', 'cabinet', 'ste', 'societe',
}

# Bidirectional normalization map for street, locality, business terms
ABBREVIATION_MAP = {
    # US & General Address Types
    'st': 'street', 'str': 'street', 'rd': 'road', 'dr': 'drive', 'ave': 'avenue',
    'blvd': 'boulevard', 'ln': 'lane', 'ct': 'court', 'pl': 'place', 'hwy': 'highway',
    'ste': 'suite', 'apt': 'apartment', 'fl': 'floor', 'bldg': 'building',
    'pk': 'park', 'pkwy': 'parkway', 'cir': 'circle', 'ter': 'terrace',
    'way': 'way', 'sq': 'square', 'cl': 'close', 'cres': 'crescent',
    'aly': 'alley', 'row': 'row', 'plz': 'plaza', 'xing': 'crossing',
    
    # Directionals
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    
    # French Address Types & Abbreviations
    'r': 'rue', 'av': 'avenue', 'bd': 'boulevard', 'all': 'allee', 'imp': 'impasse',
    'che': 'chemin', 'pl': 'place', 'pas': 'passage', 'rte': 'route', 'crs': 'cours',
    'res': 'residence', 'bat': 'batiment', 'esc': 'escalier', 'etg': 'etage',
    'cedex': 'cedex', 'bp': 'bp',
    
    # Business & Industry Terms
    'mgmt': 'management', 'tech': 'technology', 'technologies': 'technology',
    'mfg': 'manufacturing', 'dist': 'distribution', 'eng': 'engineering',
    'engr': 'engineering', 'assoc': 'associates', 'soln': 'solutions',
    'solns': 'solutions', 'sys': 'systems', 'system': 'systems',
    'dev': 'development', 'comm': 'communications', 'med': 'medical',
    'ctr': 'center', 'centre': 'center', 'hlth': 'health', 'pharma': 'pharmaceuticals',
    'fin': 'financial', 'serv': 'services', 'svcs': 'services', 'intl': 'international',
    'natl': 'national', 'global': 'global', 'mktg': 'marketing', 'auto': 'automotive',
    
    # Indian State Abbreviations
    'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam',
    'br': 'bihar', 'cg': 'chhattisgarh', 'ga': 'goa', 'gj': 'gujarat',
    'hr': 'haryana', 'hp': 'himachal pradesh', 'jh': 'jharkhand',
    'ka': 'karnataka', 'kl': 'kerala', 'mp': 'madhya pradesh',
    'mh': 'maharashtra', 'mn': 'manipur', 'ml': 'meghalaya',
    'mz': 'mizoram', 'nl': 'nagaland', 'od': 'odisha', 'pb': 'punjab',
    'rj': 'rajasthan', 'sk': 'sikkim', 'tn': 'tamil nadu',
    'ts': 'telangana', 'tg': 'telangana', 'tr': 'tripura', 'up': 'uttar pradesh',
    'uk': 'uttarakhand', 'ua': 'uttarakhand', 'wb': 'west bengal',
    'dl': 'delhi', 'jk': 'jammu and kashmir', 'la': 'ladakh',
    'py': 'puducherry', 'ch': 'chandigarh',
    
    # US State Abbreviations
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# ==============================================================================
# 3. CASCADE TIER PARAMETERS
# ==============================================================================
TIER1_SCORE = 1.00          # Exact canonical name match + address number overlap
TIER2_SCORE = 0.95          # High-confidence fuzzy name + address number exact match
TIER2_NAME_SIM_BAR = 0.85   # Minimum token set / Jaro-Winkler on core name for Tier 2

TIER3_PROB_THRESHOLD = 0.85 # LightGBM residual classifier probability threshold
TIER4_MARGIN_GUARD = 0.05   # Minimum probability margin over runner-up to accept Tier 3 match

TOP_CANDIDATES_PER_S1 = 50   # Candidate retrieval cap (expanded for higher recall)
MAX_BLOCKING_KEY_SIZE = 500  # Skip overly generic keys
BATCH_SIZE = 1000            # Inference batch size for memory efficiency
