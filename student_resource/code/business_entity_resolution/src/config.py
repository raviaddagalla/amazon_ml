"""
Configuration and Hyperparameters for Business Entity Resolution Pipeline
(Phase 2 upgrade: abbreviation dictionary, extended stopwords, French support)
"""

import os
import re

# Base Directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RESOURCE_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))

DATA_DIR = os.path.join(RESOURCE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(RESOURCE_DIR, "output")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Training & Inference File Paths
TRAIN_S1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3 = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_entity_resolver.txt")

# Blocking Hyperparameters
MAX_BLOCKING_KEY_SIZE = 500       # Maximum records allowed per blocking key
TOP_CANDIDATES_PER_S1 = 30        # Candidates generated per S1 entity for ML ranking
CANDIDATE_FILE_LIMIT = 30         # Max candidates to output in candidate_pairs.tsv

# Matching Model Hyperparameters
PROBABILITY_THRESHOLD = 0.84      # Calibrated via Phase 5 validation sweep (Macro F0.5 = 0.6862)

# ============================================================================
# ABBREVIATION EXPANSION DICTIONARY (Phase 2)
# Bidirectional mapping: all variants normalize to the canonical (first) form.
# Applied before tokenization to both name and address fields.
# ============================================================================
ABBREVIATION_GROUPS = [
    # --- US Street/Address ---
    ('road', 'rd'),
    ('street', 'st'),
    ('avenue', 'ave', 'av'),
    ('boulevard', 'blvd'),
    ('lane', 'ln'),
    ('drive', 'dr'),
    ('court', 'ct'),
    ('place', 'pl'),
    ('circle', 'cir'),
    ('highway', 'hwy'),
    ('parkway', 'pkwy', 'pky'),
    ('terrace', 'ter', 'terr'),
    ('apartment', 'apt'),
    ('suite', 'ste'),
    ('building', 'bldg'),
    ('floor', 'fl', 'flr'),
    ('north', 'n'),
    ('south', 's'),
    ('east', 'e'),
    ('west', 'w'),
    ('northeast', 'ne'),
    ('northwest', 'nw'),
    ('southeast', 'se'),
    ('southwest', 'sw'),
    ('mount', 'mt'),
    ('fort', 'ft'),
    ('saint', 'st'),  # note: overloaded with street, handled by context
    ('center', 'ctr'),
    ('square', 'sq'),
    
    # --- French Address ---
    ('rue', ),
    ('avenue', 'av'),
    ('boulevard', 'blvd', 'bd'),
    ('allee', 'all'),
    ('impasse', 'imp'),
    ('chemin', 'ch'),
    ('passage', 'pass'),
    ('voie', ),
    ('route', 'rte'),
    ('place', 'pl'),
    ('quai', ),
    
    # --- Corporate / Legal Suffixes (US) ---
    ('corporation', 'corp'),
    ('company', 'co'),
    ('incorporated', 'inc'),
    ('limited', 'ltd'),
    ('international', 'intl'),
    ('manufacturing', 'mfg'),
    ('association', 'assn', 'assoc'),
    ('department', 'dept'),
    ('national', 'natl'),
    ('general', 'gen'),
    ('management', 'mgmt'),
    ('technology', 'tech'),
    ('technologies', 'tech'),
    ('industries', 'ind'),
    ('solutions', 'sol', 'soln'),
    ('systems', 'sys'),
    ('enterprises', 'ent'),
    ('foundation', 'fdn', 'found'),
    ('university', 'univ'),
    ('institute', 'inst'),
    ('hospital', 'hosp'),
    
    # --- Indian Corporate / Legal Suffixes ---
    ('private', 'pvt'),
    ('limited', 'ltd'),
    
    # --- French Corporate / Legal Suffixes ---
    # SARL, SAS, SASU, EURL, SCI are already standalone - treated as stopwords
]

# Build the expansion dictionary: variant -> canonical
ABBREVIATION_MAP: dict[str, str] = {}
for group in ABBREVIATION_GROUPS:
    canonical = group[0]
    for variant in group:
        if variant not in ABBREVIATION_MAP:
            ABBREVIATION_MAP[variant] = canonical

# ============================================================================
# CORPORATE STOPWORDS (Extended with French terms)
# ============================================================================
CORPORATE_STOPWORDS = {
    # English legal / corporate
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'holdings', 'holding', 'llp', 'pllc', 'lp', 'plc',
    
    # French legal / corporate
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 'snc', 'gie',
    
    # German (for completeness)
    'gmbh', 'ag', 'ohg', 'kg',
    
    # General function words
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    
    # French function words
    'des', 'du', 'de', 'la', 'le', 'les', 'et', 'au', 'aux',
    
    # US Address types
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'floor', 'fl', 'unit', 'suite', 'ste', 'near', 'opp', 'behind', 'block',
    'sector', 'plot', 'no', 'building', 'bldg', 'tower',
    'court', 'ct', 'place', 'pl', 'circle', 'cir', 'highway', 'hwy',
    'parkway', 'pkwy', 'terrace', 'ter',
    
    # French address types
    'rue', 'boulevard', 'blvd', 'allee', 'all', 'impasse', 'imp',
    'chemin', 'ch', 'passage', 'pass', 'voie', 'route', 'rte',
    'quai', 'cedex', 'bp',  # CEDEX = postal routing, BP = boite postale
    
    # Indian address terms
    'nagar', 'marg', 'path', 'gali',
}

# Web and Domain Cleanup Pattern
DOMAIN_CLEAN_RE = re.compile(
    r'(\.com|\.net|\.org|\.in|\.co|\.fr|\.io|\.biz|\.edu|\.info|www\.|https?://|@)',
    re.IGNORECASE
)

# S3 ID suffix pattern (removes metadata like "(ID: 35808)")
S3_ID_SUFFIX_RE = re.compile(r'\s*\(ID:\s*\d+\)\s*$', re.IGNORECASE)

# Number Extraction Pattern
NUMBER_RE = re.compile(r'\b\d+\b')

# Postal code patterns (country-agnostic)
# US: 5 digits or 5+4; India: 6 digits; France: 5 digits
POSTAL_CODE_RE = re.compile(r'\b(\d{5}(?:-\d{4})?|\d{6})\b')
