"""
Configuration and Hyperparameters for Business Entity Resolution Pipeline
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
TOP_CANDIDATES_PER_S1 = 25        # Candidates generated per S1 entity for ML ranking
CANDIDATE_FILE_LIMIT = 25         # Max candidates to output in candidate_pairs.tsv

# Matching Model Hyperparameters
PROBABILITY_THRESHOLD = 0.50      # Tuned on held-out validation set for macro F_0.5

# Legal and Generic Stopwords (Excluding Geographic Names)
CORPORATE_STOPWORDS = {
    'inc', 'llc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'group', 'enterprises', 'enterprise',
    'holdings', 'holding', 'sarl', 'sasu', 'sas', 'sci', 'llp', 'pllc', 'gmbh',
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'to', 'a', 'an',
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'floor', 'fl', 'unit', 'suite', 'ste', 'near', 'opp', 'behind', 'block',
    'sector', 'plot', 'no', 'building', 'bldg', 'tower',
    'rue', 'boulevard', 'blvd', 'allee', 'des', 'du', 'de', 'la', 'le'
}

# Web and Domain Cleanup Pattern
DOMAIN_CLEAN_RE = re.compile(
    r'(\.com|\.net|\.org|\.in|\.co|\.fr|\.io|\.biz|\.edu|\.info|www\.|https?://|@)',
    re.IGNORECASE
)

# Number Extraction Pattern
NUMBER_RE = re.compile(r'\b\d+\b')
