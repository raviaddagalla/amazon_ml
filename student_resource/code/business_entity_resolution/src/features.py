"""
Feature Engineering for Business Entity Resolution Matching Model
"""

from rapidfuzz import fuzz

FEATURE_NAMES = [
    'n_set',
    'n_sort',
    'n_part',
    'n_ratio',
    'first_match',
    'n_jacc',
    'a_set',
    'a_part',
    'a_jacc',
    'num_overlap',
    'has_num_overlap',
    'num_mismatch',
    'blk_score',
]


def compute_pair_features(
    s1_norm_name: str,
    s1_norm_addr: str,
    s1_nums: set[str],
    s1_n_words: set[str],
    s1_a_words: set[str],
    s1_first_word: str,
    c_norm_name: str,
    c_norm_addr: str,
    c_nums: set[str],
    c_n_words: set[str],
    c_a_words: set[str],
    c_first_word: str,
    blk_score: float,
) -> list[float]:
  """Compute 13 discriminative string similarity, token overlap, and address

  features.
  """
  # Name Similarities (RapidFuzz C++ SIMD)
  n_set = fuzz.token_set_ratio(s1_norm_name, c_norm_name)
  n_sort = fuzz.token_sort_ratio(s1_norm_name, c_norm_name)
  n_part = fuzz.partial_ratio(s1_norm_name, c_norm_name)
  n_ratio = fuzz.ratio(s1_norm_name, c_norm_name)

  first_match = (
      1.0
      if s1_first_word and c_first_word and s1_first_word == c_first_word
      else 0.0
  )
  n_jacc = len(s1_n_words & c_n_words) / max(1, len(s1_n_words | c_n_words))

  # Address Similarities
  if s1_norm_addr and c_norm_addr:
    a_set = fuzz.token_set_ratio(s1_norm_addr, c_norm_addr)
    a_part = fuzz.partial_ratio(s1_norm_addr, c_norm_addr)
  else:
    a_set = 50.0
    a_part = 50.0

  a_jacc = len(s1_a_words & c_a_words) / max(1, len(s1_a_words | c_a_words))

  # Number Overlap & Mismatch Signals
  num_overlap = float(len(s1_nums & c_nums))
  has_num_overlap = 1.0 if num_overlap > 0 else 0.0
  num_mismatch = 1.0 if (s1_nums and c_nums and num_overlap == 0) else 0.0

  return [
      n_set,
      n_sort,
      n_part,
      n_ratio,
      first_match,
      n_jacc,
      a_set,
      a_part,
      a_jacc,
      num_overlap,
      has_num_overlap,
      num_mismatch,
      float(blk_score),
  ]
