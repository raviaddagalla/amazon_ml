"""
Feature Engineering for Business Entity Resolution Matching Model
(Phase 4 upgrade: 34 discriminative features, zero-overhead memory design)
"""

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, DamerauLevenshtein


FEATURE_NAMES = [
    # --- Name features (14) ---
    'n_set',              # token_set_ratio on names
    'n_sort',             # token_sort_ratio on names
    'n_part',             # partial_ratio on names
    'n_ratio',            # simple ratio on names
    'first_match',        # first token exact match
    'n_jacc',             # Jaccard similarity on name tokens
    'n_jaro_winkler',     # Jaro-Winkler similarity on names
    'n_damerau_lev',      # Normalized Damerau-Levenshtein distance on names
    'n_lcs_ratio',        # Longest common substring ratio on names
    'n_sorted_ratio',     # Ratio on sorted-token signatures
    'last_match',         # Last token exact match
    'initials_match',     # Token initials match
    'first_phonetic',     # First token metaphone match
    'phonetic_jacc',      # Jaccard similarity on token metaphones
    
    # --- Address features (10) ---
    'a_set',              # token_set_ratio on addresses
    'a_part',             # partial_ratio on addresses
    'a_jacc',             # Jaccard similarity on address tokens
    'a_jaro_winkler',     # Jaro-Winkler on addresses
    'a_ratio',            # Simple ratio on addresses
    'a_damerau_lev',      # Normalized Damerau-Levenshtein on addresses
    'a_lcs_ratio',        # Longest common substring ratio on addresses
    'addr_missing_s1',    # S1 address is missing
    'addr_missing_cand',  # Candidate address is missing
    'addr_both_present',  # Both addresses present
    
    # --- Address component features (5) ---
    'street_num_match',     # Street number exact match
    'street_num_mismatch',  # Street number mismatch (both present, different)
    'postal_match',         # Postal code exact match
    'postal_mismatch',      # Postal code mismatch (both present, different)
    'locality_jacc',        # Address token Jaccard similarity
    
    # --- Number features (3) ---
    'num_overlap',        # Count of overlapping numbers
    'has_num_overlap',    # Binary: any number overlap
    'num_mismatch',       # Binary: both have numbers but no overlap
    
    # --- Blocking & Metadata features (2) ---
    'blk_score',          # Blocking agreement score
    'is_us',              # Country indicator (1.0 for US, 0.0 for others)
]


def _longest_common_substring_ratio(s1: str, s2: str) -> float:
    """Compute LCS ratio = 2 * len(LCS) / (len(s1) + len(s2))."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    m, n = len(s1), len(s2)
    # Fast paths for exact substring containment
    if m <= n and s1 in s2:
        return (2.0 * m) / (m + n)
    if n < m and s2 in s1:
        return (2.0 * n) / (m + n)
    
    # Pre-allocate two reusable buffers
    max_len = 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ch = s1[i - 1]
        for j in range(1, n + 1):
            if ch == s2[j - 1]:
                val = prev[j - 1] + 1
                curr[j] = val
                if val > max_len:
                    max_len = val
            else:
                curr[j] = 0
        prev, curr = curr, prev
    return (2.0 * max_len) / (m + n)


def _initials_match(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Check if one side's initials match the other side's token sequence."""
    if not tokens_a or not tokens_b:
        return 0.0
    
    initials_a = ''.join(t[0] for t in tokens_a if t)
    initials_b = ''.join(t[0] for t in tokens_b if t)
    
    for t in tokens_b:
        if t == initials_a and len(initials_a) >= 2:
            return 1.0
    for t in tokens_a:
        if t == initials_b and len(initials_b) >= 2:
            return 1.0
    
    if initials_a == initials_b and len(initials_a) >= 2:
        return 0.5
    
    return 0.0


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
    s1_addr_components: dict | None = None,
    c_addr_components: dict | None = None,
    country: str = "US",
    s1_precomputed: tuple | None = None,
) -> list[float]:
    """Compute 34 discriminative features for a candidate pair."""
    
    # Precomputed S1 fields or compute on-the-fly
    if s1_precomputed is not None:
        (s1_sorted, s1_last, s1_name_tokens_list, s1_m0, s1_metaphones, s1_street_num, s1_postal) = s1_precomputed
    else:
        s1_sorted = " ".join(sorted(s1_n_words)) if s1_n_words else ""
        s1_last = list(s1_n_words)[-1] if s1_n_words else ""
        s1_name_tokens_list = s1_norm_name.split() if s1_norm_name else []
        s1_m0 = jellyfish.metaphone(s1_first_word) if s1_first_word else ""
        s1_metaphones = {jellyfish.metaphone(w) for w in s1_n_words if w}
        s1_street_num = next((n for n in s1_nums if len(n) <= 4), "")
        s1_postal = next((n for n in s1_nums if len(n) in (5, 6)), "")

    # ========================================================================
    # NAME FEATURES (14)
    # ========================================================================
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
    
    n_jaro_winkler = (
        JaroWinkler.similarity(s1_norm_name, c_norm_name) * 100.0
        if s1_norm_name and c_norm_name else 0.0
    )
    n_damerau_lev = (
        DamerauLevenshtein.normalized_similarity(s1_norm_name, c_norm_name) * 100.0
        if s1_norm_name and c_norm_name else 0.0
    )
    n_lcs_ratio = _longest_common_substring_ratio(s1_norm_name, c_norm_name) * 100.0
    
    c_sorted = " ".join(sorted(c_n_words)) if c_n_words else ""
    n_sorted_ratio = fuzz.ratio(s1_sorted, c_sorted) if s1_sorted and c_sorted else 0.0
    
    c_last = list(c_n_words)[-1] if c_n_words else ""
    last_match = 1.0 if s1_last and c_last and s1_last == c_last else 0.0
    
    c_name_tokens_list = c_norm_name.split() if c_norm_name else []
    initials_match = _initials_match(s1_name_tokens_list, c_name_tokens_list)
    
    # Phonetic name features
    c_m0 = jellyfish.metaphone(c_first_word) if c_first_word else ""
    first_phonetic = 1.0 if s1_m0 and c_m0 and s1_m0 == c_m0 else 0.0
    
    c_metaphones = {jellyfish.metaphone(w) for w in c_n_words if w}
    phonetic_jacc = (
        len(s1_metaphones & c_metaphones) / max(1, len(s1_metaphones | c_metaphones))
        if s1_metaphones or c_metaphones else 0.0
    )
    
    # ========================================================================
    # ADDRESS FEATURES (10)
    # ========================================================================
    s1_has_addr = bool(s1_norm_addr)
    c_has_addr = bool(c_norm_addr)
    addr_missing_s1 = 0.0 if s1_has_addr else 1.0
    addr_missing_cand = 0.0 if c_has_addr else 1.0
    addr_both_present = 1.0 if (s1_has_addr and c_has_addr) else 0.0
    
    if s1_has_addr and c_has_addr:
        a_set = fuzz.token_set_ratio(s1_norm_addr, c_norm_addr)
        a_part = fuzz.partial_ratio(s1_norm_addr, c_norm_addr)
        a_jaro_winkler = JaroWinkler.similarity(s1_norm_addr, c_norm_addr) * 100.0
        a_ratio = fuzz.ratio(s1_norm_addr, c_norm_addr)
        a_damerau_lev = DamerauLevenshtein.normalized_similarity(s1_norm_addr, c_norm_addr) * 100.0
        a_lcs_ratio = _longest_common_substring_ratio(s1_norm_addr, c_norm_addr) * 100.0
    else:
        a_set = 0.0
        a_part = 0.0
        a_jaro_winkler = 0.0
        a_ratio = 0.0
        a_damerau_lev = 0.0
        a_lcs_ratio = 0.0
    
    a_jacc = len(s1_a_words & c_a_words) / max(1, len(s1_a_words | c_a_words))
    
    # ========================================================================
    # ADDRESS COMPONENT FEATURES (5)
    # Efficient on-the-fly extraction from numeric and token sets
    # ========================================================================
    c_street_num = next((n for n in c_nums if len(n) <= 4), "")
    street_num_match = 1.0 if (s1_street_num and c_street_num and s1_street_num == c_street_num) else 0.0
    street_num_mismatch = 1.0 if (s1_street_num and c_street_num and s1_street_num != c_street_num) else 0.0
    
    c_postal = next((n for n in c_nums if len(n) in (5, 6)), "")
    postal_match = 1.0 if (s1_postal and c_postal and s1_postal == c_postal) else 0.0
    postal_mismatch = 1.0 if (s1_postal and c_postal and s1_postal != c_postal) else 0.0
    
    locality_jacc = a_jacc
    
    # ========================================================================
    # NUMBER FEATURES (3)
    # ========================================================================
    num_overlap = float(len(s1_nums & c_nums))
    has_num_overlap = 1.0 if num_overlap > 0 else 0.0
    num_mismatch = 1.0 if (s1_nums and c_nums and num_overlap == 0) else 0.0
    
    # ========================================================================
    # BLOCKING & METADATA (2)
    # ========================================================================
    is_us = 1.0 if country == "US" else 0.0
    
    return [
        # Name features (14)
        n_set, n_sort, n_part, n_ratio, first_match, n_jacc,
        n_jaro_winkler, n_damerau_lev, n_lcs_ratio, n_sorted_ratio,
        last_match, initials_match, first_phonetic, phonetic_jacc,
        
        # Address features (10)
        a_set, a_part, a_jacc,
        a_jaro_winkler, a_ratio, a_damerau_lev, a_lcs_ratio,
        addr_missing_s1, addr_missing_cand, addr_both_present,
        
        # Address component features (5)
        street_num_match, street_num_mismatch,
        postal_match, postal_mismatch, locality_jacc,
        
        # Number features (3)
        num_overlap, has_num_overlap, num_mismatch,
        
        # Blocking & Metadata (2)
        float(blk_score), is_us,
    ]
