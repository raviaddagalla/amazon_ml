"""
Tiered Decision Cascade Engine (v2)
Resolves candidate pairs through ordered confidence tiers:
- Tier 1: Exact canonical match (auto-accept, score=1.0)
- Tier 2: High-confidence fuzzy match with address anchor (score=0.95)
- Tier 3: Residual ML classifier for hard fuzzy cases
- Tier 4: Singleton confidence guard (suppress false positives)
Followed by global 1-to-1 bipartite mutual exclusivity resolution.
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import numpy as np
from rapidfuzz import fuzz, distance

from src_v2.config import (
    TIER1_SCORE,
    TIER2_SCORE,
    TIER2_NAME_SIM_BAR,
    TIER3_PROB_THRESHOLD,
    TIER4_MARGIN_GUARD,
)


def compute_pair_similarity_features(s1_sig: Dict, cand_sig: Dict, retrieval_score: float) -> List[float]:
    """Compute rich similarity vector for Tier 3 residual classification."""
    s1_name = s1_sig["canon_name"]
    c_name = cand_sig["canon_name"]
    
    # 1. Name string metrics
    n_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    n_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    n_ratio = fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    n_jw = distance.JaroWinkler.similarity(s1_name, c_name) if s1_name and c_name else 0.0
    n_lev = 1.0 - (distance.DamerauLevenshtein.normalized_distance(s1_name, c_name) if s1_name and c_name else 1.0)
    
    # 2. Token overlap
    s1_toks = set(s1_sig["core_name_toks"])
    c_toks = set(cand_sig["core_name_toks"])
    tok_union = s1_toks | c_toks
    n_jacc = len(s1_toks & c_toks) / len(tok_union) if tok_union else 0.0
    first_match = 1.0 if s1_sig["first_name_tok"] and s1_sig["first_name_tok"] == cand_sig["first_name_tok"] else 0.0
    
    # 3. Phonetic overlap
    s1_meta = set(s1_sig["metaphones"])
    c_meta = set(cand_sig["metaphones"])
    meta_union = s1_meta | c_meta
    meta_jacc = len(s1_meta & c_meta) / len(meta_union) if meta_union else 0.0
    
    # 4. Address metrics
    s1_addr = s1_sig["canon_addr"]
    c_addr = cand_sig["canon_addr"]
    a_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    a_jw = distance.JaroWinkler.similarity(s1_addr, c_addr) if s1_addr and c_addr else 0.0
    
    # 5. Number metrics (street numbers / pincodes)
    nums1 = s1_sig["addr_nums_set"]
    nums2 = cand_sig["addr_nums_set"]
    overlap_nums = nums1 & nums2
    has_num_overlap = 1.0 if overlap_nums else 0.0
    num_overlap_count = float(len(overlap_nums))
    has_num_mismatch = 1.0 if (nums1 and nums2 and not overlap_nums) else 0.0
    
    # 6. Locality tokens Jaccard
    loc1 = set(s1_sig["locality_toks"])
    loc2 = set(cand_sig["locality_toks"])
    loc_union = loc1 | loc2
    loc_jacc = len(loc1 & loc2) / len(loc_union) if loc_union else 0.0
    
    return [
        n_set, n_sort, n_ratio, n_jw, n_lev, n_jacc, first_match, meta_jacc,
        a_set, a_jw, has_num_overlap, num_overlap_count, has_num_mismatch,
        loc_jacc, retrieval_score
    ]


def evaluate_tier1_match(s1_sig: Dict, cand_sig: Dict) -> bool:
    """
    Tier 1: Exact canonical name match with address consistency.
    Auto-accept if canonical names match and addresses do not contradict.
    """
    if not s1_sig["canon_name"] or not cand_sig["canon_name"]:
        return False
    if s1_sig["canon_name"] != cand_sig["canon_name"]:
        return False
    
    nums1 = s1_sig["addr_nums_set"]
    nums2 = cand_sig["addr_nums_set"]
    loc_overlap = len(set(s1_sig["locality_toks"]) & set(cand_sig["locality_toks"]))
    
    s1_addr = s1_sig["canon_addr"]
    c_addr = cand_sig["canon_addr"]
    a_set = fuzz.token_set_ratio(s1_addr, c_addr) if s1_addr and c_addr else 0.0

    # Case A: Both records have numbers (street numbers / pincodes)
    if nums1 and nums2:
        # Numbers MUST overlap
        if not (nums1 & nums2):
            return False
        # And locality must not be completely contradictory
        return loc_overlap >= 1 or a_set >= 45.0

    # Case B: Both records have address text, but at least one lacks numbers
    if s1_addr and c_addr:
        # Require strong address text agreement or multiple shared locality tokens
        return a_set >= 70.0 or loc_overlap >= 2

    # Case C: One or both addresses are completely missing
    # Only accept if name is distinctive (at least 2 core tokens and 10+ characters)
    return len(s1_sig["core_name_toks"]) >= 2 and len(s1_sig["canon_name"]) >= 10


def evaluate_tier2_match(s1_sig: Dict, cand_sig: Dict) -> bool:
    """
    Tier 2: High-confidence fuzzy match with address anchor.
    Requires:
    1. Address numbers match exactly (at least 1 shared number)
    2. Address locality tokens not contradictory (loc_overlap >= 1 or a_set >= 45)
    3. Name token set ratio or Jaro-Winkler >= TIER2_NAME_SIM_BAR (0.85)
    """
    nums1 = s1_sig["addr_nums_set"]
    nums2 = cand_sig["addr_nums_set"]
    if not (nums1 and nums2 and (nums1 & nums2)):
        return False  # Tier 2 strictly requires number anchor
    
    loc_overlap = len(set(s1_sig["locality_toks"]) & set(cand_sig["locality_toks"]))
    s1_addr = s1_sig["canon_addr"]
    c_addr = cand_sig["canon_addr"]
    a_set = fuzz.token_set_ratio(s1_addr, c_addr) if s1_addr and c_addr else 0.0
    if loc_overlap == 0 and a_set < 45.0:
        return False  # Conflicting city/state
    
    s1_name = s1_sig["canon_name"]
    c_name = cand_sig["canon_name"]
    if not s1_name or not c_name:
        return False
    
    token_sim = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    jw_sim = distance.JaroWinkler.similarity(s1_name, c_name)
    
    return max(token_sim, jw_sim) >= TIER2_NAME_SIM_BAR


def resolve_bipartite_matches(candidate_assignments: List[Tuple[str, str, float]]) -> Dict[str, List[str]]:
    """
    Resolve 1-to-1 competitive assignment across all candidate pairs.
    
    Each candidate entity (S2/S3) is assigned to at most ONE S1 entity,
    giving preference to higher scores.
    
    Returns:
        dict mapping s1_id -> list of matched candidate IDs
    """
    best_cand_assignment = {}  # cand_id -> (s1_id, score)
    
    for s1_id, cand_id, score in candidate_assignments:
        prev = best_cand_assignment.get(cand_id)
        if prev is None or score > prev[1]:
            best_cand_assignment[cand_id] = (s1_id, score)
    
    s1_matches = defaultdict(list)
    for cand_id, (s1_id, _) in best_cand_assignment.items():
        s1_matches[s1_id].append(cand_id)
    
    return s1_matches
