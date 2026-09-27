"""
Canonicalization Engine for Entity Resolution (v2)
Aggressive normalization, transliteration, legal suffix removal,
and structured signature generation for deterministic matching.
"""

import re
import anyascii
import jellyfish
from typing import Tuple, List, Set, Dict

from src_v2.config import (
    S3_ID_SUFFIX_RE,
    DOMAIN_CLEAN_RE,
    CORPORATE_LEGAL_SUFFIXES,
    ABBREVIATION_MAP,
)


def clean_text_basic(text: str) -> str:
    """Basic cleaning: anyascii transliteration, lowercasing, stripping punctuation."""
    if not text or not isinstance(text, str):
        return ""
    # Strip S3 ID suffixes e.g. (ID: 12345)
    t = S3_ID_SUFFIX_RE.sub(" ", text)
    # Strip domain extensions e.g. www., .com, .in
    t = DOMAIN_CLEAN_RE.sub(" ", t)
    # Transliterate Unicode scripts (Telugu, Hindi, Bengali, French accents -> ASCII)
    t = anyascii.anyascii(t).lower()
    # Replace all non-alphanumeric with space
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


def expand_abbreviations(text: str) -> List[str]:
    """Expand abbreviations in token list."""
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(ABBREVIATION_MAP.get(w, w))
    return expanded


def canonicalize_name(name: str) -> Tuple[str, List[str], str]:
    """
    Produce canonical name signature, core tokens, and first core token.
    
    1. Transliterate & clean
    2. Expand abbreviations
    3. Strip corporate/legal suffixes
    4. Sort core tokens for word-order invariance
    
    Returns:
        (canonical_name_signature, core_tokens, first_core_token)
    """
    cleaned = clean_text_basic(name)
    if not cleaned:
        return "", [], ""
    
    tokens = expand_abbreviations(cleaned)
    # Strip corporate/legal suffixes
    core_tokens = [w for w in tokens if w not in CORPORATE_LEGAL_SUFFIXES and len(w) > 1]
    if not core_tokens:
        # If all tokens were filtered (e.g. "Services Inc"), retain raw tokens
        core_tokens = [w for w in tokens if len(w) > 1]
    
    canonical_sig = " ".join(sorted(core_tokens))
    first_token = core_tokens[0] if core_tokens else ""
    
    return canonical_sig, core_tokens, first_token


def canonicalize_address(addr: str) -> Tuple[str, List[str], List[str]]:
    """
    Produce canonical address signature, extracted numbers, and locality tokens.
    
    Numbers (street numbers, building numbers, postal codes) are critical
    deterministic anchors.
    
    Returns:
        (canonical_addr_signature, sorted_numbers, locality_tokens)
    """
    cleaned = clean_text_basic(addr)
    if not cleaned:
        return "", [], []
    
    tokens = expand_abbreviations(cleaned)
    
    # Extract numbers with leading zeros stripped
    raw_nums = [re.sub(r"^0+", "", w) for w in tokens if w.isdigit()]
    nums = sorted(list(set(n for n in raw_nums if n and len(n) <= 6)))
    
    # Locality / street name tokens (non-numbers, length > 1, not corporate suffix)
    locality_toks = sorted(list(set(
        w for w in tokens 
        if not w.isdigit() and len(w) > 1 and w not in CORPORATE_LEGAL_SUFFIXES
    )))
    
    canonical_sig = " ".join(nums) + " | " + " ".join(locality_toks)
    
    return canonical_sig, nums, locality_toks


def get_record_signatures(name: str, addr: str) -> Dict:
    """Precompute all canonical signatures and token features for a record."""
    name_sig, core_name_toks, first_name_tok = canonicalize_name(name)
    addr_sig, addr_nums, locality_toks = canonicalize_address(addr)
    
    # Phonetic metaphones of core name tokens
    metaphones = [jellyfish.metaphone(w) for w in core_name_toks if len(w) >= 3]
    
    # Character 3-grams of name signature for fuzzy typo blocking
    name_no_space = "".join(core_name_toks)
    char_trigrams = []
    if len(name_no_space) >= 3:
        char_trigrams = [name_no_space[i:i+3] for i in range(len(name_no_space) - 2)]
    
    return {
        "canon_name": name_sig,
        "core_name_toks": core_name_toks,
        "first_name_tok": first_name_tok,
        "canon_addr": addr_sig,
        "addr_nums": addr_nums,
        "addr_nums_set": set(addr_nums),
        "locality_toks": locality_toks,
        "metaphones": metaphones,
        "char_trigrams": char_trigrams,
    }
