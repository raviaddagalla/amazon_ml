"""
Preprocessing, Normalization, Transliteration, and Feature Extraction Utilities
(Phase 2 upgrade: abbreviation expansion, sorted-token signatures, address component extraction)
"""

import re
import anyascii
from src.config import (
    CORPORATE_STOPWORDS, DOMAIN_CLEAN_RE, ABBREVIATION_MAP,
    S3_ID_SUFFIX_RE, POSTAL_CODE_RE
)


def _apply_abbreviation_expansion(text: str) -> str:
    """Apply bidirectional abbreviation expansion to normalized text."""
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(ABBREVIATION_MAP.get(w, w))
    return ' '.join(expanded)


def _extract_postal_code(text: str) -> str:
    """Extract trailing postal-code-like token from address text.
    
    Supports: US (5/5+4 digits), India (6 digits), France (5 digits),
    and generic alphanumeric trailing codes.
    """
    if not text:
        return ""
    m = POSTAL_CODE_RE.findall(text)
    if m:
        # Return last match (most likely to be postal code)
        return m[-1]
    return ""


def _extract_street_number(text: str) -> str:
    """Extract leading numeric token (street/building number)."""
    if not text:
        return ""
    words = text.split()
    for w in words[:3]:  # Check first 3 tokens
        # Match pure numbers or number-letter combos (e.g., "123A", "45-B")
        cleaned = re.sub(r'[^0-9]', '', w)
        if cleaned and 1 <= len(cleaned) <= 5:
            return cleaned
    return ""


def process_record_text(
    name: str, addr: str
) -> tuple[str, str, list[str], list[str], str, list[str]]:
    """Process name and address in a single optimized pass.

    Returns: (norm_name, norm_addr, name_toks, addr_toks, first_name_tok, nums)
    """
    # 1. Normalize business name
    if isinstance(name, str) and name:
        c_name = S3_ID_SUFFIX_RE.sub('', name)  # Strip S3 ID suffix
        c_name = DOMAIN_CLEAN_RE.sub(" ", c_name)
        c_name = anyascii.anyascii(c_name).lower()
        c_name = " ".join(re.sub(r"[^a-z0-9\s]", " ", c_name).split())
        c_name = _apply_abbreviation_expansion(c_name)
        name_toks = [w for w in c_name.split() if w not in CORPORATE_STOPWORDS]
        first_name_tok = name_toks[0] if name_toks else ""
    else:
        c_name = ""
        name_toks = []
        first_name_tok = ""

    # 2. Normalize address and extract numbers
    if isinstance(addr, str) and addr:
        c_addr = DOMAIN_CLEAN_RE.sub(" ", addr)
        c_addr = anyascii.anyascii(c_addr).lower()
        # Extract numbers with leading zeros stripped
        raw_nums = re.findall(r"\b\d+\b", c_addr)
        nums = [
            n
            for raw in raw_nums
            for n in [raw.lstrip("0")]
            if n and 1 <= len(n) <= 6
        ]

        c_addr = " ".join(re.sub(r"[^a-z0-9\s]", " ", c_addr).split())
        c_addr = _apply_abbreviation_expansion(c_addr)
        addr_toks = [w for w in c_addr.split() if w not in CORPORATE_STOPWORDS]
    else:
        c_addr = ""
        nums = []
        addr_toks = []

    return c_name, c_addr, name_toks, addr_toks, first_name_tok, nums


def sorted_token_signature(tokens: list[str]) -> str:
    """Create a sorted-token signature for word-order invariant matching."""
    if not tokens:
        return ""
    return " ".join(sorted(tokens))


def extract_address_components(addr: str) -> dict:
    """Extract structured address components (country-agnostic).
    
    Returns dict with:
        street_number: leading numeric token
        postal_code: trailing postal-code-like token
        locality_tokens: remaining address tokens after removing number/postal
    """
    if not isinstance(addr, str) or not addr.strip():
        return {
            'street_number': '',
            'postal_code': '',
            'locality_tokens': [],
        }
    
    norm = anyascii.anyascii(addr).lower()
    norm = " ".join(re.sub(r"[^a-z0-9\s]", " ", norm).split())
    norm = _apply_abbreviation_expansion(norm)
    
    street_num = _extract_street_number(norm)
    postal = _extract_postal_code(norm)
    
    # Locality = everything except street number, postal code, and stopwords
    locality = []
    tokens = norm.split()
    for t in tokens:
        if t == street_num and street_num:
            continue
        if t == postal and postal:
            continue
        if t not in CORPORATE_STOPWORDS:
            locality.append(t)
    
    return {
        'street_number': street_num,
        'postal_code': postal,
        'locality_tokens': locality,
    }


def normalize_text(text: str) -> str:
    """Quick text normalization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    c = S3_ID_SUFFIX_RE.sub('', text)  # Strip S3 ID suffix
    c = DOMAIN_CLEAN_RE.sub(" ", c)
    c = anyascii.anyascii(c).lower()
    c = " ".join(re.sub(r"[^a-z0-9\s]", " ", c).split())
    c = _apply_abbreviation_expansion(c)
    return c


def extract_tokens(text: str) -> list[str]:
    """Quick non-stopword tokens."""
    norm = normalize_text(text)
    return [w for w in norm.split() if w not in CORPORATE_STOPWORDS]


def extract_numbers(addr: str) -> set[str]:
    """Quick number extraction without leading zeros."""
    if not isinstance(addr, str) or not addr.strip():
        return set()
    nums = set()
    for raw in re.findall(r"\b\d+\b", addr):
        n = raw.lstrip("0")
        if n and 1 <= len(n) <= 6:
            nums.add(n)
    return nums


def extract_first_word(text: str) -> str:
    """Extract first non-stopword token."""
    toks = extract_tokens(text)
    return toks[0] if toks else ""
