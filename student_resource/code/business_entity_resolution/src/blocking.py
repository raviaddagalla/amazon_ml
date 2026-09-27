"""
Candidate Generation and Multi-Key Inverted Index Blocking Engine
(Phase 3 upgrade: 6 high-recall, low-collision keys for maximum throughput and recall)
"""

import jellyfish
from src.config import MAX_BLOCKING_KEY_SIZE, TOP_CANDIDATES_PER_S1
from src.preprocessing import process_record_text, sorted_token_signature


def get_blocking_keys_from_tokens(
    name_toks: list[str], addr_toks: list[str], nums: list[str]
) -> list[str]:
    """Generate high-recall blocking keys from tokens (max 6 keys per record).
    
    Strategies:
    1. First name token (n1:)
    2. Name bigram - first two tokens (nb:)
    3. Phonetic metaphone key of first name token (m1:) - spelling robustness
    4. Sorted-token signature key (nsort:) - word-order transposition robustness
    5. PIN/postal code (pin:) - precise geographic locality
    6. Number + locality composite (na:) - street-level co-occurrence
    """
    keys = []
    if name_toks:
        w0 = name_toks[0]
        if len(w0) >= 2:
            keys.append("n1:" + w0)
            m0 = jellyfish.metaphone(w0)
            if len(m0) >= 2:
                keys.append("m1:" + m0)
        
        if len(name_toks) >= 2:
            w1 = name_toks[1]
            if len(w1) >= 2:
                keys.append(f"nb:{w0}_{w1}")
            # Sorted-token signature for word-order transposition robustness
            sig = sorted_token_signature(name_toks[:3])
            if len(sig) >= 4:
                keys.append("nsort:" + sig)

    if nums and addr_toks:
        for n in nums[:1]:
            for a in addr_toks[:1]:
                if len(a) >= 3 and a != n:
                    keys.append(f"na:{n}_{a}")
                    break

    for n in nums:
        if len(n) in (5, 6):
            keys.append("pin:" + n)
            break  # at most one PIN key

    return keys


def extract_blocking_keys(name: str, addr: str) -> list[str]:
    """Generate high-recall blocking keys for arbitrary record."""
    _, _, name_toks, addr_toks, _, nums = process_record_text(name, addr)
    return get_blocking_keys_from_tokens(name_toks, addr_toks, nums)
