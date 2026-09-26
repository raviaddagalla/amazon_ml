"""
Candidate Generation and Multi-Key Inverted Index Blocking Engine
"""

from collections import Counter, defaultdict
from src.config import MAX_BLOCKING_KEY_SIZE, TOP_CANDIDATES_PER_S1
from src.preprocessing import process_record_text


def get_blocking_keys_from_tokens(
    name_toks: list[str], addr_toks: list[str], nums: list[str]
) -> list[str]:
  """Generate compact, high-recall blocking keys directly from tokens."""
  keys = []
  if name_toks:
    w0 = name_toks[0]
    if len(w0) >= 2:
      keys.append("n1:" + w0)
      if len(w0) >= 4:
        keys.append("npref:" + w0[:4])
    if len(name_toks) >= 2:
      w1 = name_toks[1]
      if len(w1) >= 2:
        keys.append(f"nb:{w0}_{w1}")
        keys.append("n2:" + w1)

  if nums and addr_toks:
    # First number + first locality/street token
    for n in nums[:2]:
      for a in addr_toks[:2]:
        if len(a) >= 3 and a != n:
          keys.append(f"na:{n}_{a}")
          break

  for n in nums:
    if len(n) in (5, 6):
      keys.append("pin:" + n)

  return keys


def extract_blocking_keys(name: str, addr: str) -> list[str]:
  """Generate high-recall blocking keys for arbitrary record."""
  _, _, name_toks, addr_toks, _, nums = process_record_text(name, addr)
  return get_blocking_keys_from_tokens(name_toks, addr_toks, nums)
