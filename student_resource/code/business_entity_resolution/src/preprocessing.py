"""
Preprocessing, Normalization, Transliteration, and Feature Extraction Utilities
"""

import re
import anyascii
from src.config import CORPORATE_STOPWORDS, DOMAIN_CLEAN_RE


def process_record_text(
    name: str, addr: str
) -> tuple[str, str, list[str], list[str], str, list[str]]:
  """Process name and address in a single optimized pass.

  Returns: (norm_name, norm_addr, name_toks, addr_toks, first_name_tok, nums)
  """
  # 1. Normalize business name
  if isinstance(name, str) and name:
    c_name = DOMAIN_CLEAN_RE.sub(" ", name)
    c_name = anyascii.anyascii(c_name).lower()
    c_name = " ".join(re.sub(r"[^a-z0-9\s]", " ", c_name).split())
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
    addr_toks = [w for w in c_addr.split() if w not in CORPORATE_STOPWORDS]
  else:
    c_addr = ""
    nums = []
    addr_toks = []

  return c_name, c_addr, name_toks, addr_toks, first_name_tok, nums


def normalize_text(text: str) -> str:
  """Quick text normalization."""
  if not isinstance(text, str) or not text.strip():
    return ""
  c = DOMAIN_CLEAN_RE.sub(" ", text)
  c = anyascii.anyascii(c).lower()
  return " ".join(re.sub(r"[^a-z0-9\s]", " ", c).split())


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
