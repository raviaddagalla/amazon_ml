import time
import re
import anyascii
import jellyfish
from src.config import CORPORATE_STOPWORDS, DOMAIN_CLEAN_RE, ABBREVIATION_MAP, S3_ID_SUFFIX_RE
from src.preprocessing import process_record_text, extract_address_components
from src.blocking import get_blocking_keys_from_tokens

name = "Walmart Supercenter Store #1234 (ID: 98765)"
addr = "1234 Main Street, Suite 500, Los Angeles, CA 90001"

N = 10000

t0 = time.time()
for _ in range(N):
    c_n, c_a, n_toks, a_toks, first_w, nums = process_record_text(name, addr)
t1 = time.time()
print(f"process_record_text: {(t1-t0)*1000/N:.3f} ms/record ({(N/(t1-t0)):.0f} rec/s)")

t0 = time.time()
for _ in range(N):
    comp = extract_address_components(addr)
t1 = time.time()
print(f"extract_address_components: {(t1-t0)*1000/N:.3f} ms/record ({(N/(t1-t0)):.0f} rec/s)")

t0 = time.time()
for _ in range(N):
    keys = get_blocking_keys_from_tokens(n_toks, a_toks, nums)
t1 = time.time()
print(f"get_blocking_keys_from_tokens: {(t1-t0)*1000/N:.3f} ms/record ({(N/(t1-t0)):.0f} rec/s)")
