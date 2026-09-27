"""
Exhaustive Multi-Strategy Retrieval Engine (v2)
Builds high-recall inverted index across canonical name signatures,
core tokens, phonetic keys, character n-grams, and address numbers.
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np

from src_v2.config import MAX_BLOCKING_KEY_SIZE, TOP_CANDIDATES_PER_S1


class InvertedIndexEngine:
    """Multi-key inverted index supporting exact, token, phonetic, and numeric lookups."""
    
    def __init__(self):
        self.index = defaultdict(list)
        self.key_sizes = {}
    
    def build_index(self, pool_signatures: List[Dict]):
        """
        Build index over all candidate records in pool.
        
        Args:
            pool_signatures: list of precomputed record signature dicts from get_record_signatures
        """
        self.index.clear()
        
        for idx, sig in enumerate(pool_signatures):
            # 1. Exact canonical name signature key
            if sig["canon_name"]:
                self.index[f"nsig:{sig['canon_name']}"].append(idx)
            
            # 2. Individual core name tokens
            for tok in sig["core_name_toks"][:6]:
                if len(tok) >= 3:
                    self.index[f"tok:{tok}"].append(idx)
            
            # 3. Phonetic metaphones
            for meta in sig["metaphones"][:3]:
                if meta:
                    self.index[f"m:{meta}"].append(idx)
            
            # 4. Address numbers (postal codes, street numbers >= 3 digits)
            for num in sig["addr_nums"]:
                if len(num) >= 3:
                    self.index[f"num:{num}"].append(idx)
            
            # 5. Character trigrams for typo resilience (first 4 trigrams)
            for tri in sig["char_trigrams"][:4]:
                self.index[f"tri:{tri}"].append(idx)
        
        # Precompute key sizes and prune massive keys (> MAX_BLOCKING_KEY_SIZE)
        keys_to_prune = []
        for k, lst in self.index.items():
            if len(lst) > MAX_BLOCKING_KEY_SIZE and not k.startswith("nsig:"):
                keys_to_prune.append(k)
        for k in keys_to_prune:
            del self.index[k]
    
    def query(self, s1_sig: Dict, top_k: int = TOP_CANDIDATES_PER_S1) -> List[Tuple[int, float]]:
        """
        Query inverted index with an S1 record signature.
        
        Returns:
            list of (pool_idx, accumulated_retrieval_score) sorted descending by score
        """
        candidate_scores = defaultdict(float)
        
        # 1. Exact canonical name key (highest priority)
        if s1_sig["canon_name"]:
            key = f"nsig:{s1_sig['canon_name']}"
            if key in self.index:
                for idx in self.index[key]:
                    candidate_scores[idx] += 10.0
        
        # 2. Core name tokens
        for tok in s1_sig["core_name_toks"][:6]:
            key = f"tok:{tok}"
            if key in self.index:
                weight = 4.0 if tok == s1_sig["first_name_tok"] else 2.5
                for idx in self.index[key]:
                    candidate_scores[idx] += weight
        
        # 3. Phonetic keys
        for meta in s1_sig["metaphones"][:3]:
            key = f"m:{meta}"
            if key in self.index:
                for idx in self.index[key]:
                    candidate_scores[idx] += 2.0
        
        # 4. Address numbers (very strong locality anchor)
        for num in s1_sig["addr_nums"]:
            key = f"num:{num}"
            if key in self.index:
                for idx in self.index[key]:
                    candidate_scores[idx] += 3.0
        
        # 5. Character trigrams
        for tri in s1_sig["char_trigrams"][:4]:
            key = f"tri:{tri}"
            if key in self.index:
                for idx in self.index[key]:
                    candidate_scores[idx] += 0.5
        
        if not candidate_scores:
            return []
        
        # Sort and return top_k
        sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_cands
