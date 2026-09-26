"""
Post-processing, Mutual Exclusivity Assignment, and Submission File Exporters
"""

import os
from collections import defaultdict
from src.config import MATCHING_OUTPUT, CANDIDATE_OUTPUT, PROBABILITY_THRESHOLD


def resolve_mutual_exclusivity(
    scored_candidates: list[tuple[str, str, float]],
    threshold: float = PROBABILITY_THRESHOLD
) -> dict[str, list[str]]:
    """Apply bipartite 1-to-1 assignment constraint on candidate matches.

    Each S2 or S3 entity represents a single noisy real-world business record
    and can match at most ONE Source 1 reference entity.
    If multiple S1 entities claim the same candidate above threshold,
    the candidate is assigned exclusively to the S1 entity with the highest ML confidence score.
    """
    best_assignment: dict[str, tuple[str, float]] = {}

    for s1_id, cand_id, score in scored_candidates:
        if score >= threshold:
            if cand_id not in best_assignment or score > best_assignment[cand_id][1]:
                best_assignment[cand_id] = (s1_id, score)

    final_matches: dict[str, list[str]] = defaultdict(list)
    for cand_id, (s1_id, _) in best_assignment.items():
        final_matches[s1_id].append(cand_id)

    return final_matches


def export_submission_files(
    all_s1_ids: list[str],
    candidates_dict: dict[str, list[str]],
    matches_dict: dict[str, list[str]],
    matching_path: str = MATCHING_OUTPUT,
    candidate_path: str = CANDIDATE_OUTPUT
) -> None:
    """Write valid TSV files for matching_results.tsv and candidate_pairs.tsv.

    Strictly satisfies all validation criteria:
    - Tab-separated format (.tsv)
    - Exactly one row per Source 1 entity in exact reference order
    - Comma-separated ID lists with no spaces or quoting
    - Singletons are emitted with empty ID lists
    - Final matches are verified to be a strict subset of candidates
    """
    os.makedirs(os.path.dirname(matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(candidate_path), exist_ok=True)

    print(f"Exporting {len(all_s1_ids):,} S1 entities to submission files...", flush=True)

    # 1. Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = matches_dict.get(s1_id, [])
            match_str = ",".join(matches) if matches else ""
            f_match.write(f"{s1_id}\t{match_str}\n")

    # 2. Write candidate_pairs.tsv
    with open(candidate_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            candidates = candidates_dict.get(s1_id, [])
            # Guarantee every match is in the candidate list
            matches = matches_dict.get(s1_id, [])
            cand_set = set(candidates)
            combined_cands = list(candidates)
            for m in matches:
                if m not in cand_set:
                    combined_cands.append(m)
                    cand_set.add(m)
            cand_str = ",".join(combined_cands) if combined_cands else ""
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    print(f"Successfully generated:\n  - {matching_path}\n  - {candidate_path}", flush=True)
