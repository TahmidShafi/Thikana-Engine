"""
src/disambiguator.py

Context-Aware Administrative Homonym Disambiguator for Thikana-Engine.
Resolves ambiguous Bangladeshi place names (e.g., Mirpur in Dhaka vs. Kushtia;
Mohammadpur in Dhaka vs. Magura vs. Noakhali) by scoring candidates against
parent-child co-occurrences, postal codes, and distractor street penalties.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import AdminLevel, AdminNode
from .gazetteer import BangladeshGazetteer
from .phonetic_trie import PhoneticGazetteerMatcher, banglish_phonetic_key, levenshtein_distance


# -----------------------------------------------------------------------------
# Known Postal Code Lookup Table for Targeted Disambiguation
# -----------------------------------------------------------------------------
POSTCODE_TO_ADMIN: Dict[str, Dict[str, str]] = {
    # Dhaka Metro
    "1216": {"upazila": "Mirpur", "district": "Dhaka", "division": "Dhaka"},
    "1221": {"upazila": "Mirpur", "district": "Dhaka", "division": "Dhaka"},
    "1218": {"upazila": "Mirpur", "district": "Dhaka", "division": "Dhaka"},
    "1207": {"upazila": "Mohammadpur", "district": "Dhaka", "division": "Dhaka"},
    "1205": {"upazila": "Dhanmondi", "district": "Dhaka", "division": "Dhaka"},
    "1209": {"upazila": "Dhanmondi", "district": "Dhaka", "division": "Dhaka"},
    "1230": {"upazila": "Uttara", "district": "Dhaka", "division": "Dhaka"},
    "1212": {"upazila": "Gulshan", "district": "Dhaka", "division": "Dhaka"},
    "1213": {"upazila": "Banani", "district": "Dhaka", "division": "Dhaka"},
    "1214": {"upazila": "Badda", "district": "Dhaka", "division": "Dhaka"},
    "1000": {"upazila": "Motijheel", "district": "Dhaka", "division": "Dhaka"},
    "1208": {"upazila": "Tejgaon", "district": "Dhaka", "division": "Dhaka"},
    "1215": {"upazila": "Tejgaon", "district": "Dhaka", "division": "Dhaka"},
    # Chattogram Metro
    "4216": {"upazila": "Halishahar", "district": "Chattogram", "division": "Chattagram"},
    "4224": {"upazila": "Halishahar", "district": "Chattogram", "division": "Chattagram"},
    "4225": {"upazila": "Halishahar", "district": "Chattogram", "division": "Chattagram"},
    "4000": {"upazila": "Kotwali", "district": "Chattogram", "division": "Chattagram"},
    "4100": {"upazila": "Agrabad", "district": "Chattogram", "division": "Chattagram"},
    "4203": {"upazila": "Panchlaish", "district": "Chattogram", "division": "Chattagram"},
    "4212": {"upazila": "Chandgaon", "district": "Chattogram", "division": "Chattagram"},
    # Sylhet Metro
    "3100": {"upazila": "Shahjalal Upashahar", "district": "Sylhet", "division": "Sylhet"},
    # Kushtia
    "7030": {"upazila": "Mirpur", "district": "Kushtia", "division": "Khulna"},
    "7031": {"upazila": "Mirpur", "district": "Kushtia", "division": "Khulna"},
    "7000": {"district": "Kushtia", "division": "Khulna"},
    # Magura
    "7630": {"upazila": "Mohammadpur", "district": "Magura", "division": "Khulna"},
    "7631": {"upazila": "Mohammadpur", "district": "Magura", "division": "Khulna"},
    "7600": {"district": "Magura", "division": "Khulna"},
    # Noakhali
    "3830": {"upazila": "Subarnachar", "district": "Noakhali", "division": "Chattagram"},
    "3870": {"upazila": "Chatkhil", "district": "Noakhali", "division": "Chattagram"},
    "3850": {"upazila": "Senbug", "district": "Noakhali", "division": "Chattagram"},
}

DISTRACTOR_SUFFIXES = {
    "para", "road", "rd", "lane", "goli", "street", "bazar", "market",
    "bari", "house", "mor", "point", "পাড়া", "পাড়া", "রোড", "রাস্তা",
    "গলি", "বাজার", "মোড়", "পয়েন্ট"
}


# -----------------------------------------------------------------------------
# Disambiguation Engine
# -----------------------------------------------------------------------------
def _normalize_token(t: str) -> str:
    return t.strip().lower()


def _is_distractor_mention(candidate_name: str, context_tokens: List[str]) -> bool:
    """
    Checks if a candidate name is used as part of a street or local landmark name
    (e.g., 'Comilla' in 'Comilla Para' or 'Noakhali' in 'Noakhali Road').
    """
    c_lower = candidate_name.lower()
    for idx, token in enumerate(context_tokens):
        if token.lower() == c_lower:
            # Check next token for road/para/bazar distractor suffix
            if idx + 1 < len(context_tokens):
                next_tok = context_tokens[idx + 1].lower()
                if next_tok in DISTRACTOR_SUFFIXES:
                    return True
    return False


def resolve_location_hierarchy(
    candidate_spans: List[str],
    full_context_tokens: List[str],
    gazetteer: Optional[BangladeshGazetteer] = None,
    matcher: Optional[PhoneticGazetteerMatcher] = None,
) -> Dict[str, Any]:
    """
    Resolves ambiguous administrative candidate spans into the single most confident
    administrative node hierarchy across Bangladesh.

    Args:
      candidate_spans: Extracted spans representing potential administrative locations.
      full_context_tokens: All tokens from normalized address string.
      gazetteer: Optional pre-loaded BangladeshGazetteer instance.
      matcher: Optional pre-loaded PhoneticGazetteerMatcher instance.

    Returns:
      {
        "resolved_node": AdminNode,
        "confidence": float (0.0 to 1.0),
        "hierarchy": Dict[str, str],
        "hierarchy_bn": Dict[str, str],
        "matched_spans": List[str],
        "score": float,
        "candidates_ranked": List[Dict[str, Any]],
      }
    """
    if gazetteer is None:
        gazetteer = BangladeshGazetteer()
    if matcher is None:
        matcher = PhoneticGazetteerMatcher(gazetteer)

    if not candidate_spans:
        # Fallback to scanning context tokens
        candidate_spans = full_context_tokens

    # Normalize context tokens for fast membership testing
    normalized_context = [_normalize_token(t) for t in full_context_tokens if t.strip()]
    context_set = set(normalized_context)
    context_string = " ".join(normalized_context)

    # 1. Gather all potential AdminNode candidates from candidate spans
    candidates_pool: Dict[str, AdminNode] = {}
    span_matches_map: Dict[str, List[str]] = {}

    for span in candidate_spans:
        clean_span = span.strip()
        if not clean_span:
            continue

        # Check exact matches first
        exacts = gazetteer.find_by_exact_name(clean_span)
        if exacts:
            for node in exacts:
                candidates_pool[node.id] = node
                span_matches_map.setdefault(node.id, []).append(clean_span)
        else:
            # Phonetic search
            phonetic_matches = matcher.search(clean_span, max_results=5)
            for node, score in phonetic_matches:
                if score >= 0.5:
                    candidates_pool[node.id] = node
                    span_matches_map.setdefault(node.id, []).append(clean_span)

    if not candidates_pool:
        # No administrative candidates found
        return {
            "resolved_node": None,
            "confidence": 0.0,
            "hierarchy": {},
            "hierarchy_bn": {},
            "matched_spans": [],
            "score": 0.0,
            "candidates_ranked": [],
        }

    # 2. Score each candidate against full address context
    candidate_scores: List[Dict[str, Any]] = []

    # Check for postcode presence in context
    found_postcode = None
    for token in full_context_tokens:
        clean_t = token.strip()
        if clean_t.isdigit() and len(clean_t) == 4 and clean_t in POSTCODE_TO_ADMIN:
            found_postcode = clean_t
            break

    # Check for metro block indicators
    has_block = any(b in context_set for b in ("block", "blk", "ব্লক"))

    for node_id, node in candidates_pool.items():
        score = 0.0
        reasons: List[str] = []

        # (a) Base matching score
        node_name_clean = node.name.lower()
        spans_lower = [s.strip().lower() for s in candidate_spans if s.strip()]

        is_exact = (
            node_name_clean in context_set
            or (node.bn_name and node.bn_name in context_set)
            or any(s == node_name_clean or (node.bn_name and s == node.bn_name.lower()) for s in spans_lower)
        )
        if is_exact:
            score += 15.0
            reasons.append("exact_name_match")
        else:
            # Scaled phonetic/fuzzy match: calculate minimum edit distance to any candidate span
            min_dist = min(
                min(
                    levenshtein_distance(span_l, node_name_clean),
                    levenshtein_distance(span_l, node.bn_name.lower()) if node.bn_name else 999,
                )
                for span_l in spans_lower
            ) if spans_lower else 2
            match_score = max(4.0, 14.0 - (min_dist * 2.0))
            score += match_score
            reasons.append(f"phonetic_name_match_dist_{min_dist}")

        # Specificity weight: deeper administrative tiers carry higher base specificity
        tier_weight = {
            AdminLevel.SUBAREA: 6.0,
            AdminLevel.UNION: 5.0,
            AdminLevel.THANA: 5.0,
            AdminLevel.UPAZILA: 4.0,
            AdminLevel.DISTRICT: 2.0,
            AdminLevel.DIVISION: 1.0,
        }.get(node.level, 2.0)
        score += tier_weight
        reasons.append(f"tier_weight_{node.level.value}")

        # (b) Parent District Co-occurrence Boost
        dist = node.district
        if dist:
            dist_name = dist.name.lower()
            if dist_name in context_set or dist.bn_name in context_set:
                score += 25.0
                reasons.append(f"parent_district_cooccurrence_{dist.name}")

        # (c) Parent Division Co-occurrence Boost
        div = node.division
        if div:
            div_name = div.name.lower()
            if div_name in context_set or div.bn_name in context_set:
                score += 15.0
                reasons.append(f"parent_division_cooccurrence_{div.name}")

        # (d) Parent Upazila/Thana Co-occurrence Boost (if candidate is a union/subarea)
        upz = node.upazila
        if upz and node != upz:
            upz_name = upz.name.lower()
            if upz_name in context_set or upz.bn_name in context_set:
                score += 30.0
                reasons.append(f"parent_upazila_cooccurrence_{upz.name}")

        # (e) Child Subarea/Union Co-occurrence Boost (if candidate is Upazila/Thana)
        if node.level.is_upazila_level:
            for child in node.children:
                child_name = child.name.lower()
                if child_name in context_string or (child.bn_name and child.bn_name in context_string):
                    score += 20.0
                    reasons.append(f"child_subarea_cooccurrence_{child.name}")
                    break

        # (f) Postal Code Boost
        if found_postcode:
            post_info = POSTCODE_TO_ADMIN[found_postcode]
            p_upz = post_info.get("upazila")
            p_dist = post_info.get("district")

            matches_upz = bool(p_upz and upz and p_upz.lower() == upz.name.lower())
            matches_dist = bool(p_dist and dist and p_dist.lower() == dist.name.lower())

            if matches_upz and (not p_dist or matches_dist):
                score += 40.0
                reasons.append(f"postcode_match_upazila_{found_postcode}")
            elif matches_dist:
                score += 25.0
                reasons.append(f"postcode_match_district_{found_postcode}")


        # (g) Block/Sector Context (ensuring Chattogram & Sylhet block support)
        if has_block:
            # If candidate is Halishahar / Chandgaon in Chattogram
            if upz and upz.name in ("Halishahar", "Chandgaon"):
                score += 15.0
                reasons.append("chattogram_block_boost")
            # If candidate is Shahjalal Upashahar in Sylhet
            elif upz and "upashahar" in upz.name.lower():
                score += 15.0
                reasons.append("sylhet_block_boost")

        # (h) Distractor Road Penalty
        if _is_distractor_mention(node.name, full_context_tokens):
            # Check if there is another candidate with parent co-occurrence
            has_other_supported = any(
                n.id != node.id and n.district and (n.district.name.lower() in context_set)
                for n in candidates_pool.values()
            )
            if has_other_supported:
                score -= 35.0
                reasons.append(f"distractor_road_penalty_{node.name}")

        candidate_scores.append({
            "node": node,
            "score": score,
            "reasons": reasons,
            "spans": span_matches_map.get(node.id, []),
        })

    # Sort candidates by score descending
    candidate_scores.sort(key=lambda x: x["score"], reverse=True)
    best = candidate_scores[0]
    best_node: AdminNode = best["node"]

    # Calculate confidence score (normalized between 0.0 and 1.0)
    top_score = best["score"]
    second_score = candidate_scores[1]["score"] if len(candidate_scores) > 1 else 0.0
    gap = top_score - second_score
    confidence = min(1.0, max(0.5, 0.5 + (gap / 40.0))) if top_score > 0 else 0.0

    return {
        "resolved_node": best_node,
        "confidence": round(confidence, 3),
        "hierarchy": best_node.hierarchy_dict,
        "hierarchy_bn": best_node.hierarchy_bn_dict,
        "matched_spans": best["spans"],
        "score": best["score"],
        "candidates_ranked": [
            {
                "id": c["node"].id,
                "name": c["node"].name,
                "level": c["node"].level.value,
                "district": c["node"].district.name if c["node"].district else None,
                "score": c["score"],
                "reasons": c["reasons"],
            }
            for c in candidate_scores
        ],
    }
