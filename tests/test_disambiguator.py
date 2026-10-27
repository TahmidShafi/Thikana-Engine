"""
tests/test_disambiguator.py

Unit tests for Step 3 of Thikana-Engine:
  1. banglish_phonetic_key (Mirpur variants, Badda vs Boda, aspirations, swaps, skeletons)
  2. PhoneticTrie & PhoneticGazetteerMatcher (Trie prefix, exact, fuzzy, MeasureCollisionRate)
  3. Context-aware Disambiguator (resolve_location_hierarchy):
     - Homonyms (Mirpur in Dhaka vs. Kushtia, Mohammadpur in Dhaka vs. Magura vs. Noakhali)
     - Postal code boost (1216 -> Dhaka Mirpur, 1207 -> Dhaka Mohammadpur, 7030 -> Kushtia Mirpur)
     - Block/Sector neutrality (Halishahar Block G -> Chattogram, Shahjalal Upashahar Block B -> Sylhet)
     - Distractor road names (Comilla Para, Badda, Dhaka -> Badda, Dhaka; Noakhali Road, Mirpur 10, Dhaka -> Mirpur, Dhaka)
     - Typo resilience (Merpur, Dhanmundi, Uttora)
"""

import pytest
from src.gazetteer import BangladeshGazetteer
from src.phonetic_trie import (
    banglish_phonetic_key,
    levenshtein_distance,
    PhoneticTrie,
    PhoneticGazetteerMatcher,
)
from src.disambiguator import resolve_location_hierarchy


@pytest.fixture(scope="module")
def gazetteer() -> BangladeshGazetteer:
    return BangladeshGazetteer()


@pytest.fixture(scope="module")
def matcher(gazetteer: BangladeshGazetteer) -> PhoneticGazetteerMatcher:
    return PhoneticGazetteerMatcher(gazetteer)


# =============================================================================
# 1. Phonetic Key Encoder Tests
# =============================================================================
def test_phonetic_key_mirpur_variants():
    """Verify that 'Mirpur', 'Meerpur', 'Mirpoor', and 'মিরপুর' all produce primary key 'MIRPUR'."""
    cases = ["Mirpur", "Meerpur", "Mirpoor", "মিরপুর"]
    for case in cases:
        pk, sk = banglish_phonetic_key(case)
        assert pk == "MIRPUR", f"Failed for {case}: got {pk}"
        assert sk == "MRPR", f"Failed skeleton for {case}: got {sk}"


def test_phonetic_key_badda_vs_boda():
    """Verify that 'Badda' and 'Boda' produce 'BADA' and are ranked by Levenshtein distance."""
    pk_badda, sk_badda = banglish_phonetic_key("Badda")
    pk_boda, sk_boda = banglish_phonetic_key("Boda")

    assert pk_badda == "BADA"
    assert pk_boda == "BADA"
    assert sk_badda == "BD"
    assert sk_boda == "BD"

    # Levenshtein distance ranking:
    # Query 'Badda' ranks Badda (dist 0) ahead of Boda (dist 2)
    dist_badda = levenshtein_distance("badda", "badda")
    dist_boda = levenshtein_distance("badda", "boda")
    assert dist_badda < dist_boda
    assert dist_badda == 0
    assert dist_boda == 2

    # Query 'Boda' ranks Boda (dist 0) ahead of Badda (dist 2)
    assert levenshtein_distance("boda", "boda") == 0
    assert levenshtein_distance("boda", "badda") == 2


def test_aspirations_and_swaps():
    """Verify consonant aspirations and common Banglish swaps."""
    # bh -> b, ph -> f
    assert banglish_phonetic_key("Bhatara")[0] == banglish_phonetic_key("Batara")[0]
    assert banglish_phonetic_key("Phultala")[0] == banglish_phonetic_key("Fultala")[0]

    # kh -> k, gh -> g
    assert banglish_phonetic_key("Khilgaon")[0] == banglish_phonetic_key("Kilgaon")[0]
    assert banglish_phonetic_key("Ghorashal")[0] == banglish_phonetic_key("Gorasal")[0]

    # ch / chh -> c
    assert banglish_phonetic_key("Chawkbazar")[0] == banglish_phonetic_key("Cawkbazar")[0]
    assert banglish_phonetic_key("Chhatak")[0] == banglish_phonetic_key("Catak")[0]

    # sh / s -> s
    assert banglish_phonetic_key("Shantinagar")[0] == banglish_phonetic_key("Santinagar")[0]

    # z / j -> j
    assert banglish_phonetic_key("Zigatola")[0] == banglish_phonetic_key("Jigatola")[0]

    # v -> b
    assert banglish_phonetic_key("Vatara")[0] == banglish_phonetic_key("Batara")[0]


def test_vowel_collapsing():
    """Verify 3-class vowel collapsing: ee/ii/i/y -> I, oo/u -> U, o/a/e -> A."""
    # ee / i -> I
    assert banglish_phonetic_key("Teejgaon")[0] == banglish_phonetic_key("Tijgaon")[0]
    # oo / u -> U
    assert banglish_phonetic_key("Uttara")[0] == banglish_phonetic_key("Oottara")[0]
    # o / a / e -> A
    assert banglish_phonetic_key("Dhanmondi")[0] == banglish_phonetic_key("Dhanmandi")[0]


# =============================================================================
# 2. Trie & Phonetic Index Tests
# =============================================================================
def test_trie_prefix_and_exact_search(matcher: PhoneticGazetteerMatcher):
    # Search exact key MIRPUR
    exact_matches = matcher.trie.search_exact("MIRPUR")
    assert len(exact_matches) >= 2
    districts = {n.district.name for n in exact_matches if n.district}
    assert "Dhaka" in districts
    assert "Kushtia" in districts

    # Search prefix DAN (phonetic key of Dhanmondi has de-aspirated D)
    prefix_matches = matcher.trie.search_prefix("DAN")
    assert any(n.name == "Dhanmondi" for n in prefix_matches)

    # Fuzzy search on Trie with distance <= 1
    fuzzy = matcher.trie.fuzzy_search("MIRPOR", max_distance=1)
    assert any(n.name == "Mirpur" for n, dist in fuzzy)



def test_measure_collision_rate(matcher: PhoneticGazetteerMatcher):
    stats = matcher.MeasureCollisionRate()
    assert stats["total_entities"] >= 5285
    assert stats["unique_primary_keys"] > 7000
    assert stats["avg_collisions_per_key"] >= 1.0
    assert stats["collision_rate_pct"] > 0.0
    assert len(stats["top_collisions"]) == 10

    # Ensure keys in top collisions contain real homonyms
    top_keys = [c["key"] for c in stats["top_collisions"]]
    assert any(k in top_keys for k in ("DURGAPUR", "MIRPUR", "DAULATPUR", "RAIPUR", "MAHAMADPUR"))


# =============================================================================
# 3. Disambiguator Tests: Homonyms
# =============================================================================
def test_homonym_mirpur_dhaka_vs_kushtia(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    # Context 1: Mirpur in Dhaka
    ctx_dhaka = ["H#", "12", "RD#", "4", "Mirpur", "Dhaka"]
    res1 = resolve_location_hierarchy(["Mirpur"], ctx_dhaka, gazetteer, matcher)
    assert res1["resolved_node"].name == "Mirpur"
    assert res1["resolved_node"].district.name == "Dhaka"
    assert res1["resolved_node"].division.name == "Dhaka"

    # Context 2: Mirpur in Kushtia
    ctx_kushtia = ["Mirpur", "Kushtia", "Khulna"]
    res2 = resolve_location_hierarchy(["Mirpur"], ctx_kushtia, gazetteer, matcher)
    assert res2["resolved_node"].name == "Mirpur"
    assert res2["resolved_node"].district.name == "Kushtia"
    assert res2["resolved_node"].division.name == "Khulna"

    # Context 3: Mirpur with postal code 1216
    ctx_postcode = ["Mirpur", "1216"]
    res3 = resolve_location_hierarchy(["Mirpur"], ctx_postcode, gazetteer, matcher)
    assert res3["resolved_node"].name == "Mirpur"
    assert res3["resolved_node"].district.name == "Dhaka"


def test_homonym_mohammadpur_three_way(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    # Context 1: Mohammadpur in Dhaka
    ctx_dhaka = ["Mohammadpur", "Dhaka"]
    res1 = resolve_location_hierarchy(["Mohammadpur"], ctx_dhaka, gazetteer, matcher)
    assert res1["resolved_node"].district.name == "Dhaka"

    # Context 2: Mohammadpur with postal code 1207 (Dhaka)
    ctx_postcode = ["Mohammadpur", "1207"]
    res2 = resolve_location_hierarchy(["Mohammadpur"], ctx_postcode, gazetteer, matcher)
    assert res2["resolved_node"].district.name == "Dhaka"

    # Context 3: Mohammadpur in Magura
    ctx_magura = ["Mohammadpur", "Magura"]
    res3 = resolve_location_hierarchy(["Mohammadpur"], ctx_magura, gazetteer, matcher)
    assert res3["resolved_node"].district.name == "Magura"

    # Context 4: Mohammadpur in Noakhali (Subarnachar)
    ctx_noakhali = ["Mohammadpur", "Subarnachar", "Noakhali"]
    res4 = resolve_location_hierarchy(["Mohammadpur"], ctx_noakhali, gazetteer, matcher)
    assert res4["resolved_node"].district.name == "Noakhali"

    # Context 5: Real-world user query - Holding 45, Tajmahal Road, Mohammadpur -> Dhaka
    ctx_tajmahal = ["Holding", "45", "Tajmahal", "Road", "Mohammadpur"]
    res5 = resolve_location_hierarchy(["Tajmahal Road", "Mohammadpur"], ctx_tajmahal, gazetteer, matcher)
    assert res5["resolved_node"].district.name == "Dhaka"
    assert res5["hierarchy"]["thana"] == "Mohammadpur"

    # Context 6: Real-world user query - Village: Mohammadpur, Upazila: Chatkhil, Noakhali -> Noakhali
    ctx_chatkhil = ["Village", "Mohammadpur", "Upazila", "Chatkhil", "Noakhali"]
    res6 = resolve_location_hierarchy(["Mohammadpur", "Chatkhil", "Noakhali"], ctx_chatkhil, gazetteer, matcher)
    assert res6["resolved_node"].district.name == "Noakhali"
    assert res6["hierarchy"]["upazila"] == "Chatkhil"


# =============================================================================
# 4. Disambiguator Tests: Block / Sector Neutrality
# =============================================================================
def test_halishahar_block_chattogram(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    """Verify that 'Block' does not assume Dhaka when Halishahar is present."""
    ctx = ["Halishahar", "Block", "G", "Chittagong"]
    res = resolve_location_hierarchy(["Halishahar", "Block G"], ctx, gazetteer, matcher)
    assert res["resolved_node"].name == "Halishahar"
    assert res["resolved_node"].district.name == "Chattogram"
    assert res["resolved_node"].division.name in ("Chattogram", "Chattagram")


def test_sylhet_upashahar_block(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    """Verify that blocks in Sylhet Upashahar link to Sylhet, not Dhaka."""
    ctx = ["Shahjalal", "Upashahar", "Block", "B", "Sylhet"]
    res = resolve_location_hierarchy(
        ["Shahjalal Upashahar", "Block B"], ctx, gazetteer, matcher
    )
    assert "Upashahar" in res["resolved_node"].name
    assert res["resolved_node"].district.name == "Sylhet"
    assert res["resolved_node"].division.name == "Sylhet"


# =============================================================================
# 5. Disambiguator Tests: Distractor Street & Neighborhood Names
# =============================================================================
def test_distractor_comilla_para(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    """
    'Comilla Para, Badda, Dhaka' -> Badda, Dhaka.
    'Comilla' is part of a local para/road name in Badda, NOT Comilla District.
    """
    spans = ["Comilla Para", "Badda", "Dhaka"]
    ctx = ["Comilla", "Para", "Badda", "Dhaka"]
    res = resolve_location_hierarchy(spans, ctx, gazetteer, matcher)

    assert res["resolved_node"].district.name == "Dhaka"
    assert res["resolved_node"].name in ("Badda", "Dhaka")
    assert res["hierarchy"]["district"] == "Dhaka"
    assert res["hierarchy"]["division"] == "Dhaka"


def test_distractor_noakhali_road(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    """
    'Noakhali Road, Mirpur 10, Dhaka' -> Mirpur, Dhaka.
    'Noakhali' is part of a road name in Mirpur, NOT Noakhali District.
    """
    spans = ["Noakhali Road", "Mirpur 10", "Dhaka"]
    ctx = ["Noakhali", "Road", "Mirpur", "10", "Dhaka"]
    res = resolve_location_hierarchy(spans, ctx, gazetteer, matcher)

    assert res["resolved_node"].district.name == "Dhaka"
    assert res["resolved_node"].name in ("Mirpur 10", "Mirpur")
    assert res["hierarchy"]["district"] == "Dhaka"


# =============================================================================
# 6. Disambiguator Tests: Spelling Typos
# =============================================================================
def test_spelling_typos_resilience(
    gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher
):
    # Typo: 'Merpur-10' -> Mirpur 10
    res1 = resolve_location_hierarchy(
        ["Merpur"], ["Merpur", "Dhaka"], gazetteer, matcher
    )
    assert res1["resolved_node"].name == "Mirpur"
    assert res1["resolved_node"].district.name == "Dhaka"

    # Typo: 'Dhanmundi' -> Dhanmondi
    res2 = resolve_location_hierarchy(
        ["Dhanmundi"], ["Dhanmundi", "Dhaka"], gazetteer, matcher
    )
    assert res2["resolved_node"].name == "Dhanmondi"
    assert res2["resolved_node"].district.name == "Dhaka"

    # Typo: 'Uttora' -> Uttara
    res3 = resolve_location_hierarchy(
        ["Uttora"], ["Uttora", "Dhaka"], gazetteer, matcher
    )
    assert res3["resolved_node"].name == "Uttara"
    assert res3["resolved_node"].district.name == "Dhaka"
