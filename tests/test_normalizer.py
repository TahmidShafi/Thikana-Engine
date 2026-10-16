"""
tests/test_normalizer.py

Unit tests and benchmarks for Stage-1 Normalizer (src/normalizer.py).
Covers:
  1. Decomposed Nukta canonicalization (য+় -> য়, ড+় -> ড়, ঢ+় -> ঢ়)
  2. Composite vowel normalization (ে+া -> ো, ে+ৌ -> ৌ)
  3. Zero-width character stripping (ZWNJ, ZWJ, ZWSP, BOM)
  4. Bangla digits conversion (০-৯ -> 0-9)
  5. Banglish number words post-anchor (Road char -> RD# 4, etc.)
  6. Banglish number words pre-anchor (tin tola -> 3 FL#, etc.)
  7. House anchor variations (basha, bari, holding, h#, বাসা, বাড়ি, হোল্ডিং -> H#)
  8. Road anchor variations (road, rd, rasta, goli, lane, রোড, রাস্তা, গলি, সড়ক -> RD#)
  9. Block and Sector anchors (block, blk, ব্লক -> BLK#; sector, sec, সেক্টর -> SEC#)
  10. Floor markers (tola, flr, floor, lift, তলা, লিফটের -> FL#)
  11. Landmark relations: near (er pashe, এর পাশে, সংলগ্ন, near -> REL_NEAR)
  12. Landmark relations: behind (er pichone, এর পিছনে, behind -> REL_BEHIND)
  13. Landmark relations: front (er samne, এর সামনে, opposite -> REL_FRONT)
  14. Complex multilingual messy address end-to-end (normalized_text, tokens, spans)
  15. Latency benchmark (< 0.5ms average per address, < 1.0ms strict assertion)
"""

import time
import pytest
from src.normalizer import (
    canonicalize_bangla_unicode,
    normalize_numerals,
    normalize_address,
    tokenize,
)


# =============================================================================
# 1. Bangla Unicode Canonicalization
# =============================================================================
def test_decomposed_nukta_canonicalization():
    """Verify decomposed nuktas (base + nukta) are converted to precomposed characters."""
    # \u09a1\u09bc (ড + ়) -> \u09dc (ড়)
    raw_rra = "বা\u09a1\u09bcি"
    canon_rra = canonicalize_bangla_unicode(raw_rra)
    assert "\u09dc" in canon_rra
    assert "\u09bc" not in canon_rra
    # Precomposed: \u09ac (ব) + \u09be (া) + \u09dc (ড়) + \u09bf (ি)
    assert canon_rra == "\u09ac\u09be\u09dc\u09bf"

    # \u09af\u09bc (য + ়) -> \u09df (য়)
    raw_ya = "জায়গা\u09af\u09bc"
    canon_ya = canonicalize_bangla_unicode(raw_ya)
    assert "\u09df" in canon_ya
    assert "\u09bc" not in canon_ya
    assert canon_ya.endswith("\u09df")

    # \u09a2\u09bc (ঢ + ়) -> \u09dd (ঢ়)
    raw_rha = "আষা\u09a2\u09bc"
    canon_rha = canonicalize_bangla_unicode(raw_rha)
    assert "\u09dd" in canon_rha
    assert "\u09bc" not in canon_rha
    assert canon_rha == "আষা\u09dd"


def test_composite_vowel_canonicalization():
    """Verify composite vowel sequences are canonicalized."""
    # \u09c7\u09be (ে + া) -> \u09cb (ো)
    # 'প' (\u09aa) + 'ে' (\u09c7) + 'া' (\u09be) -> 'পো' (\u09aa\u09cb)
    raw_o = "প\u09c7\u09beস্ট অফিস"
    canon_o = canonicalize_bangla_unicode(raw_o)
    assert "\u09cb" in canon_o
    assert "\u09c7\u09be" not in canon_o
    assert canon_o == "পোস্ট অফিস"

    # \u09c7\u09d7 (ে + ৌ length mark) -> \u09cc (ৌ)
    raw_au = "ম\u09c7\u09d7লভীবাজার"
    canon_au = canonicalize_bangla_unicode(raw_au)
    assert "\u09cc" in canon_au
    assert "\u09c7\u09d7" not in canon_au
    assert canon_au == "মৌলভীবাজার"



def test_zero_width_characters_stripping():
    """Verify zero-width joiners, non-joiners, zero-width spaces, and BOM are stripped."""
    # Contains ZWNJ (\u200C), ZWJ (\u200D), ZWSP (\u200B), and BOM (\uFEFF)
    raw = "\uFEFFবাসা\u200C ১২, রোড\u200D ৫, ধানমন্ডি\u200B ৩২"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    for zw in ["\u200C", "\u200D", "\u200B", "\uFEFF"]:
        assert zw not in norm
    assert "H# 12" in norm
    assert "RD# 5" in norm
    assert "32" in norm


# =============================================================================
# 2. Numeral Normalization
# =============================================================================
def test_bangla_digits_conversion():
    """Verify all Bengali digits (০-৯) are converted to ASCII (0-9)."""
    raw = "বাসা ১২, রোড ৩৪, সেক্টর ৭, ফ্ল্যাট ৯, কোড ০"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    for bangla_digit in "০১২৩৪৫৬৭৮৯":
        assert bangla_digit not in norm
    assert "12" in norm
    assert "34" in norm
    assert "7" in norm
    assert "9" in norm
    assert "0" in norm


def test_banglish_number_words_post_anchor():
    """Verify Banglish number words after address anchors are converted to digits."""
    raw = "Road char, Sector tin, Block dui, House dosh, goli aat, holding baro"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "RD# 4" in norm
    assert "SEC# 3" in norm
    assert "BLK# 2" in norm
    assert "H# 10" in norm
    assert "RD# 8" in norm
    assert "H# 12" in norm


def test_banglish_number_words_pre_anchor():
    """Verify Banglish and Bangla number words before anchors are converted to digits."""
    raw = "tin tola, char floor, egaro no road, তিন তলা"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "3 FL#" in norm
    assert "4 FL#" in norm
    assert "11 no RD#" in norm or "11 RD#" in norm


# =============================================================================
# 3. Anchor Token Standardization
# =============================================================================
def test_house_anchor_variations():
    """Verify all house markers canonicalize to H#."""
    raw = "basha 5, bari 7, holding 9, h# 11, বাসা 12, বাড়ি 15, হোল্ডিং 20"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert norm == "H# 5, H# 7, H# 9, H# 11, H# 12, H# 15, H# 20"


def test_road_anchor_variations():
    """Verify all road markers canonicalize to RD#."""
    raw = "road 1, rd 2, rasta 3, goli 4, lane 5, রোড 6, রাস্তা 7, গলি 8, সড়ক 9"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert norm == "RD# 1, RD# 2, RD# 3, RD# 4, RD# 5, RD# 6, RD# 7, RD# 8, RD# 9"


def test_block_and_sector_anchors():
    """Verify block and sector markers canonicalize to BLK# and SEC#."""
    raw = "block a, blk b, ব্লক c, sector 3, sec 4, সেক্টর 5"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert norm == "BLK# a, BLK# b, BLK# c, SEC# 3, SEC# 4, SEC# 5"


def test_floor_markers():
    """Verify floor and lift markers canonicalize to FL#."""
    raw = "3 tola, 4th floor, flr 2, lift 5, ৩ তলা, লিফটের 4"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "3 FL#" in norm
    assert "4th FL#" in norm
    assert "FL# 2" in norm
    assert "FL# 5" in norm
    assert "FL# 4" in norm


# =============================================================================
# 4. Landmark Relations
# =============================================================================
def test_landmark_relation_near():
    """Verify near landmark relations canonicalize to REL_NEAR."""
    raw = "school er pashe, বাজার এর পাশে, মসজিদ সংলগ্ন, near post office"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "school REL_NEAR" in norm
    assert "বাজার REL_NEAR" in norm
    assert "মসজিদ REL_NEAR" in norm
    assert "REL_NEAR post office" in norm


def test_landmark_relation_behind():
    """Verify behind landmark relations canonicalize to REL_BEHIND."""
    raw = "hospital er pichone, ব্যাংকের এর পিছনে, behind park"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "hospital REL_BEHIND" in norm
    assert "ব্যাংকের REL_BEHIND" in norm
    assert "REL_BEHIND park" in norm


def test_landmark_relation_front():
    """Verify front/opposite landmark relations canonicalize to REL_FRONT."""
    raw = "station er samne, সুপারমার্কেট এর সামনে, opposite clinic"
    res = normalize_address(raw)
    norm = res["normalized_text"]
    assert "station REL_FRONT" in norm
    assert "সুপারমার্কেট REL_FRONT" in norm
    assert "REL_FRONT clinic" in norm


# =============================================================================
# 5. Complex End-to-End Multilingual Address & Span Verification
# =============================================================================
def test_complex_multilingual_messy_address():
    """Verify complex mixed address with Bangla, Banglish, decomposed nuktas, and spans."""
    raw = "বা\u09a1\u09bcি # ১২/এ, Rd চার, Block বি, ৩ তলা, ধানমন্ডি ২৭, লেক এর পাশে, ঢাকা"
    res = normalize_address(raw)

    assert res["raw_text"] == raw
    norm = res["normalized_text"]
    assert "H# 12/এ" in norm
    assert "RD# 4" in norm
    assert "BLK# বি" in norm
    assert "3 FL#" in norm
    assert "ধানমন্ডি 27" in norm
    assert "লেক REL_NEAR" in norm
    assert "ঢাকা" in norm

    # Tokens check
    tokens = res["tokens"]
    assert "H#" in tokens
    assert "RD#" in tokens
    assert "BLK#" in tokens
    assert "FL#" in tokens
    assert "REL_NEAR" in tokens

    # Spans check
    spans = res["spans"]
    assert len(spans) >= 5
    span_types = {s["type"] for s in spans}
    assert "HOUSE" in span_types
    assert "ROAD" in span_types
    assert "BLOCK" in span_types
    assert "FLOOR" in span_types
    assert "LANDMARK_RELATION" in span_types

    # Verify span offsets reference raw_text accurately
    for span in spans:
        start, end = span["start"], span["end"]
        raw_slice = raw[start:end]
        assert raw_slice == span["raw"]


# =============================================================================
# 6. Latency & Speed Benchmark
# =============================================================================
def test_latency_benchmark_under_half_millisecond():
    """
    Benchmark test asserting:
      - Mean execution latency over 1,000 iterations is < 0.5 ms.
      - Individual call latency assertion is < 1.0 ms.
    """
    test_addresses = [
        "বা\u09a1\u09bcি # ১২, Rd চার, Block বি, ৩ তলা, ধানমন্ডি ২৭, লেক এর পাশে, ঢাকা",
        "basha 5, bari 7, holding 9, h# 11, বাসা 12, বাড়ি 15, হোল্ডিং 20",
        "road 1, rd 2, rasta 3, goli 4, lane 5, রোড 6, রাস্তা 7, গলি 8, সড়ক 9",
        "block a, blk b, ব্লক c, sector 3, sec 4, সেক্টর 5",
        "3 tola, 4th floor, flr 2, lift 5, ৩ তলা, লিফটের 4",
        "school er pashe, বাজার এর পাশে, মসজিদ সংলগ্ন, near post office",
        "hospital er pichone, ব্যাংকের এর পিছনে, behind park",
        "station er samne, সুপারমার্কেট এর সামনে, opposite to clinic",
        "House # 14/A, Road # tin, Banani, ৩ তলা",
        "Holding 55, goli dui, dhanmondi 27 er pashe",
    ]

    # Warmup
    for addr in test_addresses:
        res = normalize_address(addr)
        assert res["latency_ms"] < 1.0, f"Single call exceeded 1.0ms: {res['latency_ms']:.4f}ms"

    # Benchmark 1,000 iterations
    iterations = 1000
    t0 = time.perf_counter()
    for i in range(iterations):
        normalize_address(test_addresses[i % len(test_addresses)])
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / iterations) * 1000.0

    print(f"\n[BENCHMARK] Average latency per address over {iterations} runs: {avg_ms:.4f} ms")
    assert avg_ms < 0.5, f"Average latency {avg_ms:.4f}ms exceeded the 0.5ms threshold!"
