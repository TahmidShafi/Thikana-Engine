"""
src/normalizer.py

Stage-1 Deterministic Script, Unicode, and Numeral Normalizer for Thikana-Engine.
Performs:
  1. Bangla Unicode Canonicalization (NFC, decomposed nuktas, composite vowels, zero-width stripping)
  2. Numeral Normalization (Bangla digits to ASCII, anchor-adjacent Banglish number words)
  3. Anchor Token Standardization (House, Road, Block, Sector, Floor, Landmark Relations) with span tracking
  4. Ultra-fast sub-0.5ms deterministic execution
"""

from __future__ import annotations
import re
import time
import unicodedata
from typing import Any, Dict, List, Pattern, Tuple

# -----------------------------------------------------------------------------
# Translation Tables & Unicode Canonicalization Maps
# -----------------------------------------------------------------------------

# Decomposed Nuktas -> Precomposed Bangla Characters
# Note: Python's standard unicodedata.normalize('NFC') does NOT compose Bengali nuktas!
NUKTA_MAPPINGS: Tuple[Tuple[str, str], ...] = (
    ("\u09AF\u09BC", "\u09DF"),  # য + ় -> য় (BENGALI LETTER YA)
    ("\u09A1\u09BC", "\u09DC"),  # ড + ় -> ড় (BENGALI LETTER RRA)
    ("\u09A2\u09BC", "\u09DD"),  # ঢ + ় -> ঢ় (BENGALI LETTER RHA)
)

# Composite Bangla Vowels & Legacy Font Encodings
COMPOSITE_VOWEL_MAPPINGS: Tuple[Tuple[str, str], ...] = (
    ("\u09C7\u09BE", "\u09CB"),      # ে + া -> ো (BENGALI VOWEL SIGN O)
    ("\u09C7\u09D7", "\u09CC"),      # ে + ৌ length mark -> ৌ (BENGALI VOWEL SIGN AU)
    ("\u09A4\u09CD\u200D", "\u09CE"),  # ত + ্ + ZWJ -> ৎ (Khanda Ta legacy)
)

# Single-pass C-level translation table for:
# 1. Bangla digits ০-৯ -> ASCII 0-9
# 2. Stripping zero-width characters: ZWNJ (\u200C), ZWJ (\u200D), ZWSP (\u200B), BOM (\uFEFF)
FAST_TRANSLATE_TABLE = str.maketrans({
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
    "\u200C": None,  # Zero-width non-joiner
    "\u200D": None,  # Zero-width joiner
    "\u200B": None,  # Zero-width space
    "\uFEFF": None,  # Byte order mark
})

# -----------------------------------------------------------------------------
# Numeral Word Dictionaries & Compiled Patterns
# -----------------------------------------------------------------------------

BANGLISH_NUMERALS: Dict[str, str] = {
    "ek": "1", "dui": "2", "tin": "3", "char": "4",
    "pach": "5", "paach": "5", "choy": "6", "shat": "7",
    "saat": "7", "aat": "8", "at": "8", "noy": "9",
    "dosh": "10", "egaro": "11", "baro": "12",
    # Common Bengali numeral words
    "এক": "1", "দুই": "2", "তিন": "3", "চার": "4",
    "পাঁচ": "5", "ছয়": "6", "ছয়": "6", "সাত": "7",
    "আট": "8", "নয়": "9", "নয়": "9", "দশ": "10",
    "এগারো": "11", "বারো": "12",
}

_NUM_WORDS_PATTERN = "|".join(re.escape(w) for w in BANGLISH_NUMERALS.keys())

_ANCHOR_TERMS = (
    r"road|rd|rasta|goli|lane|avenue|ave|street|st|"
    r"house|h#|basha|basa|bari|holding|"
    r"sector|sec|block|blk|"
    r"lift|floor|flr|tola|level|lvl|flat|apt|apartment|unit|"
    r"রোড|রাস্তা|গলি|সড়ক|সড়ক|লেন|"
    r"বাসা|বাড়ি|বাড়ি|হোল্ডিং|"
    r"সেক্টর|ব্লক|লিফটের|লিফট|তলা|ফ্লোর|লেভেল|ফ্ল্যাট"
)

# Anchor followed by number word (e.g. "Road char", "Rd dui", "রোড চার")
RE_ANCHOR_THEN_NUM = re.compile(
    rf"(?i)(?<![^\s,./\-#:])({_ANCHOR_TERMS})"
    rf"(?:\s*(?:no\.?|নং|number|#))?([\s:\-]+)({_NUM_WORDS_PATTERN})(?![^\s,./\-#:])"
)

# Number word followed by anchor (e.g. "tin tola", "char floor", "৩ তলা")
RE_NUM_THEN_ANCHOR = re.compile(
    rf"(?i)(?<![^\s,./\-#:])({_NUM_WORDS_PATTERN})"
    rf"([\s:\-]+(?:no\.?|নং|th|st|nd|rd|number)?[\s:\-]*)({_ANCHOR_TERMS})(?![^\s,./\-#:])"
)

# -----------------------------------------------------------------------------
# Anchor Standardization Rules
# -----------------------------------------------------------------------------

# Ordered list of (canonical_token, regex_pattern, token_type)
# Order matters: more specific compounds first
ANCHOR_SPECS: List[Tuple[str, Pattern[str], str]] = [
    # Landmark relations (compound phrases first)
    (
        "REL_NEAR",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:er\s+pashe|er\s+pase|এর\s+পাশে|সংলগ্ন|near|beside|adjacent|pashe|পাশে)(?![^\s,./\-#:])"
        ),
        "LANDMARK_RELATION",
    ),
    (
        "REL_BEHIND",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:er\s+pichone|er\s+piche|এর\s+পিছনে|behind|pichone|পিছনে|পেছনে)(?![^\s,./\-#:])"
        ),
        "LANDMARK_RELATION",
    ),
    (
        "REL_FRONT",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:er\s+samne|এর\s+সামনে|in\s+front\s+of|opposite(?:\s+to)?|samne|সামনে|বিপরীতে)(?![^\s,./\-#:])"
        ),
        "LANDMARK_RELATION",
    ),
    # House markers
    (
        "H#",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:house|basha|basa|bari|holding|h#|বাসা|বাড়ি|বাড়ি|হোল্ডিং)(?:\s*(?:no\.?|নং|number|#))?(?![^\s,./\-#:])"
        ),
        "HOUSE",
    ),
    # Road markers
    (
        "RD#",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:road|rd\.?|rasta|goli|lane|avenue|ave|street|st|রোড|রাস্তা|গলি|সড়ক|সড়ক|লেন)(?:\s*(?:no\.?|নং|number|#))?(?![^\s,./\-#:])"
        ),
        "ROAD",
    ),
    # Sector markers
    (
        "SEC#",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:sector|sec\.?|সেক্টর)(?:\s*(?:no\.?|নং|number|#))?(?![^\s,./\-#:])"
        ),
        "SECTOR",
    ),
    # Block markers
    (
        "BLK#",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:block|blk\.?|ব্লক)(?:\s*(?:no\.?|নং|number|#))?(?![^\s,./\-#:])"
        ),
        "BLOCK",
    ),
    # Floor markers
    (
        "FL#",
        re.compile(
            r"(?i)(?<![^\s,./\-#:])(?:tola|flr\.?|floor|lift|level|lvl|flat|apt\.?|apartment|unit|তলা|লিফটের|লিফট|ফ্লোর|লেভেল|ফ্ল্যাট)(?:\s*(?:no\.?|নং|number|#))?(?![^\s,./\-#:])"
        ),
        "FLOOR",
    ),
]

# Whitespace normalization pattern
RE_MULTI_SPACES = re.compile(r"[ \t]+")
RE_SPACE_PUNCT = re.compile(r"\s+([,./\-])")
RE_TOKENIZE = re.compile(r"[A-Za-z0-9_#/\-]+|[\u0980-\u09FF]+|[^\s\w,./\-]+")


# -----------------------------------------------------------------------------
# Core Normalization Functions
# -----------------------------------------------------------------------------

def canonicalize_bangla_unicode(text: str) -> str:
    """
    Canonicalizes Bangla Unicode:
      - Normalizes to NFC form.
      - Replaces decomposed Nukta sequences with single precomposed codepoints (য+় -> য়, etc.).
      - Normalizes composite vowel signs (ে+া -> ো, etc.).
      - Strips zero-width characters (ZWNJ, ZWJ, ZWSP, BOM).
    """
    if not text:
        return ""

    # Step 1: Standard NFC normalization
    out = unicodedata.normalize("NFC", text)

    # Step 2: Decomposed Nukta substitutions (Python NFC does not compose these)
    for decomposed, precomposed in NUKTA_MAPPINGS:
        if decomposed in out:
            out = out.replace(decomposed, precomposed)

    # Step 3: Composite vowel signs and legacy Khanda Ta
    for comp, canon in COMPOSITE_VOWEL_MAPPINGS:
        if comp in out:
            out = out.replace(comp, canon)

    return out


def normalize_numerals(text: str) -> str:
    """
    Normalizes numerals:
      - Converts all Bangla digits ০-৯ to ASCII 0-9.
      - Converts anchor-adjacent Banglish and Bangla number words to ASCII digits.
      - Strips zero-width characters in the same fast C-level translate pass.
    """
    if not text:
        return ""

    # Fast translation for Bangla digits + zero-width strip
    out = text.translate(FAST_TRANSLATE_TABLE)

    # Convert anchor-adjacent number words:
    # 1. Anchor then Number Word (e.g. "Road char" -> "Road 4")
    def _repl_after(m: re.Match[str]) -> str:
        anchor_part = m.group(1)
        sep = m.group(2)
        word = m.group(3).lower()
        digit = BANGLISH_NUMERALS.get(word, m.group(3))
        # Keep original anchor token text and replace just the word
        full_match = m.group(0)
        return full_match[: m.start(3) - m.start(0)] + digit

    out = RE_ANCHOR_THEN_NUM.sub(_repl_after, out)

    # 2. Number Word then Anchor (e.g. "tin tola" -> "3 tola")
    def _repl_before(m: re.Match[str]) -> str:
        word = m.group(1).lower()
        sep = m.group(2)
        anchor_part = m.group(3)
        digit = BANGLISH_NUMERALS.get(word, m.group(1))
        return digit + sep + anchor_part

    out = RE_NUM_THEN_ANCHOR.sub(_repl_before, out)

    return out


def standardize_anchors(text: str, raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Standardizes structural address anchors into canonical tags:
      - House -> H#
      - Road -> RD#
      - Sector -> SEC#
      - Block -> BLK#
      - Floor -> FL#
      - Landmark Relations -> REL_NEAR, REL_BEHIND, REL_FRONT
    Preserves original span offsets from raw_text.
    """
    spans: List[Dict[str, Any]] = []

    # Track spans from the raw input text
    for canonical_token, pattern, token_type in ANCHOR_SPECS:
        for match in pattern.finditer(raw_text):
            spans.append({
                "token": canonical_token,
                "raw": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "type": token_type,
            })

    # Sort spans by starting index in original text
    spans.sort(key=lambda s: s["start"])

    # Perform substitutions on the normalized text
    out = text
    for canonical_token, pattern, _ in ANCHOR_SPECS:
        out = pattern.sub(canonical_token, out)

    # Clean up excess whitespace
    out = RE_MULTI_SPACES.sub(" ", out).strip()
    out = RE_SPACE_PUNCT.sub(r"\1", out)

    return out, spans


def tokenize(text: str) -> List[str]:
    """
    Extracts high-level normalized tokens from standardized address text.
    """
    if not text:
        return []
    return RE_TOKENIZE.findall(text)


def normalize_address(raw_text: str) -> Dict[str, Any]:
    """
    Primary pipeline function:
    Takes raw messy address text and normalizes script, Unicode, numerals,
    and structural anchors in under 0.5 milliseconds.

    Returns:
      {
        "raw_text": str,
        "normalized_text": str,
        "tokens": List[str],
        "spans": List[Dict[str, Any]],
        "latency_ms": float
      }
    """
    start_time = time.perf_counter()

    if not raw_text or not raw_text.strip():
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "raw_text": raw_text or "",
            "normalized_text": "",
            "tokens": [],
            "spans": [],
            "latency_ms": latency_ms,
        }

    # Step 1: Bangla Unicode Canonicalization
    text = canonicalize_bangla_unicode(raw_text)

    # Step 2: Numeral Normalization
    text = normalize_numerals(text)

    # Step 3: Anchor Token Standardization & Span Tracking
    normalized_text, spans = standardize_anchors(text, raw_text)

    # Step 4: Token Extraction
    tokens = tokenize(normalized_text)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "spans": spans,
        "latency_ms": latency_ms,
    }
