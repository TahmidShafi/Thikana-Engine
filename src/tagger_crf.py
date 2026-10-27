"""
src/tagger_crf.py

Stage-2 Linear-Chain Conditional Random Field (CRF) Sequence Tagger for Thikana-Engine.
Labels normalized address tokens into 9 BIO categories:
  - HOUSE
  - FLOOR_FLAT
  - ROAD
  - BLOCK_SECTOR
  - LANDMARK
  - LANDMARK_REL
  - SUBAREA
  - THANA_UPAZILA
  - DISTRICT
  - O (Non-entity background)

Extracts rich morphological prefixes/suffixes, script type, Stage-1 anchor indicators,
window context (-2 to +2), and gazetteer phonetic lookup features.
"""

from __future__ import annotations
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import joblib
import sklearn_crfsuite

from .models import AdminLevel, AdminNode
from .gazetteer import BangladeshGazetteer
from .normalizer import normalize_address
from .phonetic_trie import PhoneticGazetteerMatcher, banglish_phonetic_key


# -----------------------------------------------------------------------------
# 9 Canonical BIO Entity Types
# -----------------------------------------------------------------------------
ENTITY_TYPES = [
    "HOUSE",
    "FLOOR_FLAT",
    "ROAD",
    "BLOCK_SECTOR",
    "LANDMARK",
    "LANDMARK_REL",
    "SUBAREA",
    "THANA_UPAZILA",
    "DISTRICT",
]

BIO_LABELS = ["O"]
for et in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{et}")
    BIO_LABELS.append(f"I-{et}")

RE_BANGLA = re.compile(r"[\u0980-\u09FF]")
RE_DIGIT = re.compile(r"^\d+$")
RE_ASCII_ALPHA = re.compile(r"^[A-Za-z]+$")

STAGE1_ANCHORS = {
    "H#": "house",
    "RD#": "road",
    "BLK#": "block",
    "SEC#": "sector",
    "FL#": "floor",
    "REL_NEAR": "rel",
    "REL_BEHIND": "rel",
    "REL_FRONT": "rel",
}

# Curated POI and Commercial/Institutional Head Noun Lexicon
POI_HEAD_NOUNS: Set[str] = {
    "bank", "limited", "ltd", "tower", "bhaban", "plaza", "mall", "market",
    "bazar", "kacha", "masjid", "mosque", "mondir", "mandir", "church", "hospital",
    "clinic", "pharmacy", "medical", "school", "college", "university", "varsity",
    "bhandar", "mistanna", "restaurant", "hotel", "hall", "cinema", "park",
    "square", "stadium", "station", "terminal", "complex", "center", "centre",
    "ভবন", "ব্যাংক", "টাওয়ার", "প্লাজা", "মার্কেট", "বাজার", "মসজিদ", "হাসপাতাল",
    "ক্লিনিক", "স্কুল", "কলেজ", "বিশ্ববিদ্যালয়", "ভাণ্ডার", "হোটেল", "মোড়",
}


# -----------------------------------------------------------------------------
# Feature Extraction
# -----------------------------------------------------------------------------
def get_script_type(token: str) -> str:
    """Classifies token into script categories."""
    if RE_DIGIT.match(token):
        return "digit"
    if RE_BANGLA.search(token):
        return "bangla"
    if RE_ASCII_ALPHA.match(token):
        return "latin"
    return "mixed_or_punct"


def extract_token_features(
    tokens: List[str],
    index: int,
    matcher: Optional[PhoneticGazetteerMatcher] = None,
) -> Dict[str, Any]:
    """
    Extracts rich syntactic, morphological, anchor, and gazetteer features
    for tokens[index] within surrounding window [-2, +2].
    """
    token = tokens[index]
    token_lower = token.lower()
    script = get_script_type(token)

    features: Dict[str, Any] = {
        "bias": 1.0,
        "token.lower": token_lower,
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.len": len(token),
        "script": script,
        "is_poi_noun": token_lower in POI_HEAD_NOUNS,
    }

    # Morphological prefixes and suffixes (lengths 2, 3, 4)
    # Essential for Banglish suffixes like -pur, -para, -gram, -nagar, -bazar, -char
    if len(token) >= 2:
        features["prefix-2"] = token_lower[:2]
        features["suffix-2"] = token_lower[-2:]
    if len(token) >= 3:
        features["prefix-3"] = token_lower[:3]
        features["suffix-3"] = token_lower[-3:]
    if len(token) >= 4:
        features["prefix-4"] = token_lower[:4]
        features["suffix-4"] = token_lower[-4:]

    # Stage-1 Anchor Token Indicators
    anchor_type = STAGE1_ANCHORS.get(token)
    if anchor_type:
        features[f"is_anchor_{anchor_type}"] = True
    else:
        features["is_anchor"] = False

    # Gazetteer Phonetic Match Features (Fast O(1) Inverted Index Lookup)
    if matcher is not None:
        # 1. Single token phonetic match (cached)
        node = matcher.lookup_token(token)
        if node:
            features["gaz_matched"] = True
            features["gaz_level"] = node.level.value
        else:
            features["gaz_matched"] = False

        # 2. Forward bigram phonetic match (token + next_token)
        if index + 1 < len(tokens):
            pk_fwd, _ = banglish_phonetic_key(f"{token} {tokens[index + 1]}")
            fwd_nodes = matcher.primary_index.get(pk_fwd)
            if fwd_nodes:
                features["gaz_fwd_bigram_match"] = True
                features["gaz_fwd_bigram_level"] = fwd_nodes[0].level.value

        # 3. Backward bigram phonetic match (prev_token + token)
        if index > 0:
            pk_bwd, _ = banglish_phonetic_key(f"{tokens[index - 1]} {token}")
            bwd_nodes = matcher.primary_index.get(pk_bwd)
            if bwd_nodes:
                features["gaz_bwd_bigram_match"] = True
                features["gaz_bwd_bigram_level"] = bwd_nodes[0].level.value

    # Left Context: Window -1
    if index > 0:
        prev_tok = tokens[index - 1]
        features.update({
            "-1:token.lower": prev_tok.lower(),
            "-1:istitle": prev_tok.istitle(),
            "-1:isdigit": prev_tok.isdigit(),
            "-1:script": get_script_type(prev_tok),
            "-1:anchor": STAGE1_ANCHORS.get(prev_tok, "none"),
            "-1:is_poi_noun": prev_tok.lower() in POI_HEAD_NOUNS,
        })
    else:
        features["BOS"] = True

    # Left Context: Window -2
    if index > 1:
        prev2_tok = tokens[index - 2]
        features.update({
            "-2:token.lower": prev2_tok.lower(),
            "-2:anchor": STAGE1_ANCHORS.get(prev2_tok, "none"),
        })

    # Right Context: Window +1
    if index < len(tokens) - 1:
        next_tok = tokens[index + 1]
        features.update({
            "+1:token.lower": next_tok.lower(),
            "+1:istitle": next_tok.istitle(),
            "+1:isdigit": next_tok.isdigit(),
            "+1:script": get_script_type(next_tok),
            "+1:anchor": STAGE1_ANCHORS.get(next_tok, "none"),
            "+1:is_poi_noun": next_tok.lower() in POI_HEAD_NOUNS,
        })
    else:
        features["EOS"] = True

    # Right Context: Window +2
    if index < len(tokens) - 2:
        next2_tok = tokens[index + 2]
        features.update({
            "+2:token.lower": next2_tok.lower(),
            "+2:anchor": STAGE1_ANCHORS.get(next2_tok, "none"),
            "+2:is_poi_noun": next2_tok.lower() in POI_HEAD_NOUNS,
        })

    return features


def extract_sequence_features(
    tokens: List[str], matcher: Optional[PhoneticGazetteerMatcher] = None
) -> List[Dict[str, Any]]:
    """Extracts features for all tokens in a sequence."""
    return [extract_token_features(tokens, i, matcher) for i in range(len(tokens))]


# -----------------------------------------------------------------------------
# CRF Address Tagger Wrapper
# -----------------------------------------------------------------------------
class CRFAddressTagger:
    """
    Linear-chain CRF Sequence Tagger for Bangladeshi addresses.
    """

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
        matcher: Optional[PhoneticGazetteerMatcher] = None,
    ):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.matcher = matcher
        self.model: Optional[sklearn_crfsuite.CRF] = None

    def fit(self, X_tokens: List[List[str]], y_tags: List[List[str]]) -> CRFAddressTagger:
        """
        Trains the CRF model on token sequences and BIO tag sequences.
        """
        print(f"Extracting features for {len(X_tokens)} sequences...")
        X_features = [
            extract_sequence_features(seq, self.matcher) for seq in X_tokens
        ]

        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
            verbose=False,
        )

        print("Fitting CRF model...")
        self.model.fit(X_features, y_tags)
        print("CRF model training complete.")
        return self

    def predict_tokens(self, tokens: List[str]) -> List[str]:
        """
        Predicts BIO tags for a sequence of tokens.
        """
        if not tokens:
            return []
        if self.model is None:
            raise RuntimeError("CRFAddressTagger is not fitted or loaded.")

        feats = [extract_sequence_features(tokens, self.matcher)]
        preds = self.model.predict(feats)
        return list(preds[0])

    def tag_address(self, raw_or_norm_input: str | Dict[str, Any]) -> Dict[str, Any]:
        """
        End-to-end inference on address text:
        Runs Stage-1 Normalizer, extracts features, predicts BIO tags,
        and aggregates tagged tokens into segmented entity spans.
        """
        start_time = time.perf_counter()

        if isinstance(raw_or_norm_input, str):
            norm_res = normalize_address(raw_or_norm_input)
        else:
            norm_res = raw_or_norm_input

        tokens = norm_res.get("tokens", [])
        if not tokens:
            return {
                "raw_text": norm_res.get("raw_text", ""),
                "normalized_text": norm_res.get("normalized_text", ""),
                "tokens": [],
                "tags": [],
                "entities": {},
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            }

        tags = self.predict_tokens(tokens)

        # Aggregate BIO tokens into segmented entity spans
        entities: Dict[str, List[str]] = {}
        curr_label: Optional[str] = None
        curr_tokens: List[str] = []

        for tok, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                if curr_label and curr_tokens:
                    entities.setdefault(curr_label, []).append(" ".join(curr_tokens))
                curr_label = tag[2:]
                curr_tokens = [tok]
            elif tag.startswith("I-") and curr_label == tag[2:]:
                curr_tokens.append(tok)
            else:
                if curr_label and curr_tokens:
                    entities.setdefault(curr_label, []).append(" ".join(curr_tokens))
                curr_label = None
                curr_tokens = []

        if curr_label and curr_tokens:
            entities.setdefault(curr_label, []).append(" ".join(curr_tokens))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "raw_text": norm_res.get("raw_text", ""),
            "normalized_text": norm_res.get("normalized_text", ""),
            "tokens": tokens,
            "tags": tags,
            "entities": entities,
            "latency_ms": latency_ms,
        }

    def save(self, filepath: str) -> None:
        """Serializes the fitted CRF model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved successfully to: {filepath}")

    @classmethod
    def load(
        cls, filepath: str, matcher: Optional[PhoneticGazetteerMatcher] = None
    ) -> CRFAddressTagger:
        """Loads a serialized CRF model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        tagger = cls(matcher=matcher)
        tagger.model = joblib.load(filepath)
        return tagger
