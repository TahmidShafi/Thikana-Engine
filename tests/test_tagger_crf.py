"""
tests/test_tagger_crf.py

Unit and integration tests for Stage-2 CRF Tagger, Feature Extraction,
Gold Evaluation Dataset, and Non-Leaky Geographic Split.
"""

from __future__ import annotations
import json
import os
import tempfile
import pytest

from src.gazetteer import BangladeshGazetteer
from src.normalizer import normalize_address
from src.phonetic_trie import PhoneticGazetteerMatcher
from src.tagger_crf import (
    CRFAddressTagger,
    ENTITY_TYPES,
    BIO_LABELS,
    get_script_type,
    extract_token_features,
    extract_sequence_features,
)


@pytest.fixture(scope="module")
def gazetteer() -> BangladeshGazetteer:
    return BangladeshGazetteer()


@pytest.fixture(scope="module")
def matcher(gazetteer: BangladeshGazetteer) -> PhoneticGazetteerMatcher:
    return PhoneticGazetteerMatcher(gazetteer)


def test_entity_types_and_bio_labels():
    """Verify 9 BIO entity types and 19 BIO tags."""
    assert len(ENTITY_TYPES) == 9
    assert "HOUSE" in ENTITY_TYPES
    assert "FLOOR_FLAT" in ENTITY_TYPES
    assert "ROAD" in ENTITY_TYPES
    assert "BLOCK_SECTOR" in ENTITY_TYPES
    assert "LANDMARK" in ENTITY_TYPES
    assert "LANDMARK_REL" in ENTITY_TYPES
    assert "SUBAREA" in ENTITY_TYPES
    assert "THANA_UPAZILA" in ENTITY_TYPES
    assert "DISTRICT" in ENTITY_TYPES

    # 1 'O' + 9 'B-' + 9 'I-' = 19
    assert len(BIO_LABELS) == 19
    assert "O" in BIO_LABELS
    for et in ENTITY_TYPES:
        assert f"B-{et}" in BIO_LABELS
        assert f"I-{et}" in BIO_LABELS


def test_script_classification():
    """Verify script classification into digit, bangla, latin, and mixed."""
    assert get_script_type("123") == "digit"
    assert get_script_type("৪৫") == "digit"  # Bengali digits match Unicode \d
    assert get_script_type("মিরপুর") == "bangla"
    assert get_script_type("Dhanmondi") == "latin"
    assert get_script_type("H#") == "mixed_or_punct"
    assert get_script_type("12/B") == "mixed_or_punct"


def test_token_feature_extraction(matcher: PhoneticGazetteerMatcher):
    """Verify token feature extraction and context window."""
    tokens = ["H#", "45", "RD#", "Tajmahal", "Mohammadpur", "Dhaka"]
    feats = extract_sequence_features(tokens, matcher)

    assert len(feats) == len(tokens)

    # First token has BOS
    assert feats[0]["BOS"] is True
    assert feats[0].get("is_anchor_house") is True

    # Last token has EOS
    assert feats[-1]["EOS"] is True
    assert feats[-1]["token.lower"] == "dhaka"
    assert feats[-1]["gaz_matched"] is True
    assert feats[-1]["gaz_level"] in ("division", "district")

    # Suffixes
    assert feats[4]["token.lower"] == "mohammadpur"
    assert feats[4]["suffix-3"] == "pur"
    assert feats[4]["gaz_matched"] is True


def test_crf_fit_predict_save_load(matcher: PhoneticGazetteerMatcher):
    """Verify CRF training, serialization, loading, and inference."""
    X_train = [
        ["H#", "12", "RD#", "4", "Dhanmondi", "Dhaka"],
        ["Plot", "5", "Road", "2", "Gulshan", "Dhaka"],
        ["H#", "88", "Sector", "3", "Uttara", "Dhaka"],
    ]
    y_train = [
        ["B-HOUSE", "I-HOUSE", "B-ROAD", "I-ROAD", "B-THANA_UPAZILA", "B-DISTRICT"],
        ["B-HOUSE", "I-HOUSE", "B-ROAD", "I-ROAD", "B-THANA_UPAZILA", "B-DISTRICT"],
        ["B-HOUSE", "I-HOUSE", "B-BLOCK_SECTOR", "I-BLOCK_SECTOR", "B-THANA_UPAZILA", "B-DISTRICT"],
    ]

    tagger = CRFAddressTagger(c1=0.1, c2=0.1, max_iterations=20, matcher=matcher)
    tagger.fit(X_train, y_train)

    test_tokens = ["H#", "15", "RD#", "8", "Dhanmondi", "Dhaka"]
    preds1 = tagger.predict_tokens(test_tokens)
    assert len(preds1) == len(test_tokens)
    assert preds1[0] == "B-HOUSE"
    assert preds1[-1] == "B-DISTRICT"

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tagger.save(tmp_path)
        loaded_tagger = CRFAddressTagger.load(tmp_path, matcher=matcher)
        preds2 = loaded_tagger.predict_tokens(test_tokens)
        assert preds1 == preds2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_tag_address_entity_span_aggregation(matcher: PhoneticGazetteerMatcher):
    """Verify tag_address aggregates BIO tokens into structured entity spans."""
    X_train = [
        ["H#", "12", "RD#", "Tajmahal", "Mohammadpur", "Dhaka"],
        ["H#", "45", "RD#", "Mirpur", "Dhaka"],
    ]
    y_train = [
        ["B-HOUSE", "I-HOUSE", "B-ROAD", "I-ROAD", "B-THANA_UPAZILA", "B-DISTRICT"],
        ["B-HOUSE", "I-HOUSE", "B-ROAD", "B-THANA_UPAZILA", "B-DISTRICT"],
    ]

    tagger = CRFAddressTagger(c1=0.1, c2=0.1, max_iterations=20, matcher=matcher)
    tagger.fit(X_train, y_train)

    res = tagger.tag_address("Holding 12, Tajmahal Road, Mohammadpur, Dhaka")
    assert "tokens" in res
    assert "tags" in res
    assert "entities" in res
    assert "latency_ms" in res
    assert res["latency_ms"] >= 0.0


def test_gold_dataset_integrity():
    """Verify gold_100.json has exactly 100 valid, consistent samples."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gold_path = os.path.join(root_dir, "data", "eval", "gold_100.json")

    assert os.path.exists(gold_path), f"gold_100.json not found at {gold_path}"

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    assert len(gold_data) == 100

    homonym_samples = 0
    for record in gold_data:
        assert "id" in record
        assert "raw_text" in record
        assert "tokens" in record
        assert "tags" in record
        assert "expected_thana" in record
        assert "expected_district" in record
        assert "is_homonym_test" in record
        assert "category" in record

        # 1:1 token to tag alignment
        assert len(record["tokens"]) == len(record["tags"])
        for tag in record["tags"]:
            assert tag in BIO_LABELS, f"Invalid tag '{tag}' in record {record['id']}"

        if record["is_homonym_test"]:
            homonym_samples += 1

    assert homonym_samples >= 25, "Expected at least 25 homonym/distractor test cases"


def test_synthetic_training_non_leaky_geographic_split():
    """Verify that train_synthetic_2500.json strictly holds out gold Upazilas."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gold_path = os.path.join(root_dir, "data", "eval", "gold_100.json")
    train_path = os.path.join(root_dir, "data", "train_synthetic_2500.json")

    assert os.path.exists(train_path), f"train_synthetic_2500.json not found at {train_path}"

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    assert len(train_data) == 2500

    holdout_upazilas = {
        item["expected_thana"].strip().lower() for item in gold_data if item.get("expected_thana")
    }

    # Verify zero overlap between train upazilas and gold holdout upazilas
    for item in train_data:
        train_upz = item["upazila"].strip().lower()
        assert train_upz not in holdout_upazilas, (
            f"Geographic leakage detected! Upazila '{train_upz}' is in holdout list!"
        )
