"""
scripts/benchmark_baseline.py

Step 4 Benchmark:
Trains the Stage-2 Linear-Chain CRF model on data/train_synthetic_2500.json,
saves the model artifact to models/crf_tagger.joblib, and evaluates two pipelines
on the holdout evaluation set data/eval/gold_100.json:

  - Pipeline A: Regex + Gazetteer Only (No ML Rule-Based Baseline)
  - Pipeline B: Stage 1 Normalizer + Stage 2 CRF Tagger + Stage 3 Disambiguator

Prints a clean Markdown table comparing:
  - Entity Macro-F1
  - End-to-End Thana/District Accuracy
  - Homonym Subset Accuracy
  - Latency: p50, p95, p99 in milliseconds
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gazetteer import BangladeshGazetteer
from src.models import AdminLevel, AdminNode
from src.normalizer import normalize_address
from src.phonetic_trie import PhoneticGazetteerMatcher, banglish_phonetic_key
from src.disambiguator import resolve_location_hierarchy
from src.tagger_crf import (
    CRFAddressTagger,
    ENTITY_TYPES,
    BIO_LABELS,
    extract_sequence_features,
)


# -----------------------------------------------------------------------------
# Pipeline A: Regex + Direct Gazetteer Only (No ML Baseline)
# -----------------------------------------------------------------------------
class RegexGazetteerBaseline:
    """
    Rule-based baseline that tags tokens based on Stage-1 anchors and
    naive exact gazetteer lookups without contextual sequence modeling.
    """

    def __init__(self, gazetteer: BangladeshGazetteer):
        self.gazetteer = gazetteer
        # Index exact lowercase names
        self.district_names: Set[str] = {
            d.name.lower() for d in gazetteer.districts
        }
        self.district_names.update({
            d.bn_name for d in gazetteer.districts if d.bn_name
        })

        self.upazila_map: Dict[str, AdminNode] = {}
        for node in gazetteer.all_nodes:
            if node.level in (AdminLevel.UPAZILA, AdminLevel.THANA):
                self.upazila_map[node.name.lower()] = node
                if node.bn_name:
                    self.upazila_map[node.bn_name] = node

    def tag_address(self, norm_res: Dict[str, Any]) -> Tuple[List[str], str | None, str | None]:
        tokens = norm_res.get("tokens", [])
        tags = ["O"] * len(tokens)
        predicted_thana: str | None = None
        predicted_district: str | None = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok_l = tok.lower()

            # Anchor heuristics
            if tok == "H#" and i + 1 < len(tokens):
                tags[i] = "B-HOUSE"
                tags[i + 1] = "I-HOUSE"
                i += 2
                continue
            elif tok == "RD#" and i + 1 < len(tokens):
                tags[i] = "B-ROAD"
                tags[i + 1] = "I-ROAD"
                i += 2
                continue
            elif tok in ("BLK#", "SEC#") and i + 1 < len(tokens):
                tags[i] = "B-BLOCK_SECTOR"
                tags[i + 1] = "I-BLOCK_SECTOR"
                i += 2
                continue
            elif tok == "FL#" and i + 1 < len(tokens):
                tags[i] = "B-FLOOR_FLAT"
                tags[i + 1] = "I-FLOOR_FLAT"
                i += 2
                continue
            elif tok in ("REL_NEAR", "REL_BEHIND", "REL_FRONT"):
                tags[i] = "B-LANDMARK_REL"
                i += 1
                continue

            # District lookup
            if tok_l in self.district_names:
                tags[i] = "B-DISTRICT"
                if not predicted_district:
                    predicted_district = tok.title()
                i += 1
                continue

            # Upazila lookup
            if tok_l in self.upazila_map:
                tags[i] = "B-THANA_UPAZILA"
                if not predicted_thana:
                    predicted_thana = self.upazila_map[tok_l].name
                    if not predicted_district and self.upazila_map[tok_l].district:
                        predicted_district = self.upazila_map[tok_l].district.name
                i += 1
                continue

            # Check 2-word combinations for district/upazila
            if i + 1 < len(tokens):
                bigram = f"{tok} {tokens[i+1]}".lower()
                if bigram in self.upazila_map:
                    tags[i] = "B-THANA_UPAZILA"
                    tags[i + 1] = "I-THANA_UPAZILA"
                    if not predicted_thana:
                        predicted_thana = self.upazila_map[bigram].name
                        if not predicted_district and self.upazila_map[bigram].district:
                            predicted_district = self.upazila_map[bigram].district.name
                    i += 2
                    continue

            tags[i] = "O"
            i += 1

        return tags, predicted_thana, predicted_district


# -----------------------------------------------------------------------------
# Pipeline B: Stage 1 + Stage 2 CRF + Stage 3 Disambiguator
# -----------------------------------------------------------------------------
class FullMLPipeline:
    """
    End-to-End Pipeline:
      Raw Address -> Stage 1 Normalizer -> Stage 2 CRF Tagger -> Stage 3 Disambiguator
    """

    def __init__(
        self,
        crf_tagger: CRFAddressTagger,
        gazetteer: BangladeshGazetteer,
        matcher: PhoneticGazetteerMatcher,
    ):
        self.crf_tagger = crf_tagger
        self.gazetteer = gazetteer
        self.matcher = matcher

    def process(self, raw_text: str) -> Dict[str, Any]:
        # Stage 1: Normalizer
        norm_res = normalize_address(raw_text)
        tokens = norm_res.get("tokens", [])

        # Stage 2: CRF Tagger
        tag_res = self.crf_tagger.tag_address(norm_res)
        tags = tag_res["tags"]
        entities = tag_res["entities"]

        # Collect candidate location spans from CRF entities
        candidate_spans: List[str] = []
        for loc_type in ("SUBAREA", "THANA_UPAZILA", "DISTRICT"):
            if loc_type in entities:
                candidate_spans.extend(entities[loc_type])

        # Fallback if CRF found no location tags: inspect tokens with gazetteer matcher
        if not candidate_spans:
            candidate_spans = tokens

        # Stage 3: Disambiguator
        disambig_res = resolve_location_hierarchy(
            candidate_spans=candidate_spans,
            full_context_tokens=tokens,
            gazetteer=self.gazetteer,
            matcher=self.matcher,
        )

        resolved_node = disambig_res.get("resolved_node")
        hierarchy = disambig_res.get("hierarchy", {})

        pred_thana = hierarchy.get("thana") or hierarchy.get("upazila")
        pred_district = hierarchy.get("district")

        return {
            "tokens": tokens,
            "tags": tags,
            "entities": entities,
            "predicted_thana": pred_thana,
            "predicted_district": pred_district,
            "hierarchy": hierarchy,
        }


# -----------------------------------------------------------------------------
# Benchmark Runner
# -----------------------------------------------------------------------------
def run_benchmark():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(root_dir, "data", "train_synthetic_2500.json")
    gold_path = os.path.join(root_dir, "data", "eval", "gold_100.json")
    model_dir = os.path.join(root_dir, "models")
    model_path = os.path.join(model_dir, "crf_tagger.joblib")

    print("=" * 70)
    print("           THIKANA-ENGINE STEP 4 BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Initialize gazetteer & phonetic matcher
    print("\n[1/4] Loading BangladeshGazetteer and Phonetic Index...")
    g_start = time.perf_counter()
    gz = BangladeshGazetteer()
    matcher = PhoneticGazetteerMatcher(gz)
    print(f"Loaded in {(time.perf_counter() - g_start):.2f}s")

    # 2. Train or Load CRF Model
    print("\n[2/4] Training Stage-2 Linear-Chain CRF on Synthetic Dataset...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    X_train = [item["tokens"] for item in train_data]
    y_train = [item["tags"] for item in train_data]

    crf = CRFAddressTagger(
        c1=0.1, c2=0.1, max_iterations=100, matcher=matcher
    )
    t_start = time.perf_counter()
    crf.fit(X_train, y_train)
    train_time = time.perf_counter() - t_start
    print(f"Trained in {train_time:.2f}s")

    print(f"Saving model artifact to: {model_path}...")
    crf.save(model_path)

    # 3. Load Gold 100 Evaluation Dataset
    print("\n[3/4] Loading Gold 100 Evaluation Dataset...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold records.")

    # 4. Evaluate Pipeline A: Regex + Gazetteer
    print("\n[4/4] Benchmarking Pipelines on Gold Evaluation Set...")
    baseline_a = RegexGazetteerBaseline(gz)
    pipeline_b = FullMLPipeline(crf, gz, matcher)

    # All evaluation labels (excluding 'O')
    eval_labels = [lbl for lbl in BIO_LABELS if lbl != "O"]

    # --- PIPELINE A BENCHMARK ---
    latencies_a: List[float] = []
    y_true_all: List[str] = []
    y_pred_a_all: List[str] = []
    correct_e2e_a = 0
    correct_homonym_a = 0
    homonym_total = 0

    for item in gold_data:
        raw_text = item["raw_text"]
        true_tags = item["tags"]
        exp_thana = item["expected_thana"].strip().lower()
        exp_dist = item["expected_district"].strip().lower()
        is_homonym = item["is_homonym_test"]
        if is_homonym:
            homonym_total += 1

        t0 = time.perf_counter()
        norm = normalize_address(raw_text)
        pred_tags_a, p_thana, p_dist = baseline_a.tag_address(norm)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_a.append(lat_ms)

        # Pad/align tag lists if lengths differ
        min_len = min(len(true_tags), len(pred_tags_a))
        y_true_all.extend(true_tags[:min_len])
        y_pred_a_all.extend(pred_tags_a[:min_len])

        match_thana = (p_thana and p_thana.strip().lower() == exp_thana)
        match_dist = (p_dist and p_dist.strip().lower() == exp_dist)
        is_e2e_correct = match_thana and match_dist
        if is_e2e_correct:
            correct_e2e_a += 1
            if is_homonym:
                correct_homonym_a += 1

    macro_f1_a = f1_score(
        y_true_all, y_pred_a_all, labels=eval_labels, average="macro", zero_division=0
    )
    e2e_acc_a = (correct_e2e_a / len(gold_data)) * 100.0
    homonym_acc_a = (correct_homonym_a / homonym_total) * 100.0 if homonym_total > 0 else 0.0

    p50_a = float(np.percentile(latencies_a, 50))
    p95_a = float(np.percentile(latencies_a, 95))
    p99_a = float(np.percentile(latencies_a, 99))

    # --- PIPELINE B BENCHMARK ---
    latencies_b: List[float] = []
    y_pred_b_all: List[str] = []
    correct_e2e_b = 0
    correct_homonym_b = 0

    for item in gold_data:
        raw_text = item["raw_text"]
        true_tags = item["tags"]
        exp_thana = item["expected_thana"].strip().lower()
        exp_dist = item["expected_district"].strip().lower()
        is_homonym = item["is_homonym_test"]

        t0 = time.perf_counter()
        res_b = pipeline_b.process(raw_text)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_b.append(lat_ms)

        pred_tags_b = res_b["tags"]
        min_len = min(len(true_tags), len(pred_tags_b))
        y_pred_b_all.extend(pred_tags_b[:min_len])

        p_thana = res_b["predicted_thana"]
        p_dist = res_b["predicted_district"]

        # Disambiguator accuracy comparison (case-insensitive)
        match_thana = (p_thana and p_thana.strip().lower() == exp_thana)
        match_dist = (p_dist and p_dist.strip().lower() == exp_dist)
        is_e2e_correct = match_thana and match_dist
        if is_e2e_correct:
            correct_e2e_b += 1
            if is_homonym:
                correct_homonym_b += 1

    macro_f1_b = f1_score(
        y_true_all, y_pred_b_all, labels=eval_labels, average="macro", zero_division=0
    )
    e2e_acc_b = (correct_e2e_b / len(gold_data)) * 100.0
    homonym_acc_b = (correct_homonym_b / homonym_total) * 100.0 if homonym_total > 0 else 0.0

    p50_b = float(np.percentile(latencies_b, 50))
    p95_b = float(np.percentile(latencies_b, 95))
    p99_b = float(np.percentile(latencies_b, 99))

    # --- RESULTS TABLE ---
    print("\n" + "=" * 70)
    print("                     EVALUATION REPORT")
    print("=" * 70)
    print(f"Total Test Addresses: {len(gold_data)} | Homonym Subset: {homonym_total}\n")

    markdown_table = f"""| Metric | Pipeline A: Regex + Gazetteer (No ML) | Pipeline B: Stage 1 + CRF + Disambiguator | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Entity Macro-F1** | `{macro_f1_a * 100:.2f}%` | `{macro_f1_b * 100:.2f}%` | `{macro_f1_b * 100 - macro_f1_a * 100:+.2f}%` |
| **End-to-End Thana/District Accuracy** | `{e2e_acc_a:.1f}%` ({correct_e2e_a}/{len(gold_data)}) | `{e2e_acc_b:.1f}%` ({correct_e2e_b}/{len(gold_data)}) | `{e2e_acc_b - e2e_acc_a:+.1f}%` |
| **Homonym Subset Accuracy** | `{homonym_acc_a:.1f}%` ({correct_homonym_a}/{homonym_total}) | `{homonym_acc_b:.1f}%` ({correct_homonym_b}/{homonym_total}) | `{homonym_acc_b - homonym_acc_a:+.1f}%` |
| **CPU Latency (p50)** | `{p50_a:.3f} ms` | `{p50_b:.3f} ms` | `{(p50_b - p50_a):+.3f} ms` |
| **CPU Latency (p95)** | `{p95_a:.3f} ms` | `{p95_b:.3f} ms` | `{(p95_b - p95_a):+.3f} ms` |
| **CPU Latency (p99)** | `{p99_a:.3f} ms` | `{p99_b:.3f} ms` | `{(p99_b - p99_a):+.3f} ms` |"""

    print(markdown_table)
    print("\n" + "=" * 70)

    # Detailed per-entity classification report for Pipeline B
    print("\nDetailed Entity Tag Classification Report (Pipeline B):")
    print(classification_report(y_true_all, y_pred_b_all, labels=eval_labels, zero_division=0))


if __name__ == "__main__":
    run_benchmark()
