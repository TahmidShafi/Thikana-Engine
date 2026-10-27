"""
scripts/benchmark_baseline.py

Comprehensive Benchmark Evaluation for Thikana-Engine:
Trains the Stage-2 Linear-Chain CRF model on data/train_synthetic_2500.json,
saves the artifact to models/crf_tagger.joblib, and evaluates 3 production pipelines
on the holdout evaluation set data/eval/gold_100.json:

  - Pipeline A: Regex + Gazetteer Only (No ML Rule-Based Baseline)
  - Pipeline B: Stage 1 Normalizer + Stage 2 CRF Tagger + Stage 3 Disambiguator
  - Pipeline C: Stage 1 Normalizer + Stage 2 ONNX Neural Tagger + Viterbi + Stage 3 Disambiguator

Prints a clean Markdown table comparing:
  - Entity Macro-F1 (computed over active ground-truth labels support > 0)
  - End-to-End Thana/District Accuracy (fairly canonicalizing 2018 official names)
  - Homonym Subset Accuracy
  - Latency percentiles: p50, p95, p99 in milliseconds
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
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
from src.tagger_onnx import ONNXSequenceTagger


def canonicalize_admin_eval_name(name: Optional[str]) -> str:
    """Canonicalizes 2018 official spellings for fair accuracy comparison."""
    if not name:
        return ""
    norm = name.strip().lower()
    mapping = {
        "chattagram": "chattogram",
        "chittagong": "chattogram",
        "comilla": "cumilla",
        "barisal": "barishal",
        "jessore": "jashore",
        "bogra": "bogura",
    }
    return mapping.get(norm, norm)


# -----------------------------------------------------------------------------
# Pipeline A: Naive Regex + Gazetteer Baseline (No Machine Learning)
# -----------------------------------------------------------------------------
class RegexGazetteerBaseline:
    """
    Simulates standard legacy e-commerce parser:
    Uses regex patterns for House/Road/Block and exact gazetteer lookups for locations.
    """

    def __init__(self, gazetteer: BangladeshGazetteer):
        self.gazetteer = gazetteer
        self.district_map: Dict[str, AdminNode] = {}
        for d in gazetteer.districts:
            self.district_map[d.name.lower()] = d
            if d.bn_name:
                self.district_map[d.bn_name.lower()] = d

        self.upazila_map: Dict[str, AdminNode] = {}
        for u in gazetteer.upazilas:
            self.upazila_map[u.name.lower()] = u
            if u.bn_name:
                self.upazila_map[u.bn_name.lower()] = u

    def tag_address(self, norm_res: Dict[str, Any]) -> Tuple[List[str], Optional[str], Optional[str]]:
        tokens: List[str] = norm_res.get("tokens", [])
        tags: List[str] = ["O"] * len(tokens)
        predicted_thana: Optional[str] = None
        predicted_district: Optional[str] = None

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            tok_l = tok.lower()

            if tok == "H#":
                tags[i] = "B-HOUSE"
                if i + 1 < len(tokens):
                    tags[i + 1] = "I-HOUSE"
                    i += 2
                    continue
                i += 1
                continue

            if tok == "RD#":
                tags[i] = "B-ROAD"
                if i + 1 < len(tokens):
                    tags[i + 1] = "I-ROAD"
                    i += 2
                    continue
                i += 1
                continue

            if tok in ("BLK#", "SEC#"):
                tags[i] = "B-BLOCK_SECTOR"
                if i + 1 < len(tokens):
                    tags[i + 1] = "I-BLOCK_SECTOR"
                    i += 2
                    continue
                i += 1
                continue

            if tok == "FL#":
                tags[i] = "B-FLOOR_FLAT"
                if i + 1 < len(tokens):
                    tags[i + 1] = "I-FLOOR_FLAT"
                    i += 2
                    continue
                i += 1
                continue

            # District lookup
            if tok_l in self.district_map:
                tags[i] = "B-DISTRICT"
                if not predicted_district:
                    predicted_district = self.district_map[tok_l].name
                i += 1
                continue

            # Upazila lookup
            if tok_l in self.upazila_map:
                tags[i] = "B-THANA_UPAZILA"
                upz_node = self.upazila_map[tok_l]
                if not predicted_thana:
                    predicted_thana = upz_node.name
                    if not predicted_district and upz_node.district is not None:
                        predicted_district = upz_node.district.name
                i += 1
                continue

            # Check 2-word combinations for district/upazila
            if i + 1 < len(tokens):
                bigram = f"{tok} {tokens[i+1]}".lower()
                if bigram in self.upazila_map:
                    tags[i] = "B-THANA_UPAZILA"
                    tags[i + 1] = "I-THANA_UPAZILA"
                    upz_node = self.upazila_map[bigram]
                    if not predicted_thana:
                        predicted_thana = upz_node.name
                        if not predicted_district and upz_node.district is not None:
                            predicted_district = upz_node.district.name
                    i += 2
                    continue

            tags[i] = "O"
            i += 1

        return tags, predicted_thana, predicted_district


# -----------------------------------------------------------------------------
# Pipeline B & C Runner
# -----------------------------------------------------------------------------
class GenericPipelineRunner:
    def __init__(self, tagger: Any, gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher):
        self.tagger = tagger
        self.gazetteer = gazetteer
        self.matcher = matcher

    def process(self, raw_text: str) -> Dict[str, Any]:
        norm_res = normalize_address(raw_text)
        tokens = norm_res.get("tokens", [])

        tag_res = self.tagger.tag_address(norm_res)
        tags = tag_res["tags"]
        entities = tag_res["entities"]

        candidate_spans: List[str] = []
        for loc_type in ("SUBAREA", "THANA_UPAZILA", "DISTRICT"):
            if loc_type in entities:
                candidate_spans.extend(entities[loc_type])

        if not candidate_spans:
            candidate_spans = tokens

        disambig_res = resolve_location_hierarchy(
            candidate_spans=candidate_spans,
            full_context_tokens=tokens,
            gazetteer=self.gazetteer,
            matcher=self.matcher,
        )

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
    onnx_path = os.path.join(model_dir, "onnx", "tagger.onnx")

    print("=" * 80)
    print("           THIKANA-ENGINE PRODUCTION BENCHMARK EVALUATION")
    print("=" * 80)

    # 1. Initialize gazetteer & phonetic matcher
    print("\n[1/5] Loading BangladeshGazetteer and Phonetic Index...")
    g_start = time.perf_counter()
    gz = BangladeshGazetteer()
    matcher = PhoneticGazetteerMatcher(gz)
    print(f"Loaded in {(time.perf_counter() - g_start):.2f}s")

    # 2. Train or Load CRF Model
    print("\n[2/5] Training Stage-2 Linear-Chain CRF on Synthetic Dataset...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    X_train = [item["tokens"] for item in train_data]
    y_train = [item["tags"] for item in train_data]

    crf = CRFAddressTagger(c1=0.1, c2=0.1, max_iterations=100, matcher=matcher)
    t_start = time.perf_counter()
    crf.fit(X_train, y_train)
    train_time = time.perf_counter() - t_start
    print(f"Trained in {train_time:.2f}s")

    print(f"Saving CRF model artifact to: {model_path}...")
    crf.save(model_path)

    # 3. Load ONNX Sequence Tagger
    print("\n[3/5] Initializing Stage-2 ONNX Sequence Tagger with Viterbi Decoder...")
    onnx_tagger = ONNXSequenceTagger(model_path=onnx_path, matcher=matcher, intra_op_num_threads=2)

    # 4. Load Gold 100 Evaluation Dataset
    print("\n[4/5] Loading Gold 100 Evaluation Dataset...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    print(f"Loaded {len(gold_data)} gold records.")

    # 5. Evaluate Pipelines
    print("\n[5/5] Benchmarking Pipelines on Gold Evaluation Set...")
    baseline_a = RegexGazetteerBaseline(gz)
    pipeline_b = GenericPipelineRunner(crf, gz, matcher)
    pipeline_c = GenericPipelineRunner(onnx_tagger, gz, matcher)

    # Ground truth tags aggregation
    y_true_all: List[str] = []
    for item in gold_data:
        y_true_all.extend(item["tags"])

    # Active labels in ground truth (excluding 'O' and 0-support classes)
    active_eval_labels = sorted(list(set(y_true_all) - {"O"}))

    def evaluate_pipeline(name: str, runner_func) -> Dict[str, Any]:
        latencies: List[float] = []
        y_pred_all: List[str] = []
        correct_e2e = 0
        correct_homonym = 0
        homonym_total = 0

        for item in gold_data:
            raw_text = item["raw_text"]
            true_tags = item["tags"]
            exp_thana = item["expected_thana"]
            exp_dist = item["expected_district"]
            is_homonym = item["is_homonym_test"]
            if is_homonym:
                homonym_total += 1

            t0 = time.perf_counter()
            pred_tags, p_thana, p_dist = runner_func(raw_text)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            min_len = min(len(true_tags), len(pred_tags))
            y_pred_all.extend(pred_tags[:min_len])

            match_thana = (
                canonicalize_admin_eval_name(p_thana) == canonicalize_admin_eval_name(exp_thana)
            )
            match_dist = (
                canonicalize_admin_eval_name(p_dist) == canonicalize_admin_eval_name(exp_dist)
            )
            if match_thana and match_dist:
                correct_e2e += 1
                if is_homonym:
                    correct_homonym += 1

        macro_f1 = f1_score(
            y_true_all[: len(y_pred_all)],
            y_pred_all,
            labels=active_eval_labels,
            average="macro",
            zero_division=0,
        )
        e2e_acc = (correct_e2e / len(gold_data)) * 100.0
        homonym_acc = (correct_homonym / homonym_total) * 100.0 if homonym_total > 0 else 0.0

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        return {
            "name": name,
            "macro_f1": macro_f1 * 100.0,
            "e2e_acc": e2e_acc,
            "correct_e2e": correct_e2e,
            "homonym_acc": homonym_acc,
            "correct_homonym": correct_homonym,
            "homonym_total": homonym_total,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "y_pred": y_pred_all,
        }

    # Run Benchmark on all 3 Pipelines
    res_a = evaluate_pipeline(
        "Pipeline A (Regex+Gazetteer)",
        lambda txt: baseline_a.tag_address(normalize_address(txt)),
    )
    res_b = evaluate_pipeline(
        "Pipeline B (CRF+Disambiguator)",
        lambda txt: (
            pipeline_b.process(txt)["tags"],
            pipeline_b.process(txt)["predicted_thana"],
            pipeline_b.process(txt)["predicted_district"],
        ),
    )
    res_c = evaluate_pipeline(
        "Pipeline C (ONNX+Viterbi+Disambiguator)",
        lambda txt: (
            pipeline_c.process(txt)["tags"],
            pipeline_c.process(txt)["predicted_thana"],
            pipeline_c.process(txt)["predicted_district"],
        ),
    )

    # --- RESULTS TABLE ---
    print("\n" + "=" * 80)
    print("                     EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Test Addresses: {len(gold_data)} | Homonym Subset: {res_b['homonym_total']}\n")

    markdown_table = f"""| Metric | Pipeline A: Regex + Gazetteer (No ML) | Pipeline B: Stage 1 + CRF + Disambiguator | Pipeline C: Stage 1 + ONNX + Viterbi |
| :--- | :---: | :---: | :---: |
| **Entity Macro-F1** | `{res_a['macro_f1']:.2f}%` | **`{res_b['macro_f1']:.2f}%`** | `{res_c['macro_f1']:.2f}%` |
| **End-to-End Thana/District Accuracy** | `{res_a['e2e_acc']:.1f}%` ({res_a['correct_e2e']}/{len(gold_data)}) | **`{res_b['e2e_acc']:.1f}%`** ({res_b['correct_e2e']}/{len(gold_data)}) | `{res_c['e2e_acc']:.1f}%` ({res_c['correct_e2e']}/{len(gold_data)}) |
| **Homonym Subset Accuracy** | `{res_a['homonym_acc']:.1f}%` ({res_a['correct_homonym']}/{res_a['homonym_total']}) | **`{res_b['homonym_acc']:.1f}%`** ({res_b['correct_homonym']}/{res_b['homonym_total']}) | `{res_c['homonym_acc']:.1f}%` ({res_c['correct_homonym']}/{res_c['homonym_total']}) |
| **CPU Latency (p50)** | `{res_a['p50']:.3f} ms` | `{res_b['p50']:.3f} ms` | `{res_c['p50']:.3f} ms` |
| **CPU Latency (p95)** | `{res_a['p95']:.3f} ms` | `{res_b['p95']:.3f} ms` | `{res_c['p95']:.3f} ms` |
| **CPU Latency (p99)** | `{res_a['p99']:.3f} ms` | `{res_b['p99']:.3f} ms` | `{res_c['p99']:.3f} ms` |"""

    print(markdown_table)
    print("\n" + "=" * 80)

    # Detailed per-entity classification report for Pipeline B
    print("\nDetailed Entity Tag Classification Report (Pipeline B - CRF):")
    print(
        classification_report(
            y_true_all[: len(res_b["y_pred"])],
            res_b["y_pred"],
            labels=active_eval_labels,
            zero_division=0,
        )
    )

    print("\nDetailed Entity Tag Classification Report (Pipeline C - ONNX):")
    print(
        classification_report(
            y_true_all[: len(res_c["y_pred"])],
            res_c["y_pred"],
            labels=active_eval_labels,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    run_benchmark()
