"""
src/pipeline.py

Unified End-to-End Address Parsing Pipeline for Thikana-Engine.
Chains:
  Stage 1: Deterministic Script, Unicode, and Numeral Normalizer (src/normalizer.py)
  Stage 2: Modular Sequence Tagger: 'crf' (default) or 'onnx' (src/tagger_crf.py / src/tagger_onnx.py)
  Stage 3: Context-Aware Homonym Disambiguator (src/disambiguator.py)
  Stage 4: Multi-Courier Hub Mapper & Truthful Geocoder (src/courier_mapper.py)

Returns a strict Pydantic AddressParseResponse.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from .models import (
    AddressParseResponse,
    AdminLevel,
    AdminNode,
    CourierRouting,
    DisambiguationCandidate,
    GeocodeInfo,
    ParsedComponents,
    StageLatencies,
)
from .gazetteer import BangladeshGazetteer
from .normalizer import normalize_address
from .phonetic_trie import PhoneticGazetteerMatcher
from .disambiguator import resolve_location_hierarchy
from .tagger_crf import CRFAddressTagger
from .tagger_onnx import ONNXSequenceTagger
from .courier_mapper import MultiCourierHubMapper, resolve_truthful_geocode

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CRF_PATH = DEFAULT_MODELS_DIR / "crf_tagger.joblib"
DEFAULT_ONNX_PATH = DEFAULT_MODELS_DIR / "onnx" / "tagger.onnx"


class ThikanaPipeline:
    """
    Unified end-to-end Bangladeshi address parsing engine.
    Supports modular tagger backends:
      - backend='crf'  (default, self-contained, linear-chain CRF)
      - backend='onnx' (high-performance INT8 / ONNX runtime with Viterbi decoding)
    """

    def __init__(
        self,
        backend: str = "crf",
        crf_model_path: Optional[str | Path] = None,
        onnx_model_path: Optional[str | Path] = None,
        gazetteer: Optional[BangladeshGazetteer] = None,
        matcher: Optional[PhoneticGazetteerMatcher] = None,
        courier_seed_path: Optional[str | Path] = None,
    ):
        self.backend = backend.lower()
        if self.backend not in ("crf", "onnx"):
            raise ValueError(f"Invalid backend '{backend}'. Supported backends: 'crf', 'onnx'.")

        self.crf_model_path = Path(crf_model_path) if crf_model_path else DEFAULT_CRF_PATH
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path else DEFAULT_ONNX_PATH

        # Initialize shared gazetteer and phonetic inverted index
        self.gazetteer = gazetteer if gazetteer is not None else BangladeshGazetteer()
        self.matcher = matcher if matcher is not None else PhoneticGazetteerMatcher(self.gazetteer)

        # Initialize Stage-4 Multi-Courier Hub Mapper
        self.courier_mapper = MultiCourierHubMapper(seed_path=courier_seed_path)

        # Lazy-loaded tagger instances
        self._crf_tagger: Optional[CRFAddressTagger] = None
        self._onnx_tagger: Optional[ONNXSequenceTagger] = None

        # Preload active backend
        self._init_backend(self.backend)

    def _init_backend(self, backend: str) -> None:
        """Initializes the requested sequence tagger backend."""
        if backend == "crf":
            if self._crf_tagger is None:
                if self.crf_model_path.exists():
                    self._crf_tagger = CRFAddressTagger.load(
                        str(self.crf_model_path), matcher=self.matcher
                    )
                else:
                    logger.warning(
                        f"CRF model artifact not found at {self.crf_model_path}; "
                        "initializing un-fitted tagger."
                    )
                    self._crf_tagger = CRFAddressTagger(matcher=self.matcher)
        elif backend == "onnx":
            if self._onnx_tagger is None:
                self._onnx_tagger = ONNXSequenceTagger(
                    model_path=self.onnx_model_path,
                    matcher=self.matcher,
                    intra_op_num_threads=2,
                )

    def set_backend(self, backend: str) -> None:
        """Switches active tagger backend dynamically ('crf' <-> 'onnx')."""
        backend_clean = backend.lower()
        if backend_clean not in ("crf", "onnx"):
            raise ValueError(f"Invalid backend '{backend}'. Must be 'crf' or 'onnx'.")
        self._init_backend(backend_clean)
        self.backend = backend_clean

    def parse(self, raw_address: str) -> AddressParseResponse:
        """
        Executes the full 4-stage pipeline:
          Stage 1: Script, Unicode, Numeral Normalization
          Stage 2: Sequence Tagging (CRF or ONNX) -> Entity Extraction
          Stage 3: Context-Aware Homonym Disambiguation -> Administrative Hierarchy
          Stage 4: Multi-Courier Hub Mapping & Truthful Geocoding

        Returns:
          Strict Pydantic AddressParseResponse
        """
        start_time = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 1: Deterministic Script, Unicode & Numeral Normalization
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        norm_res = normalize_address(raw_address)
        tokens = norm_res.get("tokens", [])
        norm_text = norm_res.get("normalized_text", "")
        t1 = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 2: Sequence Tagging & Entity Span Aggregation
        # ---------------------------------------------------------------------
        if self.backend == "onnx":
            assert self._onnx_tagger is not None
            tag_res = self._onnx_tagger.tag_address(norm_res)
        else:
            assert self._crf_tagger is not None
            tag_res = self._crf_tagger.tag_address(norm_res)

        tags = tag_res.get("tags", [])
        entities = tag_res.get("entities", {})
        t2 = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 3: Context-Aware Administrative Homonym Disambiguation
        # ---------------------------------------------------------------------
        candidate_spans: List[str] = []
        for loc_key in ("SUBAREA", "THANA_UPAZILA", "DISTRICT"):
            if loc_key in entities:
                candidate_spans.extend(entities[loc_key])

        if not candidate_spans:
            # Fallback to all tokens if no specific location entity was extracted
            candidate_spans = tokens

        disambig_res = resolve_location_hierarchy(
            candidate_spans=candidate_spans,
            full_context_tokens=tokens,
            gazetteer=self.gazetteer,
            matcher=self.matcher,
            tagged_spans_by_type=entities,
        )

        resolved_node = disambig_res.get("resolved_node")
        hierarchy = disambig_res.get("hierarchy", {})
        hierarchy_bn = disambig_res.get("hierarchy_bn", {})
        confidence = float(disambig_res.get("confidence", 0.0))
        candidates_ranked = disambig_res.get("candidates_ranked", [])
        t3 = time.perf_counter()

        # Ambiguity check
        is_ambiguous = False
        if len(candidates_ranked) > 1:
            margin = candidates_ranked[0]["score"] - candidates_ranked[1]["score"]
            if margin <= 12.0:
                is_ambiguous = True

        disambiguation_candidates = [
            DisambiguationCandidate(
                id=c["id"],
                name=c["name"],
                level=c["level"],
                district=c.get("district"),
                score=round(c["score"], 2),
                reasons=c.get("reasons", []),
            )
            for c in candidates_ranked[:5]
        ]

        # ---------------------------------------------------------------------
        # Stage 4: Multi-Courier Hub Mapping & Truthful Geocoding
        # ---------------------------------------------------------------------
        courier_routing = self.courier_mapper.map_courier_routing(
            hierarchy=hierarchy,
            resolved_node=resolved_node,
        )
        geocode_info = resolve_truthful_geocode(
            resolved_node=resolved_node,
            hierarchy=hierarchy,
        )
        t4 = time.perf_counter()

        s1_ms = (t1 - t0) * 1000.0
        s2_ms = (t2 - t1) * 1000.0
        s3_ms = (t3 - t2) * 1000.0
        s4_ms = (t4 - t3) * 1000.0
        total_latency_ms = (t4 - start_time) * 1000.0

        stage_latencies = StageLatencies(
            stage1_normalizer=round(s1_ms, 3),
            stage2_tagger=round(s2_ms, 3),
            stage3_disambiguator=round(s3_ms, 3),
            stage4_courier_mapper=round(s4_ms, 3),
            total=round(total_latency_ms, 3),
        )

        parsed_components = ParsedComponents(
            house=" ".join(entities.get("HOUSE", [])) or None,
            floor_flat=" ".join(entities.get("FLOOR_FLAT", [])) or None,
            road=" ".join(entities.get("ROAD", [])) or None,
            block_sector=" ".join(entities.get("BLOCK_SECTOR", [])) or None,
            landmark=" ".join(entities.get("LANDMARK", [])) or None,
            subarea=hierarchy.get("subarea") or (" ".join(entities.get("SUBAREA", [])) if "SUBAREA" in entities else None),
            thana_upazila=hierarchy.get("thana") or hierarchy.get("upazila") or (" ".join(entities.get("THANA_UPAZILA", [])) if "THANA_UPAZILA" in entities else None),
            district=hierarchy.get("district") or (" ".join(entities.get("DISTRICT", [])) if "DISTRICT" in entities else None),
            division=hierarchy.get("division") or None,
        )

        return AddressParseResponse(
            raw_address=raw_address,
            normalized_address=norm_text,
            tokens=tokens,
            tags=tags,
            entities=entities,
            parsed_components=parsed_components,
            hierarchy=hierarchy,
            hierarchy_bn=hierarchy_bn,
            resolved_node_id=resolved_node.id if resolved_node else None,
            resolved_level=resolved_node.level.value if resolved_node else None,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            disambiguation_candidates=disambiguation_candidates,
            geocode=geocode_info,
            courier_routing=courier_routing,
            backend=self.backend,
            stage_latencies_ms=stage_latencies,
            latency_ms=round(total_latency_ms, 3),
        )

    def parse_batch(self, raw_addresses: List[str]) -> List[AddressParseResponse]:
        """Parses a batch of addresses sequentially."""
        return [self.parse(addr) for addr in raw_addresses]
