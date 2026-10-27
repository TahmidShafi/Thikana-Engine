"""
tests/test_pipeline.py

Unit and Latency Benchmark tests for Stage-4 Multi-Courier Hub Mapper (src/courier_mapper.py),
Modular ONNX Tagger with Viterbi decoding (src/tagger_onnx.py), and unified
ThikanaPipeline (src/pipeline.py).
"""

from __future__ import annotations
import numpy as np
import pytest
import time

from src.models import (
    AddressParseResponse,
    AdminLevel,
    AdminNode,
    CourierRouting,
    GeocodeInfo,
)
from src.gazetteer import BangladeshGazetteer
from src.phonetic_trie import PhoneticGazetteerMatcher
from src.courier_mapper import (
    MultiCourierHubMapper,
    sync_pathao_zones,
    resolve_truthful_geocode,
)
from src.tagger_onnx import (
    BIO_LABELS,
    ONNXSequenceTagger,
    ViterbiBIODecoder,
    vectorize_tokens,
)
from src.pipeline import ThikanaPipeline


@pytest.fixture(scope="module")
def gazetteer() -> BangladeshGazetteer:
    return BangladeshGazetteer()


@pytest.fixture(scope="module")
def matcher(gazetteer: BangladeshGazetteer) -> PhoneticGazetteerMatcher:
    return PhoneticGazetteerMatcher(gazetteer)


@pytest.fixture(scope="module")
def courier_mapper() -> MultiCourierHubMapper:
    return MultiCourierHubMapper()


@pytest.fixture(scope="module")
def pipeline(gazetteer: BangladeshGazetteer, matcher: PhoneticGazetteerMatcher) -> ThikanaPipeline:
    return ThikanaPipeline(backend="crf", gazetteer=gazetteer, matcher=matcher)


# =============================================================================
# 1. Courier Mapper Tests: Pathao, Steadfast, RedX
# =============================================================================
def test_courier_mapping_mirpur_10(courier_mapper: MultiCourierHubMapper):
    """Verify subarea-level mapping for Dhaka Mirpur 10."""
    hierarchy = {
        "division": "Dhaka",
        "district": "Dhaka",
        "thana": "Mirpur",
        "subarea": "Mirpur 10",
    }
    routing = courier_mapper.map_courier_routing(hierarchy)

    # Pathao
    assert routing.pathao.city_id == 1
    assert routing.pathao.zone_id == 101
    assert routing.pathao.area_id == 10110
    assert routing.pathao.area_name == "Mirpur 10"

    # Steadfast
    assert routing.steadfast.hub_code == "DHK-MIR"
    assert "Mirpur" in str(routing.steadfast.hub_name)

    # RedX
    assert routing.redx.delivery_area_id == 1001


def test_courier_mapping_tajmahal_road(courier_mapper: MultiCourierHubMapper):
    """Verify subarea-level mapping for Mohammadpur Tajmahal Road."""
    hierarchy = {
        "division": "Dhaka",
        "district": "Dhaka",
        "thana": "Mohammadpur",
        "subarea": "Tajmahal Road",
    }
    routing = courier_mapper.map_courier_routing(hierarchy)

    assert routing.pathao.city_id == 1
    assert routing.pathao.zone_id == 102
    assert routing.pathao.area_id == 10202
    assert routing.steadfast.hub_code == "DHK-MOH"
    assert routing.redx.delivery_area_id == 1002


def test_courier_mapping_chattogram_halishahar(courier_mapper: MultiCourierHubMapper):
    """Verify Chattogram Halishahar Block G routing."""
    hierarchy = {
        "division": "Chattogram",
        "district": "Chattogram",
        "thana": "Halishahar",
        "subarea": "Block G",
    }
    routing = courier_mapper.map_courier_routing(hierarchy)

    assert routing.pathao.city_id == 2
    assert routing.pathao.zone_id == 201
    assert routing.pathao.area_id == 20107
    assert routing.steadfast.hub_code == "CTG-HAL"
    assert routing.redx.delivery_area_id == 2001


def test_courier_mapping_district_fallback(courier_mapper: MultiCourierHubMapper):
    """Verify clean district fallback when thana/subarea is unknown or unseeded."""
    hierarchy = {
        "division": "Chattagram",
        "district": "Noakhali",
        "upazila": "UnseededRuralUpazila",
    }
    routing = courier_mapper.map_courier_routing(hierarchy)

    assert routing.pathao.city_id == 9
    assert routing.pathao.city_name == "Noakhali"
    assert routing.steadfast.hub_code == "NOA-SDR"
    assert routing.redx.delivery_area_id == 9000


def test_sync_pathao_zones_offline_fallback():
    """Verify sync_pathao_zones cleanly falls back to seed data when token is None."""
    data = sync_pathao_zones(bearer_token=None)
    assert "cities" in data
    assert "thanas" in data
    assert "dhaka" in data["cities"]
    assert data["cities"]["dhaka"]["city_id"] == 1


# =============================================================================
# 2. Truthful Geocoding Precision Tests
# =============================================================================
def test_truthful_geocoding_district_centroid(gazetteer: BangladeshGazetteer):
    """
    When coordinates originate from the District node (from nuhil/bangladesh-geocode),
    precision must be truthfully tagged as 'DISTRICT_CENTROID'.
    """
    # A rural union or thana that only has parent district coordinates
    dist_node = next(d for d in gazetteer.districts if d.name.lower() == "dhaka")
    assert dist_node is not None

    upz_nodes = [c for c in dist_node.children if c.level == AdminLevel.UPAZILA]
    assert len(upz_nodes) > 0
    test_node = upz_nodes[0]

    geocode = resolve_truthful_geocode(test_node, {"district": "Dhaka", "upazila": test_node.name})
    assert geocode.lat is not None
    assert geocode.lon is not None
    assert geocode.precision == "DISTRICT_CENTROID"
    assert geocode.source_node == "Dhaka"


def test_truthful_geocoding_subarea_centroid():
    """
    When coordinates explicitly originate from a subarea node (e.g. metro seed),
    precision must be truthfully tagged as 'SUBAREA_CENTROID'.
    """
    node = AdminNode(
        id="metro_subarea_1",
        name="Mirpur 10",
        bn_name="মিরপুর ১০",
        level=AdminLevel.SUBAREA,
        lat=23.8069,
        lon=90.3687,
    )
    geocode = resolve_truthful_geocode(node, {"district": "Dhaka", "subarea": "Mirpur 10"})
    assert geocode.lat == 23.8069
    assert geocode.lon == 90.3687
    assert geocode.precision == "SUBAREA_CENTROID"
    assert geocode.source_node == "Mirpur 10"


def test_truthful_geocoding_unknown():
    """Verify unknown precision when node is None."""
    geocode = resolve_truthful_geocode(None, {})
    assert geocode.lat is None
    assert geocode.lon is None
    assert geocode.precision == "UNKNOWN"


# =============================================================================
# 3. ONNX Sequence Tagger & Viterbi BIO Decoder Tests
# =============================================================================
def test_viterbi_decoder_prevents_orphan_i_tags():
    """
    Critical requirement: Viterbi BIO decoder must NEVER produce an orphan I-* tag
    even when emissions strongly favor an I-* tag at the beginning or after O.
    """
    decoder = ViterbiBIODecoder(BIO_LABELS)

    # Synthetic emissions of shape (3, num_labels)
    # Token 0: strongly biased towards I-ROAD (an illegal start tag!)
    # Token 1: biased towards O
    # Token 2: strongly biased towards I-HOUSE (an illegal transition from O!)
    emissions = np.zeros((3, len(BIO_LABELS)), dtype=np.float32)
    emissions[0, BIO_LABELS.index("I-ROAD")] = 100.0  # Illegal start
    emissions[0, BIO_LABELS.index("B-ROAD")] = 50.0   # Legal fallback

    emissions[1, BIO_LABELS.index("O")] = 100.0

    emissions[2, BIO_LABELS.index("I-HOUSE")] = 100.0  # Illegal after O
    emissions[2, BIO_LABELS.index("B-HOUSE")] = 50.0   # Legal fallback

    decoded_tags = decoder.decode(emissions)

    # Token 0 must NOT be I-ROAD (must decode to a legal start tag like B-ROAD)
    assert not decoded_tags[0].startswith("I-")
    assert decoded_tags[0] == "B-ROAD"

    # Token 1 is O
    assert decoded_tags[1] == "O"

    # Token 2 must NOT be I-HOUSE (cannot follow O; must decode to B-HOUSE)
    assert not decoded_tags[2].startswith("I-")
    assert decoded_tags[2] == "B-HOUSE"


def test_onnx_tagger_inference_and_session_options(matcher: PhoneticGazetteerMatcher):
    """Verify ONNXSequenceTagger session creation, thread options, and tag prediction."""
    tagger = ONNXSequenceTagger(matcher=matcher, intra_op_num_threads=2)

    # Check session options
    assert tagger.session_options.intra_op_num_threads == 2

    # Predict tokens
    tokens = ["H#", "45", "RD#", "Tajmahal", "Mohammadpur", "Dhaka"]
    tags = tagger.predict_tokens(tokens)

    assert len(tags) == len(tokens)
    for tag in tags:
        assert tag in BIO_LABELS
    # Ensure no orphan I-* at start
    assert not tags[0].startswith("I-")

    # Verify tag_address
    res = tagger.tag_address("H# 45, RD# Tajmahal, Mohammadpur, Dhaka")
    assert "tokens" in res
    assert "tags" in res
    assert "entities" in res
    assert res["latency_ms"] >= 0.0


# =============================================================================
# 4. End-to-End ThikanaPipeline Tests (CRF & ONNX)
# =============================================================================
def test_pipeline_end_to_end_crf(pipeline: ThikanaPipeline):
    """Test full end-to-end address resolution with backend='crf'."""
    raw = "House 12, Road 4, Mirpur 10, Dhaka"
    resp = pipeline.parse(raw)

    assert isinstance(resp, AddressParseResponse)
    assert resp.backend == "crf"
    assert resp.raw_address == raw
    assert len(resp.tokens) > 0
    assert len(resp.tags) == len(resp.tokens)

    # Administrative hierarchy
    assert resp.hierarchy.get("district") == "Dhaka"
    assert resp.hierarchy.get("thana") == "Mirpur"
    assert resp.hierarchy.get("subarea") == "Mirpur 10"

    # Courier routing
    assert resp.courier_routing.pathao.city_id == 1
    assert resp.courier_routing.pathao.zone_id == 101
    assert resp.courier_routing.pathao.area_id == 10110
    assert resp.courier_routing.steadfast.hub_code == "DHK-MIR"
    assert resp.courier_routing.redx.delivery_area_id == 1001

    # Geocoding precision
    assert resp.geocode.precision in ("SUBAREA_CENTROID", "DISTRICT_CENTROID")
    assert resp.geocode.lat is not None
    assert resp.latency_ms > 0.0


def test_pipeline_end_to_end_rural_address(pipeline: ThikanaPipeline):
    """Test full end-to-end address resolution for a rural Union in Noakhali."""
    raw = "Village: Mohammadpur, Upazila: Chatkhil, Noakhali"
    resp = pipeline.parse(raw)

    assert resp.hierarchy.get("district") == "Noakhali"
    assert resp.hierarchy.get("upazila") == "Chatkhil"
    assert resp.courier_routing.steadfast.hub_code in ("NOA-CHK", "NOA-SDR")
    assert resp.geocode.precision in ("DISTRICT_CENTROID", "SUBAREA_CENTROID")


def test_pipeline_backend_switching(pipeline: ThikanaPipeline):
    """Verify dynamic switching between 'crf' and 'onnx' backends."""
    raw = "Holding 45, Tajmahal Road, Mohammadpur, Dhaka"

    # 1. Run CRF
    pipeline.set_backend("crf")
    resp_crf = pipeline.parse(raw)
    assert resp_crf.backend == "crf"
    assert resp_crf.hierarchy.get("district") == "Dhaka"

    # 2. Switch to ONNX
    pipeline.set_backend("onnx")
    resp_onnx = pipeline.parse(raw)
    assert resp_onnx.backend == "onnx"
    assert resp_onnx.hierarchy.get("district") == "Dhaka"

    # Reset back to CRF
    pipeline.set_backend("crf")
    assert pipeline.backend == "crf"


def test_pipeline_parse_batch(pipeline: ThikanaPipeline):
    """Verify batch parsing functionality."""
    addresses = [
        "House 12, Road 4, Dhanmondi, Dhaka",
        "Holding 88, Sector 4, Uttara, Dhaka",
        "Halishahar Block G, Chattogram",
    ]
    responses = pipeline.parse_batch(addresses)
    assert len(responses) == 3
    for resp in responses:
        assert isinstance(resp, AddressParseResponse)
        assert resp.hierarchy.get("district") in ("Dhaka", "Chattogram")


# =============================================================================
# 5. Latency Benchmark Test
# =============================================================================
def test_pipeline_latency_benchmark(pipeline: ThikanaPipeline):
    """
    Benchmarks end-to-end pipeline latency across 50 iterations.
    Ensures p50 and p95 CPU execution latency remain ultra-fast.
    """
    raw_addr = "Flat 4A, House 24, Road 11, Block D, Banani, Dhaka"
    latencies: list[float] = []

    # Warm-up run
    pipeline.parse(raw_addr)

    for _ in range(50):
        t0 = time.perf_counter()
        resp = pipeline.parse(raw_addr)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))

    print(f"\n[BENCHMARK] End-to-End ThikanaPipeline Latency over 50 runs:")
    print(f"  Average: {avg_lat:.2f} ms")
    print(f"  p50:     {p50:.2f} ms")
    print(f"  p95:     {p95:.2f} ms")
    print(f"  p99:     {p99:.2f} ms")

    # Assert reasonable execution speed (p50 < 10ms)
    assert p50 < 10.0, f"Expected p50 < 10ms, got {p50:.2f}ms"
