"""
tests/test_api.py

FastAPI Microservice Integration Tests for Thikana-Engine:
- GET /health
- GET / (Playground UI)
- POST /v1/parse (Standard, Homonym, Distractor, Rural, Invalid backend, Validation)
- POST /v1/batch-resolve (Batch resolution, throughput stats, max limit validation)
"""

from __future__ import annotations
from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


# =============================================================================
# 1. Health & Playground UI Endpoint Tests
# =============================================================================
def test_health_check(client: TestClient):
    """Verify /health returns 200 with service operational status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "thikana-engine"
    assert data["version"] == "1.0.0"
    assert data["gazetteer_nodes_loaded"] >= 5000


def test_serve_playground_ui(client: TestClient):
    """Verify GET / returns the static index.html playground."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Thikana-Engine" in response.text
    assert "Leaflet" in response.text or "leaflet" in response.text


# =============================================================================
# 2. Single Address Parse Tests (POST /v1/parse)
# =============================================================================
def test_parse_single_urban_dhaka(client: TestClient):
    """Verify parsing of standard urban Dhaka address with subarea precision."""
    payload = {
        "address": "House 12, Road 4, Mirpur 10, Dhaka",
        "backend": "crf",
    }
    response = client.post("/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["raw_address"] == payload["address"]
    assert len(data["tokens"]) > 0
    assert len(data["tags"]) == len(data["tokens"])

    # Fine-grained components
    comp = data["parsed_components"]
    assert comp["district"] == "Dhaka"
    assert comp["thana_upazila"] == "Mirpur"
    assert comp["subarea"] == "Mirpur 10"

    # Courier routing
    courier = data["courier_routing"]
    assert courier["pathao"]["city_id"] == 1
    assert courier["pathao"]["zone_id"] == 101
    assert courier["pathao"]["area_id"] == 10110
    assert courier["steadfast"]["hub_code"] == "DHK-MIR"
    assert courier["redx"]["delivery_area_id"] == 1001

    # Truthful Geocoding
    geo = data["geocode"]
    assert geo["precision"] in ("SUBAREA_CENTROID", "DISTRICT_CENTROID")
    assert geo["lat"] is not None
    assert geo["lon"] is not None

    # Latencies & SLA check (< 10ms)
    lats = data["stage_latencies_ms"]
    assert lats["total"] < 10.0
    assert lats["stage1_normalizer"] >= 0.0
    assert lats["stage2_tagger"] >= 0.0
    assert lats["stage3_disambiguator"] >= 0.0
    assert lats["stage4_courier_mapper"] >= 0.0


def test_parse_homonym_disambiguation(client: TestClient):
    """Verify homonym disambiguation for Mohammadpur without explicit District."""
    payload = {
        "address": "Holding 45, Tajmahal Road, Mohammadpur",
        "backend": "crf",
    }
    response = client.post("/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Disambiguates to Dhaka (due to Tajmahal Road child co-occurrence)
    assert data["hierarchy"]["district"] == "Dhaka"
    assert data["hierarchy"]["thana"] == "Mohammadpur"
    assert len(data["disambiguation_candidates"]) > 0


def test_parse_distractor_address(client: TestClient):
    """Verify anti-greedy resolution when a distractor institution is mentioned."""
    payload = {
        "address": "Uttara Bank Limited, Road 27, Dhanmondi, Dhaka",
        "backend": "crf",
    }
    response = client.post("/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Must resolve to Dhanmondi, not Uttara
    assert data["hierarchy"]["thana"] == "Dhanmondi"
    assert data["hierarchy"]["district"] == "Dhaka"


def test_parse_rural_bangla_address(client: TestClient):
    """Verify resolution of rural address with native Bengali script."""
    payload = {
        "address": "গ্রাম: মোহাম্মদপুর, উপজেলা: চাটখিল, নোয়াখালী",
        "backend": "crf",
    }
    response = client.post("/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["hierarchy"]["district"] == "Noakhali"
    assert data["hierarchy"]["upazila"] == "Chatkhil"
    assert data["courier_routing"]["steadfast"]["hub_code"] in ("NOA-CHK", "NOA-SDR")


def test_parse_onnx_backend(client: TestClient):
    """Verify parsing works with ONNX backend."""
    payload = {
        "address": "Halishahar Block G, Chattogram",
        "backend": "onnx",
    }
    response = client.post("/v1/parse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "onnx"
    assert data["hierarchy"]["district"] in ("Chattogram", "Chittagong")


def test_parse_validation_errors(client: TestClient):
    """Verify input validation returns 400 or 422 appropriately."""
    # Empty address string
    resp_empty = client.post("/v1/parse", json={"address": ""})
    assert resp_empty.status_code == 422

    # Invalid backend parameter
    resp_invalid_backend = client.post(
        "/v1/parse",
        json={"address": "Dhanmondi, Dhaka", "backend": "quantum_neural_net"}
    )
    assert resp_invalid_backend.status_code == 400


# =============================================================================
# 3. Batch Address Resolution Tests (POST /v1/batch-resolve)
# =============================================================================
def test_batch_resolve_success(client: TestClient):
    """Verify batch endpoint parses multiple addresses and computes throughput."""
    addresses = [
        "H# 45, Rd 4, Mirpur 10, Dhaka",
        "Holding 45, Tajmahal Road, Mohammadpur",
        "Halishahar Block G, Chattogram",
        "Uttara Bank Limited, Road 27, Dhanmondi, Dhaka",
        "গ্রাম: মোহাম্মদপুর, উপজেলা: চাটখিল, নোয়াখালী",
    ]
    response = client.post(
        "/v1/batch-resolve",
        json={"addresses": addresses, "backend": "crf"},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["total_addresses"] == 5
    assert data["total_latency_ms"] > 0.0
    assert data["avg_latency_ms"] > 0.0
    assert data["throughput_addr_per_sec"] > 0.0
    assert len(data["results"]) == 5

    # Check individual results
    assert data["results"][0]["hierarchy"]["district"] == "Dhaka"
    assert data["results"][4]["hierarchy"]["district"] == "Noakhali"


def test_batch_resolve_limit_exceeded(client: TestClient):
    """Verify batch endpoint rejects requests with > 100 addresses."""
    excessive_addresses = [f"Address {i}, Dhaka" for i in range(105)]
    response = client.post(
        "/v1/batch-resolve",
        json={"addresses": excessive_addresses, "backend": "crf"},
    )
    assert response.status_code == 422
