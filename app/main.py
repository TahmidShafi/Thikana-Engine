"""
app/main.py

FastAPI Microservice for Thikana-Engine:
- POST /v1/parse: Single address parser with stage latency breakdown & courier routing.
- POST /v1/batch-resolve: Batch address parser (up to 100 addresses) with throughput statistics.
- GET /: Interactive Dark-Mode Single-Page Developer Playground.
- GET /health: Health check and status endpoint.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import AddressParseResponse
from src.pipeline import ThikanaPipeline

logger = logging.getLogger("thikana.api")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Singleton pipeline instance
_pipeline: Optional[ThikanaPipeline] = None


def get_pipeline() -> ThikanaPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing singleton ThikanaPipeline...")
        _pipeline = ThikanaPipeline(backend="crf")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup warmup
    logger.info("Starting up Thikana-Engine API...")
    pipeline = get_pipeline()
    # Warmup parse
    pipeline.parse("House 1, Road 1, Dhanmondi, Dhaka")
    logger.info("Thikana-Engine Pipeline warmed up and ready.")
    yield
    logger.info("Shutting down Thikana-Engine API.")


app = FastAPI(
    title="Thikana-Engine API",
    version="1.0.0",
    description=(
        "Ultra-low-latency, multi-tier Bangladeshi address parser, "
        "homonym disambiguator, and multi-courier routing microservice."
    ),
    lifespan=lifespan,
)

# Enable CORS for cross-origin playground and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for playground assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class ParseRequest(BaseModel):
    address: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Raw Bangla or Romanized Banglish address string.",
        examples=["Holding 45, Tajmahal Road, Mohammadpur, Dhaka"],
    )
    backend: Optional[str] = Field(
        "crf",
        description="Sequence tagger backend: 'crf' (default) or 'onnx'.",
        examples=["crf"],
    )


class BatchResolveRequest(BaseModel):
    addresses: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of raw addresses to resolve in batch (maximum 100 addresses).",
        examples=[
            [
                "H# 45, Rd 4, Mirpur 10, Dhaka",
                "Holding 45, Tajmahal Road, Mohammadpur",
                "Halishahar Block G, Chattogram",
                "Uttara Bank Limited, Road 27, Dhanmondi, Dhaka",
                "গ্রাম: মোহাম্মদপুর, উপজেলা: চাটখিল, নোয়াখালী",
            ]
        ],
    )
    backend: Optional[str] = Field(
        "crf",
        description="Sequence tagger backend: 'crf' (default) or 'onnx'.",
        examples=["crf"],
    )


class BatchResolveResponse(BaseModel):
    total_addresses: int
    total_latency_ms: float
    avg_latency_ms: float
    throughput_addr_per_sec: float
    results: List[AddressParseResponse]


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", summary="Serve Developer Playground UI")
async def serve_playground():
    """Serves the interactive single-page Dark Mode Developer Playground."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground UI file app/static/index.html not found.",
        )
    return FileResponse(index_file)


@app.get("/health", summary="Health Check")
async def health_check():
    """Returns microservice health and operational status."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "service": "thikana-engine",
        "version": "1.0.0",
        "backend": pipeline.backend,
        "gazetteer_nodes_loaded": len(pipeline.gazetteer.nodes_by_id),
    }


@app.post(
    "/v1/parse",
    response_model=AddressParseResponse,
    summary="Parse Single Address",
    description=(
        "Executes 4-stage pipeline on a single address: Script/Numeral Normalization, "
        "BIO Sequence Tagging, Context-Aware Disambiguation, and Multi-Courier Hub Mapping."
    ),
)
async def parse_address(request: ParseRequest) -> AddressParseResponse:
    pipeline = get_pipeline()
    requested_backend = (request.backend or "crf").lower()
    if requested_backend not in ("crf", "onnx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported backend '{request.backend}'. Must be 'crf' or 'onnx'.",
        )

    if pipeline.backend != requested_backend:
        pipeline.set_backend(requested_backend)

    return pipeline.parse(request.address)


@app.post(
    "/v1/batch-resolve",
    response_model=BatchResolveResponse,
    summary="Batch Address Resolution",
    description="Parses up to 100 addresses in a single call and returns aggregate throughput metrics.",
)
async def batch_resolve_addresses(request: BatchResolveRequest) -> BatchResolveResponse:
    if len(request.addresses) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size exceeds maximum limit of 100 addresses (got {len(request.addresses)}).",
        )

    pipeline = get_pipeline()
    requested_backend = (request.backend or "crf").lower()
    if requested_backend not in ("crf", "onnx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported backend '{request.backend}'. Must be 'crf' or 'onnx'.",
        )

    if pipeline.backend != requested_backend:
        pipeline.set_backend(requested_backend)

    t_start = time.perf_counter()
    results: List[AddressParseResponse] = []
    for addr in request.addresses:
        results.append(pipeline.parse(addr))
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    count = len(results)
    avg_lat = total_time_ms / max(1, count)
    throughput = (count / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    return BatchResolveResponse(
        total_addresses=count,
        total_latency_ms=round(total_time_ms, 3),
        avg_latency_ms=round(avg_lat, 3),
        throughput_addr_per_sec=round(throughput, 1),
        results=results,
    )
