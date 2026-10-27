"""
src/courier_mapper.py

Stage-4 Multi-Courier Hub Mapper & Truthful Geocoder for Thikana-Engine.
Maps resolved administrative hierarchies to:
  - Pathao Merchant Routing: city_id, zone_id, area_id
  - Steadfast Courier Hub: hub_code, hub_name
  - RedX Logistics: delivery_area_id, area_name

Provides sync_pathao_zones() for dynamic API updates with fallback to
bundled seed tables (data/raw/courier_zones_seed.json), and guarantees
truthful geocoding precision tagging (SUBAREA_CENTROID vs. DISTRICT_CENTROID).
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from .models import (
    AdminLevel,
    AdminNode,
    CourierRouting,
    GeocodeInfo,
    PathaoRouting,
    RedxRouting,
    SteadfastRouting,
)

logger = logging.getLogger(__name__)

DEFAULT_SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "courier_zones_seed.json"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "pathao_zones.json"


# -----------------------------------------------------------------------------
# Pathao API Dynamic Synchronization Utility
# -----------------------------------------------------------------------------
def sync_pathao_zones(
    bearer_token: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Dynamically fetches live city/zone/area IDs from Pathao's merchant API.
    If bearer_token is provided and the network request succeeds, updates the local cache.
    If bearer_token is None or request fails, cleanly falls back to cached data or local seed table.

    Returns:
      A dictionary containing Pathao city, zone, and area mappings.
    """
    target_cache = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    if bearer_token and bearer_token.strip():
        headers = {
            "Authorization": f"Bearer {bearer_token.strip()}",
            "Accept": "application/json",
            "User-Agent": "Thikana-Engine/0.5.0",
        }
        try:
            logger.info("Attempting live sync with Pathao Merchant API...")
            # 1. Fetch cities
            resp = requests.get(
                "https://api-hermes.pathao.com/aladdin/api/v1/city-list",
                headers=headers,
                timeout=timeout_sec,
            )
            if resp.status_code == 200:
                cities_data = resp.json().get("data", {}).get("data", [])
                synced_payload: Dict[str, Any] = {"cities": {}, "zones": {}, "timestamp": None}

                for city in cities_data:
                    c_id = city.get("city_id")
                    c_name = city.get("city_name", "").strip().lower()
                    if c_id and c_name:
                        synced_payload["cities"][c_name] = {"city_id": c_id, "name": city.get("city_name")}

                        # Fetch zones for city
                        z_resp = requests.get(
                            f"https://api-hermes.pathao.com/aladdin/api/v1/cities/{c_id}/zone-list",
                            headers=headers,
                            timeout=timeout_sec,
                        )
                        if z_resp.status_code == 200:
                            zones = z_resp.json().get("data", {}).get("data", [])
                            for z in zones:
                                z_id = z.get("zone_id")
                                z_name = z.get("zone_name", "").strip().lower()
                                key = f"{c_name}:{z_name}"
                                synced_payload["zones"][key] = {
                                    "city_id": c_id,
                                    "zone_id": z_id,
                                    "city_name": city.get("city_name"),
                                    "zone_name": z.get("zone_name"),
                                }

                # Cache dynamically fetched data
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(target_cache, "w", encoding="utf-8") as f:
                    json.dump(synced_payload, f, indent=2, ensure_ascii=False)
                logger.info(f"Pathao zones live sync successful -> cached to {target_cache}")
                return synced_payload
            else:
                logger.warning(f"Pathao API returned HTTP {resp.status_code}, falling back to local seed.")
        except Exception as e:
            logger.warning(f"Failed to fetch live Pathao API ({e}); falling back to local seed/cache.")

    # Fallback: check dynamic cache first, then bundled seed table
    if target_cache.exists():
        try:
            with open(target_cache, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache at {target_cache}: {e}")

    # Bundled seed table fallback
    if DEFAULT_SEED_PATH.exists():
        with open(DEFAULT_SEED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"cities": {}, "thanas": {}, "subareas": {}, "district_fallbacks": {}}


# -----------------------------------------------------------------------------
# Multi-Courier Hub Mapper Class
# -----------------------------------------------------------------------------
class MultiCourierHubMapper:
    """
    Stage-4 Routing Mapper that produces standard delivery payloads for
    Pathao, Steadfast, and RedX from resolved administrative entities.
    """

    def __init__(
        self,
        seed_path: Optional[str | Path] = None,
        bearer_token: Optional[str] = None,
    ):
        self.seed_path = Path(seed_path) if seed_path else DEFAULT_SEED_PATH
        self.data: Dict[str, Any] = {}
        self._load_seed_data(bearer_token)

    def _load_seed_data(self, bearer_token: Optional[str] = None) -> None:
        """Loads courier routing tables with offline fallback."""
        if bearer_token:
            self.data = sync_pathao_zones(bearer_token)
        elif self.seed_path.exists():
            with open(self.seed_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"cities": {}, "thanas": {}, "subareas": {}, "district_fallbacks": {}}

    def map_courier_routing(
        self,
        hierarchy: Dict[str, str],
        resolved_node: Optional[AdminNode] = None,
    ) -> CourierRouting:
        """
        Maps resolved hierarchy to Pathao, Steadfast, and RedX delivery codes.
        Hierarchy traversal order:
          1. Subarea exact match
          2. Thana/Upazila exact match
          3. District fallback
        """
        dist_key = (hierarchy.get("district") or "").strip().lower()
        thana_key = (hierarchy.get("thana") or hierarchy.get("upazila") or "").strip().lower()
        subarea_key = (hierarchy.get("subarea") or hierarchy.get("union") or "").strip().lower()

        subareas_map: Dict[str, Any] = self.data.get("subareas", {})
        thanas_map: Dict[str, Any] = self.data.get("thanas", {})
        district_fallbacks: Dict[str, Any] = self.data.get("district_fallbacks", {})

        # 1. Subarea Match
        full_subarea_key = f"{dist_key}:{thana_key}:{subarea_key}"
        if full_subarea_key in subareas_map:
            entry = subareas_map[full_subarea_key]
            return self._build_routing(entry, dist_key, thana_key, subarea_key)

        # 2. Thana / Upazila Match
        full_thana_key = f"{dist_key}:{thana_key}"
        if full_thana_key in thanas_map:
            entry = thanas_map[full_thana_key]
            return self._build_routing(entry, dist_key, thana_key, subarea_key)

        # 3. District Fallback
        if dist_key in district_fallbacks:
            entry = district_fallbacks[dist_key]
            return self._build_routing(entry, dist_key, thana_key, subarea_key)

        # 4. Unknown Generic Fallback
        return CourierRouting(
            pathao=PathaoRouting(city_name=dist_key.title() if dist_key else None),
            steadfast=SteadfastRouting(hub_name=f"{dist_key.title()} Hub" if dist_key else None),
            redx=RedxRouting(area_name=dist_key.title() if dist_key else None),
        )

    def _build_routing(
        self,
        entry: Dict[str, Any],
        dist_key: str,
        thana_key: str,
        subarea_key: str,
    ) -> CourierRouting:
        """Constructs CourierRouting Pydantic model from raw entry dictionary."""
        p_data = entry.get("pathao", {})
        s_data = entry.get("steadfast", {})
        r_data = entry.get("redx", {})

        pathao_rt = PathaoRouting(
            city_id=p_data.get("city_id"),
            zone_id=p_data.get("zone_id"),
            area_id=p_data.get("area_id"),
            city_name=p_data.get("city_name") or dist_key.title(),
            zone_name=p_data.get("zone_name") or thana_key.title(),
            area_name=p_data.get("area_name") or (subarea_key.title() if subarea_key else None),
        )

        steadfast_rt = SteadfastRouting(
            hub_code=s_data.get("hub_code"),
            hub_name=s_data.get("hub_name"),
        )

        redx_rt = RedxRouting(
            delivery_area_id=r_data.get("delivery_area_id"),
            area_name=r_data.get("area_name"),
        )

        return CourierRouting(
            pathao=pathao_rt,
            steadfast=steadfast_rt,
            redx=redx_rt,
        )


# -----------------------------------------------------------------------------
# Truthful Geocoding Precision Resolution
# -----------------------------------------------------------------------------
def resolve_truthful_geocode(
    resolved_node: Optional[AdminNode],
    hierarchy: Dict[str, str],
) -> GeocodeInfo:
    """
    Truthfully determines geographic coordinates and explicit precision level:
      - 'SUBAREA_CENTROID': Coordinates explicitly originated from the subarea node.
      - 'DISTRICT_CENTROID': Coordinates fell back to the enclosing District centroid.
      - 'UNKNOWN': No coordinates available.
    """
    if resolved_node is None:
        return GeocodeInfo(lat=None, lon=None, precision="UNKNOWN", source_node=None)

    # 1. Did the resolved node itself have coordinates?
    if resolved_node.lat is not None and resolved_node.lon is not None:
        if resolved_node.level in (AdminLevel.SUBAREA, AdminLevel.UNION, AdminLevel.THANA, AdminLevel.UPAZILA):
            return GeocodeInfo(
                lat=resolved_node.lat,
                lon=resolved_node.lon,
                precision="SUBAREA_CENTROID",
                source_node=resolved_node.name,
            )
        elif resolved_node.level == AdminLevel.DISTRICT:
            return GeocodeInfo(
                lat=resolved_node.lat,
                lon=resolved_node.lon,
                precision="DISTRICT_CENTROID",
                source_node=resolved_node.name,
            )

    # 2. Check enclosing district coordinates (nuhil/bangladesh-geocode district centroids)
    dist = resolved_node.district
    if dist is not None and dist.lat is not None and dist.lon is not None:
        return GeocodeInfo(
            lat=dist.lat,
            lon=dist.lon,
            precision="DISTRICT_CENTROID",
            source_node=dist.name,
        )

    # 3. Check enclosing division coordinates
    div = resolved_node.division
    if div is not None and div.lat is not None and div.lon is not None:
        return GeocodeInfo(
            lat=div.lat,
            lon=div.lon,
            precision="DISTRICT_CENTROID",
            source_node=div.name,
        )

    return GeocodeInfo(lat=None, lon=None, precision="UNKNOWN", source_node=None)
