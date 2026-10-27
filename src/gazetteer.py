"""
src/gazetteer.py

Loads administrative geocode datasets and custom metro seeds into an in-memory
parent-child hierarchy: Division -> District -> Upazila/Thana -> Union/Subarea.
Provides fast name lookup supporting homonyms and bilingual (English/Bengali) queries.
"""

from __future__ import annotations
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .models import AdminLevel, AdminNode



logger = logging.getLogger(__name__)

# Known historical renamings, transliteration variants, and orthographic variations
COMMON_ALIASES: Dict[str, List[str]] = {
    # English division & district variants
    "chattogram": ["chittagong", "chattagram"],
    "chittagong": ["chattogram", "chattagram"],
    "chattagram": ["chattogram", "chittagong"],
    "cumilla": ["comilla"],
    "comilla": ["cumilla"],
    "barishal": ["barisal"],
    "barisal": ["barishal"],
    "jashore": ["jessore"],
    "jessore": ["jashore"],
    "bogura": ["bogra"],
    "bogra": ["bogura"],
    # English spelling variants
    "muhammadpur": ["mohammadpur"],
    "mohammadpur": ["muhammadpur", "mahammadpur"],
    "hatia": ["hatiya"],
    "hatiya": ["hatia"],
    "sonaimori": ["sonaimuri"],
    "sonaimuri": ["sonaimori"],
    "bogra sadar": ["bogura sadar"],
    "bogura sadar": ["bogra sadar"],
    "doulatpur": ["daulatpur"],
    "daulatpur": ["doulatpur"],
    "mohonpur": ["mohanpur"],
    "mohanpur": ["mohonpur"],
    # Bengali common orthographic & historical variants
    "মহম্মদপুর": ["মোহাম্মদপুর"],
    "মোহাম্মদপুর": ["মহম্মদপুর"],
    "কুমিল্লা": ["কোমিল্লা"],
    "চট্টগ্রাম": ["চিটাগাং"],
    "চিটাগাং": ["চট্টগ্রাম"],
    "যশোর": ["যশোহর"],
}

OFFICIAL_2018_CANONICAL_NAMES: Dict[str, str] = {
    "chittagong": "Chattogram",
    "chattagram": "Chattogram",
    "comilla": "Cumilla",
    "barisal": "Barishal",
    "jessore": "Jashore",
    "bogra": "Bogura",
}


def canonicalize_admin_name(name: str) -> str:
    """Canonicalizes official 2018 English spellings for divisions/districts."""
    return OFFICIAL_2018_CANONICAL_NAMES.get(name.strip().lower(), name.strip())


def _normalize_name(name: str) -> str:
    """Normalize a query or entity name for exact index lookup."""
    if not name:
        return ""
    # Normalize unicode to standard NFC form (crucial for Bengali conjuncts and nuktas)
    norm = unicodedata.normalize("NFC", name)
    # Collapse multiple whitespace characters and lower-case
    cleaned = re.sub(r"\s+", " ", norm.strip()).casefold()
    return cleaned


def _slugify(text: str) -> str:
    """Generate a clean ASCII slug for custom IDs."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "item"


class BangladeshGazetteer:
    """
    In-memory administrative gazetteer for Bangladesh.

    Hierarchy:
      Division (Tier 1)
        └── District (Tier 2)
              └── Upazila / Metro Thana (Tier 3)
                    └── Union / Metro Subarea (Tier 4)
    """

    def __init__(
        self,
        raw_data_dir: Optional[Union[str, Path]] = None,
        divisions_file: Optional[Union[str, Path]] = None,
        districts_file: Optional[Union[str, Path]] = None,
        upazilas_file: Optional[Union[str, Path]] = None,
        unions_file: Optional[Union[str, Path]] = None,
        metro_thanas_file: Optional[Union[str, Path]] = None,
        auto_load: bool = True,
    ) -> None:
        """
        Initialize the gazetteer. If auto_load=True, loads data immediately.
        """
        base_dir = (
            Path(raw_data_dir)
            if raw_data_dir
            else Path(__file__).resolve().parent.parent / "data" / "raw"
        )
        self.raw_data_dir = base_dir

        self.divisions_file = (
            Path(divisions_file) if divisions_file else base_dir / "divisions.json"
        )
        self.districts_file = (
            Path(districts_file) if districts_file else base_dir / "districts.json"
        )
        self.upazilas_file = (
            Path(upazilas_file) if upazilas_file else base_dir / "upazilas.json"
        )
        self.unions_file = (
            Path(unions_file) if unions_file else base_dir / "unions.json"
        )
        self.metro_thanas_file = (
            Path(metro_thanas_file)
            if metro_thanas_file
            else base_dir / "dhaka_ctg_thanas.json"
        )

        # Storage collections (maps ID -> AdminNode)
        self._divisions_by_id: Dict[str, AdminNode] = {}
        self._districts_by_id: Dict[str, AdminNode] = {}
        self._upazilas_by_id: Dict[str, AdminNode] = {}  # Rural Upazilas & Metro Thanas
        self._unions_by_id: Dict[str, AdminNode] = {}  # Rural Unions & Metro Subareas

        # Canonical unique node list
        self._all_nodes: List[AdminNode] = []
        self._all_nodes_by_id: Dict[str, AdminNode] = {}

        # Fast lookup indexes
        self._name_index: Dict[str, List[AdminNode]] = {}

        if auto_load:
            self.load_data()

    # -------------------------------------------------------------------------
    # Properties & Accessors
    # -------------------------------------------------------------------------
    @property
    def divisions(self) -> List[AdminNode]:
        """List of all unique administrative divisions."""
        seen: Set[str] = set()
        divs: List[AdminNode] = []
        for node in self._divisions_by_id.values():
            if node.id not in seen:
                seen.add(node.id)
                divs.append(node)
        return divs

    @property
    def districts(self) -> List[AdminNode]:
        """List of all unique administrative districts."""
        seen: Set[str] = set()
        dists: List[AdminNode] = []
        for node in self._districts_by_id.values():
            if node.id not in seen:
                seen.add(node.id)
                dists.append(node)
        return dists

    @property
    def upazilas(self) -> List[AdminNode]:
        """List of all unique Upazilas and Metro Thanas."""
        seen: Set[str] = set()
        upzs: List[AdminNode] = []
        for node in self._upazilas_by_id.values():
            if node.id not in seen:
                seen.add(node.id)
                upzs.append(node)
        return upzs

    @property
    def unions(self) -> List[AdminNode]:
        """List of all unique Unions and Metro Subareas."""
        seen: Set[str] = set()
        uns: List[AdminNode] = []
        for node in self._unions_by_id.values():
            if node.id not in seen:
                seen.add(node.id)
                uns.append(node)
        return uns

    @property
    def all_nodes(self) -> List[AdminNode]:
        """List of all unique administrative nodes in the gazetteer."""
        return list(self._all_nodes)

    @property
    def nodes_by_id(self) -> Dict[str, AdminNode]:
        """Dictionary mapping node IDs to AdminNode objects."""
        return dict(self._all_nodes_by_id)

    def stats(self) -> Dict[str, int]:
        """Summary statistics of loaded gazetteer entities."""
        upz_list = self.upazilas
        union_list = self.unions
        metro_thanas = sum(1 for u in upz_list if u.level == AdminLevel.THANA)
        metro_subareas = sum(1 for u in union_list if u.level == AdminLevel.SUBAREA)
        return {
            "divisions": len(self.divisions),
            "districts": len(self.districts),
            "upazilas_total": len(upz_list),
            "upazilas_rural": len(upz_list) - metro_thanas,
            "thanas_metro": metro_thanas,
            "unions_total": len(union_list),
            "unions_rural": len(union_list) - metro_subareas,
            "subareas_metro": metro_subareas,
            "total_nodes": len(self._all_nodes),
            "indexed_unique_names": len(self._name_index),
        }

    # -------------------------------------------------------------------------
    # Loader Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _read_table_json(file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses JSON exported either as phpMyAdmin table export or raw list of dicts.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Gazetteer data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict) and entry.get("type") == "table" and "data" in entry:
                    return entry["data"]
            if raw and isinstance(raw[0], dict) and "type" not in raw[0]:
                return raw
        elif isinstance(raw, dict):
            if "data" in raw and isinstance(raw["data"], list):
                return raw["data"]

        return []

    def _add_to_index(self, key: str, node: AdminNode) -> None:
        """Internal helper to add a node to the name index without duplicates."""
        if not key:
            return
        if key not in self._name_index:
            self._name_index[key] = []
        if node not in self._name_index[key]:
            self._name_index[key].append(node)

    def _register_node(self, node: AdminNode) -> None:
        """Register node into global maps and bilingual exact name indexes."""
        self._all_nodes.append(node)
        self._all_nodes_by_id[node.id] = node
        if node.raw_id and node.raw_id not in self._all_nodes_by_id:
            self._all_nodes_by_id[node.raw_id] = node

        # Index English name
        en_key = _normalize_name(node.name)
        if en_key:
            self._add_to_index(en_key, node)
            if en_key in COMMON_ALIASES:
                for alias in COMMON_ALIASES[en_key]:
                    self._add_to_index(_normalize_name(alias), node)

        # Index Bangla name
        bn_key = _normalize_name(node.bn_name)
        if bn_key:
            self._add_to_index(bn_key, node)
            if bn_key in COMMON_ALIASES:
                for alias in COMMON_ALIASES[bn_key]:
                    self._add_to_index(_normalize_name(alias), node)

    # -------------------------------------------------------------------------
    # Main Data Loading Logic
    # -------------------------------------------------------------------------
    def load_data(self) -> None:
        """Loads all raw datasets and custom seeds into hierarchy."""
        self._divisions_by_id.clear()
        self._districts_by_id.clear()
        self._upazilas_by_id.clear()
        self._unions_by_id.clear()
        self._all_nodes.clear()
        self._all_nodes_by_id.clear()
        self._name_index.clear()

        self._load_divisions()
        self._load_districts()
        self._load_upazilas()
        self._load_unions()
        self._load_metro_thanas()

        logger.info("Gazetteer loaded successfully: %s", self.stats())

    def _load_divisions(self) -> None:
        rows = self._read_table_json(self.divisions_file)
        for r in rows:
            raw_id = str(r["id"]).strip()
            raw_name = str(r["name"]).strip()
            name = canonicalize_admin_name(raw_name)
            bn_name = str(r.get("bn_name", "")).strip()
            url = str(r.get("url", "")).strip() or None

            node_id = f"division_{raw_id}"
            node = AdminNode(
                id=node_id,
                raw_id=raw_id,
                name=name,
                bn_name=bn_name,
                level=AdminLevel.DIVISION,
                parent_id=None,
                parent=None,
                url=url,
            )
            self._divisions_by_id[node_id] = node
            self._divisions_by_id[raw_id] = node
            self._register_node(node)

    def _load_districts(self) -> None:
        rows = self._read_table_json(self.districts_file)
        for r in rows:
            raw_id = str(r["id"]).strip()
            raw_div_id = str(r.get("division_id", "")).strip()
            raw_name = str(r["name"]).strip()
            name = canonicalize_admin_name(raw_name)
            bn_name = str(r.get("bn_name", "")).strip()

            lat = float(r["lat"]) if r.get("lat") else None
            lon = float(r["lon"]) if r.get("lon") else None
            url = str(r.get("url", "")).strip() or None

            parent_div = self._divisions_by_id.get(raw_div_id)
            node_id = f"district_{raw_id}"
            parent_id = parent_div.id if parent_div else raw_div_id

            node = AdminNode(
                id=node_id,
                raw_id=raw_id,
                name=name,
                bn_name=bn_name,
                level=AdminLevel.DISTRICT,
                parent_id=parent_id or None,
                parent=parent_div,
                lat=lat,
                lon=lon,
                url=url,
            )
            if parent_div:
                parent_div.children.append(node)

            self._districts_by_id[node_id] = node
            self._districts_by_id[raw_id] = node
            self._register_node(node)

    def _load_upazilas(self) -> None:
        rows = self._read_table_json(self.upazilas_file)
        for r in rows:
            raw_id = str(r["id"]).strip()
            raw_dist_id = str(r.get("district_id", "")).strip()
            name = str(r["name"]).strip()
            bn_name = str(r.get("bn_name", "")).strip()
            url = str(r.get("url", "")).strip() or None

            parent_dist = self._districts_by_id.get(raw_dist_id)
            node_id = f"upazila_{raw_id}"
            parent_id = parent_dist.id if parent_dist else raw_dist_id

            node = AdminNode(
                id=node_id,
                raw_id=raw_id,
                name=name,
                bn_name=bn_name,
                level=AdminLevel.UPAZILA,
                parent_id=parent_id or None,
                parent=parent_dist,
                url=url,
            )
            if parent_dist:
                parent_dist.children.append(node)

            self._upazilas_by_id[node_id] = node
            self._upazilas_by_id[raw_id] = node
            self._register_node(node)

    def _load_unions(self) -> None:
        rows = self._read_table_json(self.unions_file)
        for r in rows:
            raw_id = str(r["id"]).strip()
            raw_upz_id = str(r.get("upazilla_id") or r.get("upazila_id") or "").strip()
            name = str(r["name"]).strip()
            bn_name = str(r.get("bn_name", "")).strip()
            url = str(r.get("url", "")).strip() or None

            parent_upz = self._upazilas_by_id.get(raw_upz_id)
            node_id = f"union_{raw_id}"
            parent_id = parent_upz.id if parent_upz else raw_upz_id

            node = AdminNode(
                id=node_id,
                raw_id=raw_id,
                name=name,
                bn_name=bn_name,
                level=AdminLevel.UNION,
                parent_id=parent_id or None,
                parent=parent_upz,
                url=url,
            )
            if parent_upz:
                parent_upz.children.append(node)

            self._unions_by_id[node_id] = node
            self._unions_by_id[raw_id] = node
            self._register_node(node)

    def _load_metro_thanas(self) -> None:
        """Loads custom metropolitan Thanas and Subareas (Dhaka, Ctg, Sylhet, etc.)."""
        if not self.metro_thanas_file.exists():
            logger.warning("Metro thanas seed file %s does not exist", self.metro_thanas_file)
            return

        with open(self.metro_thanas_file, "r", encoding="utf-8") as f:
            metro_entries = json.load(f)

        for entry in metro_entries:
            raw_dist_id = str(entry.get("district_id", "")).strip()
            dist_name = entry.get("district_name", "")
            parent_dist = self._districts_by_id.get(raw_dist_id)

            # Fallback lookup by district name if ID doesn't match directly
            if not parent_dist and dist_name:
                matches = self.find_by_exact_name(dist_name, level=AdminLevel.DISTRICT)
                if matches:
                    parent_dist = matches[0]
                    raw_dist_id = parent_dist.raw_id or parent_dist.id

            for thana_data in entry.get("thanas", []):
                t_name = str(thana_data["name"]).strip()
                t_bn = str(thana_data.get("bn_name", "")).strip()
                t_slug = _slugify(t_name)
                thana_id = f"metro_thana_{raw_dist_id}_{t_slug}"

                parent_id = parent_dist.id if parent_dist else raw_dist_id
                thana_node = AdminNode(
                    id=thana_id,
                    raw_id=thana_id,
                    name=t_name,
                    bn_name=t_bn,
                    level=AdminLevel.THANA,
                    parent_id=parent_id or None,
                    parent=parent_dist,
                    metadata={"source": "metro_seed", "district_name": dist_name},
                )
                if parent_dist:
                    parent_dist.children.append(thana_node)

                self._upazilas_by_id[thana_id] = thana_node
                self._register_node(thana_node)

                # Subareas / neighborhoods under metro thana
                for sub_data in thana_data.get("subareas", []):
                    s_name = str(sub_data["name"]).strip()
                    s_bn = str(sub_data.get("bn_name", "")).strip()
                    s_slug = _slugify(s_name)
                    sub_id = f"metro_subarea_{t_slug}_{s_slug}"

                    sub_node = AdminNode(
                        id=sub_id,
                        raw_id=sub_id,
                        name=s_name,
                        bn_name=s_bn,
                        level=AdminLevel.SUBAREA,
                        parent_id=thana_id,
                        parent=thana_node,
                        metadata={"source": "metro_seed", "thana_name": t_name},
                    )
                    thana_node.children.append(sub_node)
                    self._unions_by_id[sub_id] = sub_node
                    self._register_node(sub_node)

    # -------------------------------------------------------------------------
    # Search & Query Methods
    # -------------------------------------------------------------------------
    def find_by_exact_name(
        self, query: str, level: Optional[AdminLevel] = None
    ) -> List[AdminNode]:
        """
        Finds all administrative nodes matching the exact name (case-insensitive).
        Matches against both English name and Bangla bn_name.
        Captures homonyms across different districts/divisions.

        Args:
            query: The name in English or Bangla (e.g. "Mirpur", "মিরপুর", "Mohammadpur").
            level: Optional AdminLevel to filter results (e.g. AdminLevel.UPAZILA).

        Returns:
            List of matching AdminNode instances.
        """
        if not query or not query.strip():
            return []

        key = _normalize_name(query)
        matches = self._name_index.get(key, [])

        if level is not None:
            # Handle level grouping (e.g., if UPAZILA is requested, include THANA if appropriate)
            if level == AdminLevel.UPAZILA:
                matches = [m for m in matches if m.level.is_upazila_level]
            elif level == AdminLevel.UNION:
                matches = [m for m in matches if m.level.is_union_level]
            else:
                matches = [m for m in matches if m.level == level]

        return list(matches)

    def find_by_id(
        self, node_id: str, level: Optional[AdminLevel] = None
    ) -> Optional[AdminNode]:
        """
        Lookup any administrative node by ID or raw ID.
        If level is provided, restricts lookup to that administrative level.
        """
        sid = node_id.strip()
        if level is not None:
            if level == AdminLevel.DIVISION:
                return self._divisions_by_id.get(sid)
            elif level == AdminLevel.DISTRICT:
                return self._districts_by_id.get(sid)
            elif level.is_upazila_level:
                return self._upazilas_by_id.get(sid)
            elif level.is_union_level:
                return self._unions_by_id.get(sid)

        # Direct global ID lookup
        if sid in self._all_nodes_by_id:
            return self._all_nodes_by_id[sid]

        # Prefixed fallback lookups
        for prefix in ("division_", "district_", "upazila_", "union_"):
            cand = prefix + sid
            if cand in self._all_nodes_by_id:
                return self._all_nodes_by_id[cand]

        return None

    def get_division(self, query: str) -> Optional[AdminNode]:
        """Get division by exact name or ID."""
        if query in self._divisions_by_id:
            return self._divisions_by_id[query]
        matches = self.find_by_exact_name(query, level=AdminLevel.DIVISION)
        return matches[0] if matches else None

    def get_district(self, query: str) -> Optional[AdminNode]:
        """Get district by exact name or ID."""
        if query in self._districts_by_id:
            return self._districts_by_id[query]
        matches = self.find_by_exact_name(query, level=AdminLevel.DISTRICT)
        return matches[0] if matches else None
