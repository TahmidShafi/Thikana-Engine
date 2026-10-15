"""
Data models for Bangladesh administrative gazetteer hierarchy.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AdminLevel(str, Enum):
    DIVISION = "division"
    DISTRICT = "district"
    UPAZILA = "upazila"
    THANA = "thana"
    UNION = "union"
    SUBAREA = "subarea"

    @property
    def tier(self) -> int:
        """Numeric hierarchy tier (1: Division -> 4: Union/Subarea)."""
        if self == AdminLevel.DIVISION:
            return 1
        elif self == AdminLevel.DISTRICT:
            return 2
        elif self in (AdminLevel.UPAZILA, AdminLevel.THANA):
            return 3
        else:
            return 4

    @property
    def is_upazila_level(self) -> bool:
        return self in (AdminLevel.UPAZILA, AdminLevel.THANA)

    @property
    def is_union_level(self) -> bool:
        return self in (AdminLevel.UNION, AdminLevel.SUBAREA)


class AdminNode(BaseModel):
    """
    Represents an administrative node in the Bangladesh hierarchy.
    Hierarchy:
      Division -> District -> Upazila / Thana -> Union / Subarea
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    bn_name: str
    level: AdminLevel
    raw_id: Optional[str] = None
    parent_id: Optional[str] = None
    parent: Optional[AdminNode] = Field(default=None, repr=False, exclude=True)
    children: List[AdminNode] = Field(default_factory=list, repr=False, exclude=True)
    lat: Optional[float] = None
    lon: Optional[float] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        parent_name = self.parent.name if self.parent else None
        return (
            f"AdminNode(id={self.id!r}, name={self.name!r}, bn_name={self.bn_name!r}, "
            f"level={self.level.value!r}, parent={parent_name!r})"
        )

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def division(self) -> Optional[AdminNode]:
        """Traverse upwards to find the enclosing Division node."""
        curr: Optional[AdminNode] = self
        while curr is not None:
            if curr.level == AdminLevel.DIVISION:
                return curr
            curr = curr.parent
        return None

    @property
    def district(self) -> Optional[AdminNode]:
        """Traverse upwards to find the enclosing District node."""
        curr: Optional[AdminNode] = self
        while curr is not None:
            if curr.level == AdminLevel.DISTRICT:
                return curr
            curr = curr.parent
        return None

    @property
    def upazila(self) -> Optional[AdminNode]:
        """Traverse upwards to find the enclosing Upazila or Thana node."""
        curr: Optional[AdminNode] = self
        while curr is not None:
            if curr.level.is_upazila_level:
                return curr
            curr = curr.parent
        return None

    @property
    def thana(self) -> Optional[AdminNode]:
        """Alias for upazila / thana level node."""
        return self.upazila

    @property
    def hierarchy_path(self) -> List[AdminNode]:
        """Return nodes from root (Division) down to this node."""
        path = []
        curr: Optional[AdminNode] = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path

    @property
    def hierarchy_dict(self) -> Dict[str, str]:
        """Dictionary of English administrative names along the hierarchy."""
        result: Dict[str, str] = {}
        div = self.division
        dist = self.district
        upz = self.upazila
        if div:
            result["division"] = div.name
        if dist:
            result["district"] = dist.name
        if upz:
            key = "thana" if upz.level == AdminLevel.THANA else "upazila"
            result[key] = upz.name
        if self.level.is_union_level:
            key = "subarea" if self.level == AdminLevel.SUBAREA else "union"
            result[key] = self.name
        return result

    @property
    def hierarchy_bn_dict(self) -> Dict[str, str]:
        """Dictionary of Bangla administrative names along the hierarchy."""
        result: Dict[str, str] = {}
        div = self.division
        dist = self.district
        upz = self.upazila
        if div:
            result["division"] = div.bn_name
        if dist:
            result["district"] = dist.bn_name
        if upz:
            key = "thana" if upz.level == AdminLevel.THANA else "upazila"
            result[key] = upz.bn_name
        if self.level.is_union_level:
            key = "subarea" if self.level == AdminLevel.SUBAREA else "union"
            result[key] = self.bn_name
        return result

    @property
    def full_name_en(self) -> str:
        """Formatted string representation e.g. 'Mirpur 10, Mirpur, Dhaka, Dhaka'."""
        path = self.hierarchy_path
        path.reverse()
        return ", ".join(node.name for node in path)

    @property
    def full_name_bn(self) -> str:
        """Formatted Bengali string representation e.g. 'মিরপুর ১০, মিরপুর, ঢাকা, ঢাকা'."""
        path = self.hierarchy_path
        path.reverse()
        return ", ".join(node.bn_name for node in path)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a clean, serializable summary without circular references."""
        return {
            "id": self.id,
            "raw_id": self.raw_id,
            "name": self.name,
            "bn_name": self.bn_name,
            "level": self.level.value,
            "parent_id": self.parent_id,
            "hierarchy": self.hierarchy_dict,
            "hierarchy_bn": self.hierarchy_bn_dict,
            "full_name_en": self.full_name_en,
            "full_name_bn": self.full_name_bn,
            "lat": self.lat,
            "lon": self.lon,
            "url": self.url,
            "children_count": len(self.children),
        }
