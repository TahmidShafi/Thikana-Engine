"""
tests/test_gazetteer.py

Unit tests for BangladeshGazetteer and AdminNode hierarchy.
Verifies:
  - Dataset loading and count integrity
  - Parent-child hierarchy (Division -> District -> Upazila/Thana -> Union/Subarea)
  - Homonym resolution ("Mirpur" in Dhaka vs Kushtia, "Mohammadpur" in Dhaka, Magura, Noakhali)
  - Bilingual lookup (English and Bengali)
  - Metro Thanas and Subareas (Dhaka, Chattogram, Sylhet)
  - Case-insensitivity, whitespace tolerance, and level filtering
  - Node helper methods and serialization
"""

import pytest
from pathlib import Path
from src.models import AdminLevel, AdminNode
from src.gazetteer import BangladeshGazetteer, _normalize_name


@pytest.fixture(scope="module")
def gazetteer() -> BangladeshGazetteer:
    """Fixture providing a singleton BangladeshGazetteer instance for tests."""
    return BangladeshGazetteer()


# =============================================================================
# 1. Loading & Entity Count Integrity
# =============================================================================
def test_gazetteer_loading_and_counts(gazetteer: BangladeshGazetteer):
    stats = gazetteer.stats()
    assert stats["divisions"] == 8, "Bangladesh must have 8 divisions"
    assert stats["districts"] == 64, "Bangladesh must have 64 districts"
    assert stats["upazilas_rural"] == 494, "Raw dataset has 494 rural upazilas"
    assert stats["thanas_metro"] >= 29, "Metro seed has at least 29 urban thanas"
    assert stats["upazilas_total"] >= 523, "Total upazilas/thanas must be at least 523"
    assert stats["unions_rural"] == 4540, "Raw dataset has 4540 rural unions"
    assert stats["subareas_metro"] >= 150, "Metro seed has at least 150 urban subareas"
    assert stats["unions_total"] >= 4690, "Total unions/subareas must be at least 4690"
    assert stats["total_nodes"] >= 5285, "Total administrative nodes must be at least 5285"
    assert stats["indexed_unique_names"] > 8000


# =============================================================================
# 2. Parent-Child Hierarchy Structure
# =============================================================================
def test_division_roots(gazetteer: BangladeshGazetteer):
    for div in gazetteer.divisions:
        assert div.is_root is True
        assert div.parent is None
        assert div.level == AdminLevel.DIVISION
        assert len(div.children) > 0, f"Division {div.name} must have child districts"


def test_district_parent_child_hierarchy(gazetteer: BangladeshGazetteer):
    dhaka_dist = gazetteer.get_district("Dhaka")
    assert dhaka_dist is not None
    assert dhaka_dist.parent is not None
    assert dhaka_dist.parent.name == "Dhaka"
    assert dhaka_dist.parent.level == AdminLevel.DIVISION
    assert dhaka_dist.division == dhaka_dist.parent

    # Dhaka district should have both rural upazilas (e.g. Savar) and metro thanas (e.g. Mirpur)
    child_names = {c.name for c in dhaka_dist.children}
    assert "Savar" in child_names
    assert "Dhamrai" in child_names
    assert "Mirpur" in child_names
    assert "Mohammadpur" in child_names
    assert "Dhanmondi" in child_names
    assert "Uttara" in child_names


def test_hierarchy_chain_from_subarea(gazetteer: BangladeshGazetteer):
    # Lookup Mirpur 10 subarea
    matches = gazetteer.find_by_exact_name("Mirpur 10")
    assert len(matches) == 1
    m10 = matches[0]

    assert m10.level == AdminLevel.SUBAREA
    assert m10.is_leaf is True

    # Immediate parent must be Mirpur thana
    assert m10.parent is not None
    assert m10.parent.name == "Mirpur"
    assert m10.parent.level == AdminLevel.THANA

    # Grandparent must be Dhaka district
    assert m10.district is not None
    assert m10.district.name == "Dhaka"
    assert m10.district.level == AdminLevel.DISTRICT

    # Great-grandparent must be Dhaka division
    assert m10.division is not None
    assert m10.division.name == "Dhaka"
    assert m10.division.level == AdminLevel.DIVISION

    # Hierarchy path
    path = m10.hierarchy_path
    assert len(path) == 4
    assert [p.name for p in path] == ["Dhaka", "Dhaka", "Mirpur", "Mirpur 10"]


# =============================================================================
# 3. Homonym Resolution (Prompt Requirement: Mirpur & Mohammadpur)
# =============================================================================
def test_homonym_mirpur(gazetteer: BangladeshGazetteer):
    """
    Verifies capturing homonyms for 'Mirpur' in Dhaka vs. Kushtia.
    """
    results = gazetteer.find_by_exact_name("Mirpur")
    assert len(results) >= 2, "Must find at least Dhaka and Kushtia matches for Mirpur"

    # Filter to Level 3 (Upazila or Thana)
    upazila_matches = [m for m in results if m.level.is_upazila_level]
    assert len(upazila_matches) == 2, "Must have exactly 2 level-3 nodes named Mirpur"

    districts = {m.district.name: m for m in upazila_matches if m.district}
    assert "Dhaka" in districts, "Mirpur must exist under Dhaka district"
    assert "Kushtia" in districts, "Mirpur must exist under Kushtia district"

    # Check division parents
    dhaka_mirpur = districts["Dhaka"]
    assert dhaka_mirpur.division.name == "Dhaka"
    assert dhaka_mirpur.level == AdminLevel.THANA

    kushtia_mirpur = districts["Kushtia"]
    assert kushtia_mirpur.division.name == "Khulna"
    assert kushtia_mirpur.level == AdminLevel.UPAZILA


def test_homonym_mohammadpur(gazetteer: BangladeshGazetteer):
    """
    Verifies capturing homonyms for 'Mohammadpur' in Dhaka, Magura, and Noakhali.
    """
    results = gazetteer.find_by_exact_name("Mohammadpur")
    assert len(results) >= 3, "Must capture Mohammadpur in Dhaka, Magura, and Noakhali"

    districts = {m.district.name for m in results if m.district}
    assert "Dhaka" in districts, "Mohammadpur must exist in Dhaka (Metro Thana)"
    assert "Magura" in districts, "Mohammadpur must exist in Magura (Upazila)"
    assert "Noakhali" in districts, "Mohammadpur must exist in Noakhali (Union)"

    # Verify Dhaka match
    dhaka_moh = [m for m in results if m.district and m.district.name == "Dhaka"][0]
    assert dhaka_moh.level == AdminLevel.THANA
    assert dhaka_moh.division.name == "Dhaka"

    # Verify Magura match
    magura_moh = [m for m in results if m.district and m.district.name == "Magura" and m.level == AdminLevel.UPAZILA][0]
    assert magura_moh.level == AdminLevel.UPAZILA
    assert magura_moh.division.name == "Khulna"

    # Verify Noakhali matches
    noakhali_mohs = [m for m in results if m.district and m.district.name == "Noakhali"]
    assert len(noakhali_mohs) >= 1
    for nm in noakhali_mohs:
        assert nm.level == AdminLevel.UNION
        assert nm.division.name in ("Chattogram", "Chattagram")


# =============================================================================
# 4. Bilingual Search (Bangla bn_name support)
# =============================================================================
def test_bangla_search_mirpur(gazetteer: BangladeshGazetteer):
    results = gazetteer.find_by_exact_name("মিরপুর")
    assert len(results) >= 2
    districts = {m.district.name for m in results if m.district}
    assert "Dhaka" in districts
    assert "Kushtia" in districts


def test_bangla_search_mohammadpur(gazetteer: BangladeshGazetteer):
    results = gazetteer.find_by_exact_name("মোহাম্মদপুর")
    districts = {m.district.name for m in results if m.district}
    assert "Dhaka" in districts
    assert "Magura" in districts
    assert "Noakhali" in districts


def test_bangla_search_dhanmondi(gazetteer: BangladeshGazetteer):
    results = gazetteer.find_by_exact_name("ধানমন্ডি")
    assert len(results) == 1
    dhan = results[0]
    assert dhan.name == "Dhanmondi"
    assert dhan.district.name == "Dhaka"


def test_bangla_search_division_and_district(gazetteer: BangladeshGazetteer):
    results = gazetteer.find_by_exact_name("ঢাকা")
    assert len(results) >= 2
    levels = {r.level for r in results}
    assert AdminLevel.DIVISION in levels
    assert AdminLevel.DISTRICT in levels


# =============================================================================
# 5. Metro Thanas & Subareas Seeds (Dhaka, Chattogram, Sylhet)
# =============================================================================
def test_dhaka_metro_seeds(gazetteer: BangladeshGazetteer):
    expected_thanas = [
        "Mirpur",
        "Mohammadpur",
        "Dhanmondi",
        "Uttara",
        "Gulshan",
        "Banani",
        "Badda",
    ]
    for thana_name in expected_thanas:
        matches = gazetteer.find_by_exact_name(thana_name, level=AdminLevel.UPAZILA)
        dhaka_match = [m for m in matches if m.district and m.district.name == "Dhaka"]
        assert len(dhaka_match) == 1, f"Expected Dhaka metro thana {thana_name} not found"
        assert dhaka_match[0].level == AdminLevel.THANA

    # Test subareas
    for sub in ["Mirpur 10", "Mirpur 11"]:
        matches = gazetteer.find_by_exact_name(sub)
        assert len(matches) == 1, f"Expected subarea {sub} not found"
        assert matches[0].level == AdminLevel.SUBAREA
        assert matches[0].parent.name == "Mirpur"


def test_chattogram_metro_seeds(gazetteer: BangladeshGazetteer):
    expected_ctg_thanas = [
        "Halishahar",
        "Panchlaish",
        "Kotwali",
        "Chandgaon",
        "Agrabad",
    ]
    for thana_name in expected_ctg_thanas:
        matches = gazetteer.find_by_exact_name(thana_name, level=AdminLevel.UPAZILA)
        ctg_match = [m for m in matches if m.district and m.district.name == "Chattogram"]
        assert len(ctg_match) == 1, f"Expected Chattogram thana {thana_name} not found"
        assert ctg_match[0].level == AdminLevel.THANA
        assert ctg_match[0].division.name in ("Chattogram", "Chattagram")


def test_sylhet_metro_seeds(gazetteer: BangladeshGazetteer):
    expected_sylhet_thanas = [
        "Shahjalal Upashahar",
        "Amberkhana",
    ]
    for thana_name in expected_sylhet_thanas:
        matches = gazetteer.find_by_exact_name(thana_name, level=AdminLevel.UPAZILA)
        syl_match = [m for m in matches if m.district and m.district.name == "Sylhet"]
        assert len(syl_match) == 1, f"Expected Sylhet thana {thana_name} not found"
        assert syl_match[0].level == AdminLevel.THANA
        assert syl_match[0].division.name == "Sylhet"


# =============================================================================
# 6. Search Features, Case-Insensitivity & Whitespace
# =============================================================================
def test_case_insensitivity(gazetteer: BangladeshGazetteer):
    res_lower = gazetteer.find_by_exact_name("mirpur")
    res_upper = gazetteer.find_by_exact_name("MIRPUR")
    res_title = gazetteer.find_by_exact_name("Mirpur")

    assert len(res_lower) == len(res_upper) == len(res_title)
    ids_lower = {n.id for n in res_lower}
    ids_upper = {n.id for n in res_upper}
    assert ids_lower == ids_upper


def test_whitespace_tolerance(gazetteer: BangladeshGazetteer):
    res1 = gazetteer.find_by_exact_name("  Mirpur  ")
    res2 = gazetteer.find_by_exact_name("Mirpur")
    assert {n.id for n in res1} == {n.id for n in res2}

    res_spaced = gazetteer.find_by_exact_name("Mirpur    10")
    assert len(res_spaced) == 1
    assert res_spaced[0].name == "Mirpur 10"


def test_level_filtering(gazetteer: BangladeshGazetteer):
    # Without filter, returns thanas, upazilas, and unions named Mirpur
    all_mirpurs = gazetteer.find_by_exact_name("Mirpur")
    assert any(m.level.is_union_level for m in all_mirpurs)

    # Filtered by UPAZILA level (includes THANA)
    upz_mirpurs = gazetteer.find_by_exact_name("Mirpur", level=AdminLevel.UPAZILA)
    for m in upz_mirpurs:
        assert m.level.is_upazila_level


def test_empty_and_unknown_queries(gazetteer: BangladeshGazetteer):
    assert gazetteer.find_by_exact_name("") == []
    assert gazetteer.find_by_exact_name("   ") == []
    assert gazetteer.find_by_exact_name("NonExistentPlaceXYZ987") == []


# =============================================================================
# 7. Helper Properties, Serialization & Model Integrity
# =============================================================================
def test_node_full_names_and_summary_dict(gazetteer: BangladeshGazetteer):
    matches = gazetteer.find_by_exact_name("Mirpur 10")
    node = matches[0]

    assert node.full_name_en == "Mirpur 10, Mirpur, Dhaka, Dhaka"
    assert "মিরপুর ১০" in node.full_name_bn
    assert "ঢাকা" in node.full_name_bn

    summary = node.to_summary_dict()
    assert summary["name"] == "Mirpur 10"
    assert summary["level"] == "subarea"
    assert summary["hierarchy"]["division"] == "Dhaka"
    assert summary["hierarchy"]["district"] == "Dhaka"
    assert summary["hierarchy"]["thana"] == "Mirpur"
    assert summary["hierarchy"]["subarea"] == "Mirpur 10"


def test_pydantic_model_dump_without_recursion(gazetteer: BangladeshGazetteer):
    dhaka_dist = gazetteer.get_district("Dhaka")
    dump = dhaka_dist.model_dump()
    assert dump["name"] == "Dhaka"
    assert "parent" not in dump or dump["parent"] is None
    assert "children" not in dump or dump["children"] == []


def test_find_by_id(gazetteer: BangladeshGazetteer):
    # Lookup by prefixed ID
    node = gazetteer.find_by_id("division_6")
    assert node is not None
    assert node.name == "Dhaka"
    assert node.level == AdminLevel.DIVISION

    # Lookup by level + raw ID
    node_dist = gazetteer.find_by_id("47", level=AdminLevel.DISTRICT)
    assert node_dist is not None
    assert node_dist.name == "Dhaka"
    assert node_dist.level == AdminLevel.DISTRICT


# =============================================================================
# 8. Aliases and Custom Initialization
# =============================================================================
def test_historical_aliases(gazetteer: BangladeshGazetteer):
    # Chittagong -> Chattogram / Chattagram
    ctg_matches = gazetteer.find_by_exact_name("Chittagong")
    assert len(ctg_matches) >= 2
    assert any(m.level == AdminLevel.DISTRICT and m.name == "Chattogram" for m in ctg_matches)

    # Cumilla / Comilla
    comilla_matches = gazetteer.find_by_exact_name("Cumilla", level=AdminLevel.DISTRICT)
    assert len(comilla_matches) == 1
    assert comilla_matches[0].name in ("Cumilla", "Comilla")

    # Also verify alias search for Comilla
    comilla_alias_matches = gazetteer.find_by_exact_name("Comilla", level=AdminLevel.DISTRICT)
    assert len(comilla_alias_matches) == 1
    assert comilla_alias_matches[0].name in ("Cumilla", "Comilla")

    # Jessore -> Jashore (alias lookup)
    jashore_matches = gazetteer.find_by_exact_name("Jessore", level=AdminLevel.DISTRICT)
    assert len(jashore_matches) == 1
    assert jashore_matches[0].name == "Jashore"


def test_custom_data_dir_not_found():
    import pytest
    with pytest.raises(FileNotFoundError):
        BangladeshGazetteer(raw_data_dir="/non/existent/path/xyz")

