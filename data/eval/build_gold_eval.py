"""
data/eval/build_gold_eval.py

Constructs 100 realistic, PII-free, hand-crafted Bangladeshi delivery addresses
covering:
  - Commercial / institutional homonym distractors (e.g., 'Uttara Bank, Dhanmondi', 'Noakhali Tower, Paltan')
  - Ambiguous homonyms (Mirpur Dhaka vs. Kushtia, Mohammadpur Dhaka vs. Magura vs. Noakhali)
  - Dhaka, Chattogram, Sylhet, Rajshahi, Khulna metropolitan addresses
  - Rural addresses with C/O, Village/Gram, Post, and Union fields
  - Bangla Unicode addresses with precomposed nuktas and numerals
  - Banglish typos, informal landmarks, and relation markers

Saves the dataset to data/eval/gold_100.json with 1:1 token-level BIO tags.
"""

from __future__ import annotations
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.normalizer import normalize_address


# -----------------------------------------------------------------------------
# 9 BIO Entity Types:
# HOUSE, FLOOR_FLAT, ROAD, BLOCK_SECTOR, LANDMARK, LANDMARK_REL,
# SUBAREA, THANA_UPAZILA, DISTRICT (plus O)
# -----------------------------------------------------------------------------

RAW_GOLD_SPECS: List[Dict[str, Any]] = [
    # =========================================================================
    # Group 1: Commercial & Institutional Homonym Distractors (15 addresses)
    # The landmark contains a distant district/thana name as a distractor!
    # =========================================================================
    {
        "id": "gold_001",
        "raw": "Uttara Bank Limited, Road 27, Dhanmondi, Dhaka",
        "expected_thana": "Dhanmondi",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        # Uttara Bank (LANDMARK), RD# 27 (ROAD), Dhanmondi (THANA), Dhaka (DISTRICT)
        "spans": [
            ("Uttara Bank Limited", "LANDMARK"),
            ("Road 27", "ROAD"),
            ("Dhanmondi", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_002",
        "raw": "Noakhali Tower, 55 Purana Paltan, Dhaka",
        "expected_thana": "Paltan",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Noakhali Tower", "LANDMARK"),
            ("55", "HOUSE"),
            ("Purana Paltan", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_003",
        "raw": "Comilla Mistanna Bhandar, Section 10, Mirpur, Dhaka",
        "expected_thana": "Mirpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Comilla Mistanna Bhandar", "LANDMARK"),
            ("Section 10", "BLOCK_SECTOR"),
            ("Mirpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_004",
        "raw": "Rajshahi Sweets, Agrabad Commercial Area, Double Mooring, Chattogram",
        "expected_thana": "Double Mooring",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Rajshahi Sweets", "LANDMARK"),
            ("Agrabad Commercial Area", "SUBAREA"),
            ("Double Mooring", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_005",
        "raw": "Sylhet Tea House, GEC Circle, Panchlaish, Chattogram",
        "expected_thana": "Panchlaish",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Sylhet Tea House", "LANDMARK"),
            ("GEC Circle", "LANDMARK"),
            ("Panchlaish", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_006",
        "raw": "Barisal Plaza, KDA Avenue, Sonadanga, Khulna",
        "expected_thana": "Sonadanga",
        "expected_district": "Khulna",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Barisal Plaza", "LANDMARK"),
            ("KDA Avenue", "ROAD"),
            ("Sonadanga", "THANA_UPAZILA"),
            ("Khulna", "DISTRICT"),
        ],
    },
    {
        "id": "gold_007",
        "raw": "Bogra Curd Corner, Shaheb Bazar, Boalia, Rajshahi",
        "expected_thana": "Boalia",
        "expected_district": "Rajshahi",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Bogra Curd Corner", "LANDMARK"),
            ("Shaheb Bazar", "SUBAREA"),
            ("Boalia", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_008",
        "raw": "Khulna Cloth Store, Station Road, Kotwali, Rangpur",
        "expected_thana": "Kotwali",
        "expected_district": "Rangpur",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Khulna Cloth Store", "LANDMARK"),
            ("Station Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Rangpur", "DISTRICT"),
        ],
    },
    {
        "id": "gold_009",
        "raw": "Faridpur Enterprise, Jubilee Road, Kotwali, Chattogram",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Faridpur Enterprise", "LANDMARK"),
            ("Jubilee Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_010",
        "raw": "Kushtia Rice Agency, Khatunganj, Kotwali, Chattogram",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Kushtia Rice Agency", "LANDMARK"),
            ("Khatunganj", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_011",
        "raw": "Mymensingh Medical Hall, College Road, Bogura Sadar, Bogura",
        "expected_thana": "Bogura Sadar",
        "expected_district": "Bogura",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Mymensingh Medical Hall", "LANDMARK"),
            ("College Road", "ROAD"),
            ("Bogura Sadar", "THANA_UPAZILA"),
            ("Bogura", "DISTRICT"),
        ],
    },
    {
        "id": "gold_012",
        "raw": "Rangpur Engineering Works, Station Road, Tongi, Gazipur",
        "expected_thana": "Tongi",
        "expected_district": "Gazipur",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Rangpur Engineering Works", "LANDMARK"),
            ("Station Road", "ROAD"),
            ("Tongi", "THANA_UPAZILA"),
            ("Gazipur", "DISTRICT"),
        ],
    },
    {
        "id": "gold_013",
        "raw": "Tangail Saree Biponi, Elephant Road, New Market, Dhaka",
        "expected_thana": "New Market",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Tangail Saree Biponi", "LANDMARK"),
            ("Elephant Road", "ROAD"),
            ("New Market", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_014",
        "raw": "Dinajpur Store, Zero Point, Motijheel, Dhaka",
        "expected_thana": "Motijheel",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Dinajpur Store", "LANDMARK"),
            ("Zero Point", "LANDMARK"),
            ("Motijheel", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_015",
        "raw": "Jessore Transport Agency, Shahid Tajuddin Ahmed Sarani, Tejgaon, Dhaka",
        "expected_thana": "Tejgaon",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "commercial_distractor",
        "spans": [
            ("Jessore Transport Agency", "LANDMARK"),
            ("Shahid Tajuddin Ahmed Sarani", "ROAD"),
            ("Tejgaon", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 2: Geographic Homonyms & Postal Disambiguation (14 addresses)
    # Identical place names disambiguated by parent context or postal codes
    # =========================================================================
    {
        "id": "gold_016",
        "raw": "Holding 45, Tajmahal Road, Mohammadpur, Dhaka",
        "expected_thana": "Mohammadpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Holding 45", "HOUSE"),
            ("Tajmahal Road", "ROAD"),
            ("Mohammadpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_017",
        "raw": "Village: Mohammadpur, Upazila: Chatkhil, Noakhali",
        "expected_thana": "Chatkhil",
        "expected_district": "Noakhali",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Mohammadpur", "SUBAREA"),
            ("Chatkhil", "THANA_UPAZILA"),
            ("Noakhali", "DISTRICT"),
        ],
    },
    {
        "id": "gold_018",
        "raw": "Chowdhury Bari, Village: Mohammadpur, Upazila: Mohammadpur, Magura",
        "expected_thana": "Mohammadpur",
        "expected_district": "Magura",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Chowdhury Bari", "HOUSE"),
            ("Mohammadpur", "SUBAREA"),
            ("Mohammadpur", "THANA_UPAZILA"),
            ("Magura", "DISTRICT"),
        ],
    },
    {
        "id": "gold_019",
        "raw": "House 12, Road 4, Section 11, Mirpur, Dhaka",
        "expected_thana": "Mirpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("House 12", "HOUSE"),
            ("Road 4", "ROAD"),
            ("Section 11", "BLOCK_SECTOR"),
            ("Mirpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_020",
        "raw": "Mirpur College Road, Mirpur Bazar, Mirpur, Kushtia",
        "expected_thana": "Mirpur",
        "expected_district": "Kushtia",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Mirpur College Road", "ROAD"),
            ("Mirpur Bazar", "SUBAREA"),
            ("Mirpur", "THANA_UPAZILA"),
            ("Kushtia", "DISTRICT"),
        ],
    },
    {
        "id": "gold_021",
        "raw": "Halishahar Housing Estate, Block G, Halishahar, Chattogram",
        "expected_thana": "Halishahar",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Halishahar Housing Estate", "SUBAREA"),
            ("Block G", "BLOCK_SECTOR"),
            ("Halishahar", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_022",
        "raw": "Shahjalal Upashahar, Block D, Main Road, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Shahjalal Upashahar", "SUBAREA"),
            ("Block D", "BLOCK_SECTOR"),
            ("Main Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_023",
        "raw": "Chandgaon Residential Area, Block B, Road 3, Chandgaon, Chattogram",
        "expected_thana": "Chandgaon",
        "expected_district": "Chattogram",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Chandgaon Residential Area", "SUBAREA"),
            ("Block B", "BLOCK_SECTOR"),
            ("Road 3", "ROAD"),
            ("Chandgaon", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_024",
        "raw": "Durgapur Bazar, Upazila: Durgapur, Netrokona",
        "expected_thana": "Durgapur",
        "expected_district": "Netrokona",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Durgapur Bazar", "SUBAREA"),
            ("Durgapur", "THANA_UPAZILA"),
            ("Netrokona", "DISTRICT"),
        ],
    },
    {
        "id": "gold_025",
        "raw": "Durgapur Thana More, Durgapur, Rajshahi",
        "expected_thana": "Durgapur",
        "expected_district": "Rajshahi",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Durgapur Thana More", "LANDMARK"),
            ("Durgapur", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_026",
        "raw": "Daulatpur College Road, Daulatpur, Kushtia",
        "expected_thana": "Daulatpur",
        "expected_district": "Kushtia",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Daulatpur College Road", "ROAD"),
            ("Daulatpur", "THANA_UPAZILA"),
            ("Kushtia", "DISTRICT"),
        ],
    },
    {
        "id": "gold_027",
        "raw": "Daulatpur Bazar, Daulatpur, Manikganj",
        "expected_thana": "Daulatpur",
        "expected_district": "Manikganj",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Daulatpur Bazar", "SUBAREA"),
            ("Daulatpur", "THANA_UPAZILA"),
            ("Manikganj", "DISTRICT"),
        ],
    },
    {
        "id": "gold_028",
        "raw": "Raipur Bazar Road, Raipur, Lakshmipur",
        "expected_thana": "Raipur",
        "expected_district": "Lakshmipur",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Raipur Bazar Road", "ROAD"),
            ("Raipur", "THANA_UPAZILA"),
            ("Lakshmipur", "DISTRICT"),
        ],
    },
    {
        "id": "gold_029",
        "raw": "Holding 10, Road 4, Postcode 1216, Mirpur, Dhaka",
        "expected_thana": "Mirpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "homonym_disambiguation",
        "spans": [
            ("Holding 10", "HOUSE"),
            ("Road 4", "ROAD"),
            ("1216", "O"),
            ("Mirpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 3: Dhaka Metropolitan Urban Delivery Addresses (18 addresses)
    # =========================================================================
    {
        "id": "gold_030",
        "raw": "House 14, Road 8, Dhanmondi, Dhaka",
        "expected_thana": "Dhanmondi",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("House 14", "HOUSE"),
            ("Road 8", "ROAD"),
            ("Dhanmondi", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_031",
        "raw": "Flat 5A, Plot 22, Road 11, Block D, Banani, Dhaka",
        "expected_thana": "Banani",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Flat 5A", "FLOOR_FLAT"),
            ("Plot 22", "HOUSE"),
            ("Road 11", "ROAD"),
            ("Block D", "BLOCK_SECTOR"),
            ("Banani", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_032",
        "raw": "Level 4, House 35, Road 27, Gulshan 1, Dhaka",
        "expected_thana": "Gulshan",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Level 4", "FLOOR_FLAT"),
            ("House 35", "HOUSE"),
            ("Road 27", "ROAD"),
            ("Gulshan 1", "SUBAREA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_033",
        "raw": "Holding 88, Road 2, Sector 4, Uttara, Dhaka",
        "expected_thana": "Uttara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Holding 88", "HOUSE"),
            ("Road 2", "ROAD"),
            ("Sector 4", "BLOCK_SECTOR"),
            ("Uttara", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_034",
        "raw": "House 12/B, Lane 5, Block C, Bashundhara R/A, Vatara, Dhaka",
        "expected_thana": "Vatara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("House 12/B", "HOUSE"),
            ("Lane 5", "ROAD"),
            ("Block C", "BLOCK_SECTOR"),
            ("Bashundhara R/A", "SUBAREA"),
            ("Vatara", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_035",
        "raw": "Shop 14, Level 2, Shimanto Square, Dhanmondi, Dhaka",
        "expected_thana": "Dhanmondi",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Shop 14", "HOUSE"),
            ("Level 2", "FLOOR_FLAT"),
            ("Shimanto Square", "LANDMARK"),
            ("Dhanmondi", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_036",
        "raw": "Bari 7, Rasta 3, Block A, Lalmatia, Mohammadpur, Dhaka",
        "expected_thana": "Mohammadpur",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Bari 7", "HOUSE"),
            ("Rasta 3", "ROAD"),
            ("Block A", "BLOCK_SECTOR"),
            ("Lalmatia", "SUBAREA"),
            ("Mohammadpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_037",
        "raw": "Holding 50, Kazi Nazrul Islam Avenue, Farmgate, Tejgaon, Dhaka",
        "expected_thana": "Tejgaon",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Holding 50", "HOUSE"),
            ("Kazi Nazrul Islam Avenue", "ROAD"),
            ("Farmgate", "SUBAREA"),
            ("Tejgaon", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_038",
        "raw": "Plot 10, Road 2, Block J, Baridhara, Vatara, Dhaka",
        "expected_thana": "Vatara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Plot 10", "HOUSE"),
            ("Road 2", "ROAD"),
            ("Block J", "BLOCK_SECTOR"),
            ("Baridhara", "SUBAREA"),
            ("Vatara", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_039",
        "raw": "House 3, Road 17, Sector 7, Uttara Model Town, Uttara, Dhaka",
        "expected_thana": "Uttara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("House 3", "HOUSE"),
            ("Road 17", "ROAD"),
            ("Sector 7", "BLOCK_SECTOR"),
            ("Uttara Model Town", "SUBAREA"),
            ("Uttara", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_040",
        "raw": "Flat B2, 4th Floor, Shantinagar Plaza, Shantinagar, Paltan, Dhaka",
        "expected_thana": "Paltan",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Flat B2", "FLOOR_FLAT"),
            ("4th Floor", "FLOOR_FLAT"),
            ("Shantinagar Plaza", "LANDMARK"),
            ("Shantinagar", "SUBAREA"),
            ("Paltan", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_041",
        "raw": "Holding 105, Malibagh Chowdhury Para, Ramna, Dhaka",
        "expected_thana": "Ramna",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Holding 105", "HOUSE"),
            ("Malibagh Chowdhury Para", "SUBAREA"),
            ("Ramna", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_042",
        "raw": "House 24, Road 2, Block F, Aftabnagar, Badda, Dhaka",
        "expected_thana": "Badda",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("House 24", "HOUSE"),
            ("Road 2", "ROAD"),
            ("Block F", "BLOCK_SECTOR"),
            ("Aftabnagar", "SUBAREA"),
            ("Badda", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_043",
        "raw": "Holding 15, Green Road, Kalabagan, Dhaka",
        "expected_thana": "Kalabagan",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Holding 15", "HOUSE"),
            ("Green Road", "ROAD"),
            ("Kalabagan", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_044",
        "raw": "Plot 6, Avenue 1, Section 6, Block C, Mirpur, Dhaka",
        "expected_thana": "Mirpur",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Plot 6", "HOUSE"),
            ("Avenue 1", "ROAD"),
            ("Section 6", "BLOCK_SECTOR"),
            ("Block C", "BLOCK_SECTOR"),
            ("Mirpur", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_045",
        "raw": "House 42, Road 13/A, Sector 11, Uttara, Dhaka",
        "expected_thana": "Uttara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("House 42", "HOUSE"),
            ("Road 13/A", "ROAD"),
            ("Sector 11", "BLOCK_SECTOR"),
            ("Uttara", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_046",
        "raw": "Flat 3C, House 19, Road 4, DOHS Banani, Dhaka",
        "expected_thana": "Banani",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Flat 3C", "FLOOR_FLAT"),
            ("House 19", "HOUSE"),
            ("Road 4", "ROAD"),
            ("DOHS Banani", "SUBAREA"),
            ("Dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_047",
        "raw": "Holding 21, Indira Road, Tejgaon, Dhaka",
        "expected_thana": "Tejgaon",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "dhaka_metro",
        "spans": [
            ("Holding 21", "HOUSE"),
            ("Indira Road", "ROAD"),
            ("Tejgaon", "THANA_UPAZILA"),
            ("Dhaka", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 4: Chattogram Metro Delivery Addresses (10 addresses)
    # =========================================================================
    {
        "id": "gold_048",
        "raw": "Flat 4A, Sanmar Ocean City, GEC More, Nasirabad, Panchlaish, Chattogram",
        "expected_thana": "Panchlaish",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Flat 4A", "FLOOR_FLAT"),
            ("Sanmar Ocean City", "LANDMARK"),
            ("GEC More", "LANDMARK"),
            ("Nasirabad", "SUBAREA"),
            ("Panchlaish", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_049",
        "raw": "Holding 12, Road 1, Block H, Halishahar Housing Estate, Halishahar, Chattogram",
        "expected_thana": "Halishahar",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 12", "HOUSE"),
            ("Road 1", "ROAD"),
            ("Block H", "BLOCK_SECTOR"),
            ("Halishahar Housing Estate", "SUBAREA"),
            ("Halishahar", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_050",
        "raw": "Plot 8, Road 4, Chandgaon Residential Area, Chandgaon, Chattogram",
        "expected_thana": "Chandgaon",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Plot 8", "HOUSE"),
            ("Road 4", "ROAD"),
            ("Chandgaon Residential Area", "SUBAREA"),
            ("Chandgaon", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_051",
        "raw": "Shop 25, Akhtaruzzaman Center, Agrabad Commercial Area, Double Mooring, Chattogram",
        "expected_thana": "Double Mooring",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Shop 25", "HOUSE"),
            ("Akhtaruzzaman Center", "LANDMARK"),
            ("Agrabad Commercial Area", "SUBAREA"),
            ("Double Mooring", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_052",
        "raw": "Holding 102, Strand Road, Sadarghat, Kotwali, Chattogram",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 102", "HOUSE"),
            ("Strand Road", "ROAD"),
            ("Sadarghat", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_053",
        "raw": "House 15, Lane 2, Mehedibag, Kotwali, Chattogram",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("House 15", "HOUSE"),
            ("Lane 2", "ROAD"),
            ("Mehedibag", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_054",
        "raw": "Holding 45, CDA Avenue, Muradpur, Panchlaish, Chattogram",
        "expected_thana": "Panchlaish",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 45", "HOUSE"),
            ("CDA Avenue", "ROAD"),
            ("Muradpur", "SUBAREA"),
            ("Panchlaish", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_055",
        "raw": "Flat 6B, Equity Heights, Katalganj, Panchlaish, Chattogram",
        "expected_thana": "Panchlaish",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Flat 6B", "FLOOR_FLAT"),
            ("Equity Heights", "LANDMARK"),
            ("Katalganj", "SUBAREA"),
            ("Panchlaish", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_056",
        "raw": "Holding 80, Station Road, Kotwali, Chattogram",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 80", "HOUSE"),
            ("Station Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },
    {
        "id": "gold_057",
        "raw": "Plot 14, Sector 2, CEPZ, Patenga, Chattogram",
        "expected_thana": "Patenga",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Plot 14", "HOUSE"),
            ("Sector 2", "BLOCK_SECTOR"),
            ("CEPZ", "SUBAREA"),
            ("Patenga", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 5: Sylhet Metro Delivery Addresses (6 addresses)
    # =========================================================================
    {
        "id": "gold_058",
        "raw": "House 12, Road 3, Block E, Shahjalal Upashahar, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("House 12", "HOUSE"),
            ("Road 3", "ROAD"),
            ("Block E", "BLOCK_SECTOR"),
            ("Shahjalal Upashahar", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_059",
        "raw": "Holding 45, Amberkhana Point, Airport Road, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 45", "HOUSE"),
            ("Amberkhana Point", "LANDMARK"),
            ("Airport Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_060",
        "raw": "Flat 2A, Al Hamra Shopping City, Zindabazar, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Flat 2A", "FLOOR_FLAT"),
            ("Al Hamra Shopping City", "LANDMARK"),
            ("Zindabazar", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_061",
        "raw": "Holding 78, Subidbazar, Sylhet Sadar, Sylhet",
        "expected_thana": "Sylhet Sadar",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 78", "HOUSE"),
            ("Subidbazar", "SUBAREA"),
            ("Sylhet Sadar", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_062",
        "raw": "House 5, Lane 1, Shibganj, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("House 5", "HOUSE"),
            ("Lane 1", "ROAD"),
            ("Shibganj", "SUBAREA"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },
    {
        "id": "gold_063",
        "raw": "Holding 22, Kumarpara Road, Kotwali, Sylhet",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 22", "HOUSE"),
            ("Kumarpara Road", "ROAD"),
            ("Kotwali", "THANA_UPAZILA"),
            ("Sylhet", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 6: Rajshahi & Khulna Divisional Cities (8 addresses)
    # =========================================================================
    {
        "id": "gold_064",
        "raw": "Holding 150, Greater Road, Kazihata, Rajputpara, Rajshahi",
        "expected_thana": "Rajputpara",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 150", "HOUSE"),
            ("Greater Road", "ROAD"),
            ("Kazihata", "SUBAREA"),
            ("Rajputpara", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_065",
        "raw": "House 25, Road 2, Padma Residential Area, Chandrima, Rajshahi",
        "expected_thana": "Chandrima",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("House 25", "HOUSE"),
            ("Road 2", "ROAD"),
            ("Padma Residential Area", "SUBAREA"),
            ("Chandrima", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_066",
        "raw": "Shop 4, New Market, Shaheb Bazar, Boalia, Rajshahi",
        "expected_thana": "Boalia",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Shop 4", "HOUSE"),
            ("New Market", "LANDMARK"),
            ("Shaheb Bazar", "SUBAREA"),
            ("Boalia", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_067",
        "raw": "Holding 85, Station Road, Ghoramara, Boalia, Rajshahi",
        "expected_thana": "Boalia",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 85", "HOUSE"),
            ("Station Road", "ROAD"),
            ("Ghoramara", "SUBAREA"),
            ("Boalia", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_068",
        "raw": "House 12, Mujgunni Residential Area, Khalishpur, Khulna",
        "expected_thana": "Khalishpur",
        "expected_district": "Khulna",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("House 12", "HOUSE"),
            ("Mujgunni Residential Area", "SUBAREA"),
            ("Khalishpur", "THANA_UPAZILA"),
            ("Khulna", "DISTRICT"),
        ],
    },
    {
        "id": "gold_069",
        "raw": "Holding 45, KDA Avenue, Sonadanga, Khulna",
        "expected_thana": "Sonadanga",
        "expected_district": "Khulna",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 45", "HOUSE"),
            ("KDA Avenue", "ROAD"),
            ("Sonadanga", "THANA_UPAZILA"),
            ("Khulna", "DISTRICT"),
        ],
    },
    {
        "id": "gold_070",
        "raw": "Plot 10, Road 3, Boyra Main Road, Khalishpur, Khulna",
        "expected_thana": "Khalishpur",
        "expected_district": "Khulna",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Plot 10", "HOUSE"),
            ("Road 3", "ROAD"),
            ("Boyra Main Road", "ROAD"),
            ("Khalishpur", "THANA_UPAZILA"),
            ("Khulna", "DISTRICT"),
        ],
    },
    {
        "id": "gold_071",
        "raw": "Holding 220, Sir Iqbal Road, Khulna Sadar, Khulna",
        "expected_thana": "Khulna Sadar",
        "expected_district": "Khulna",
        "is_homonym_test": False,
        "category": "metro_regional",
        "spans": [
            ("Holding 220", "HOUSE"),
            ("Sir Iqbal Road", "ROAD"),
            ("Khulna Sadar", "THANA_UPAZILA"),
            ("Khulna", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 7: Rural Addresses with C/O, Village, Union, Upazila (15 addresses)
    # =========================================================================
    {
        "id": "gold_072",
        "raw": "C/O Abdul Karim, Village: Char Bhabanipur, Union: Char Ishwar, Hatiya, Noakhali",
        "expected_thana": "Hatiya",
        "expected_district": "Noakhali",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Abdul Karim", "O"),
            ("Char Bhabanipur", "SUBAREA"),
            ("Char Ishwar", "SUBAREA"),
            ("Hatiya", "THANA_UPAZILA"),
            ("Noakhali", "DISTRICT"),
        ],
    },
    {
        "id": "gold_073",
        "raw": "Care of Master Rafiq, Village: Uttar Rampur, Post: Harinarayanpur, Kushtia Sadar, Kushtia",
        "expected_thana": "Kushtia Sadar",
        "expected_district": "Kushtia",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Master Rafiq", "O"),
            ("Uttar Rampur", "SUBAREA"),
            ("Harinarayanpur", "SUBAREA"),
            ("Kushtia Sadar", "THANA_UPAZILA"),
            ("Kushtia", "DISTRICT"),
        ],
    },
    {
        "id": "gold_074",
        "raw": "C/O Dr. M. Rahman, Gram: Shampur, Dakghar: Durgapur, Durgapur, Netrokona",
        "expected_thana": "Durgapur",
        "expected_district": "Netrokona",
        "is_homonym_test": True,
        "category": "rural_union",
        "spans": [
            ("Dr. M. Rahman", "O"),
            ("Shampur", "SUBAREA"),
            ("Durgapur", "SUBAREA"),
            ("Durgapur", "THANA_UPAZILA"),
            ("Netrokona", "DISTRICT"),
        ],
    },
    {
        "id": "gold_075",
        "raw": "Village: Daskhin Para, Union: Fatepur, Upazila: Madhabpur, Habiganj",
        "expected_thana": "Madhabpur",
        "expected_district": "Habiganj",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Daskhin Para", "SUBAREA"),
            ("Fatepur", "SUBAREA"),
            ("Madhabpur", "THANA_UPAZILA"),
            ("Habiganj", "DISTRICT"),
        ],
    },
    {
        "id": "gold_076",
        "raw": "C/O Haji Shamsul Haque, Gram: Boro Bari, Union: Joyag, Sonaimuri, Noakhali",
        "expected_thana": "Sonaimuri",
        "expected_district": "Noakhali",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Haji Shamsul Haque", "O"),
            ("Boro Bari", "HOUSE"),
            ("Joyag", "SUBAREA"),
            ("Sonaimuri", "THANA_UPAZILA"),
            ("Noakhali", "DISTRICT"),
        ],
    },
    {
        "id": "gold_077",
        "raw": "Village: Paikpara, Post: Mirzapur, Upazila: Mirzapur, Tangail",
        "expected_thana": "Mirzapur",
        "expected_district": "Tangail",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Paikpara", "SUBAREA"),
            ("Mirzapur", "SUBAREA"),
            ("Mirzapur", "THANA_UPAZILA"),
            ("Tangail", "DISTRICT"),
        ],
    },
    {
        "id": "gold_078",
        "raw": "Gram: Gobindapur, Union: Gopalpur, Upazila: Mithapukur, Rangpur",
        "expected_thana": "Mithapukur",
        "expected_district": "Rangpur",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Gobindapur", "SUBAREA"),
            ("Gopalpur", "SUBAREA"),
            ("Mithapukur", "THANA_UPAZILA"),
            ("Rangpur", "DISTRICT"),
        ],
    },
    {
        "id": "gold_079",
        "raw": "C/O Principal Jalal Uddin, Village: Madhabpur, Union: Pathannagar, Chhagalnaiya, Feni",
        "expected_thana": "Chhagalnaiya",
        "expected_district": "Feni",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Principal Jalal Uddin", "O"),
            ("Madhabpur", "SUBAREA"),
            ("Pathannagar", "SUBAREA"),
            ("Chhagalnaiya", "THANA_UPAZILA"),
            ("Feni", "DISTRICT"),
        ],
    },
    {
        "id": "gold_080",
        "raw": "Village: Ramchandrapur, Union: Bausha, Upazila: Bagha, Rajshahi",
        "expected_thana": "Bagha",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Ramchandrapur", "SUBAREA"),
            ("Bausha", "SUBAREA"),
            ("Bagha", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_081",
        "raw": "Care of Abdus Sattar, Gram: Kanthaltoli, Union: Patharghata, Patharghata, Barguna",
        "expected_thana": "Patharghata",
        "expected_district": "Barguna",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Abdus Sattar", "O"),
            ("Kanthaltoli", "SUBAREA"),
            ("Patharghata", "SUBAREA"),
            ("Patharghata", "THANA_UPAZILA"),
            ("Barguna", "DISTRICT"),
        ],
    },
    {
        "id": "gold_082",
        "raw": "Village: Shialkhol, Union: Shialkhol, Sirajganj Sadar, Sirajganj",
        "expected_thana": "Sirajganj Sadar",
        "expected_district": "Sirajganj",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Shialkhol", "SUBAREA"),
            ("Shialkhol", "SUBAREA"),
            ("Sirajganj Sadar", "THANA_UPAZILA"),
            ("Sirajganj", "DISTRICT"),
        ],
    },
    {
        "id": "gold_083",
        "raw": "C/O Nurul Islam, Village: Jagannathpur, Union: Alinagar, Bhola Sadar, Bhola",
        "expected_thana": "Bhola Sadar",
        "expected_district": "Bhola",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Nurul Islam", "O"),
            ("Jagannathpur", "SUBAREA"),
            ("Alinagar", "SUBAREA"),
            ("Bhola Sadar", "THANA_UPAZILA"),
            ("Bhola", "DISTRICT"),
        ],
    },
    {
        "id": "gold_084",
        "raw": "Village: Krishnapur, Dakghar: Mohanpur, Upazila: Mohanpur, Rajshahi",
        "expected_thana": "Mohanpur",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Krishnapur", "SUBAREA"),
            ("Mohanpur", "SUBAREA"),
            ("Mohanpur", "THANA_UPAZILA"),
            ("Rajshahi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_085",
        "raw": "Gram: Shibpur, Union: Belabo, Upazila: Belabo, Narsingdi",
        "expected_thana": "Belabo",
        "expected_district": "Narsingdi",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Shibpur", "SUBAREA"),
            ("Belabo", "SUBAREA"),
            ("Belabo", "THANA_UPAZILA"),
            ("Narsingdi", "DISTRICT"),
        ],
    },
    {
        "id": "gold_086",
        "raw": "C/O Master Abul Kashem, Village: Choto Kumira, Sitakunda, Chattogram",
        "expected_thana": "Sitakunda",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "rural_union",
        "spans": [
            ("Master Abul Kashem", "O"),
            ("Choto Kumira", "SUBAREA"),
            ("Sitakunda", "THANA_UPAZILA"),
            ("Chattogram", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 8: Bangla Script Delivery Addresses (7 addresses)
    # =========================================================================
    {
        "id": "gold_087",
        "raw": "বাসা ১২, রোড ৪, ব্লক সি, বনানী, ঢাকা",
        "expected_thana": "Banani",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("বাসা ১২", "HOUSE"),
            ("রোড ৪", "ROAD"),
            ("ব্লক সি", "BLOCK_SECTOR"),
            ("বনানী", "THANA_UPAZILA"),
            ("ঢাকা", "DISTRICT"),
        ],
    },
    {
        "id": "gold_088",
        "raw": "ফ্ল্যাট ৩বি, হোল্ডিং ৪৫, তাজমহল রোড, মোহাম্মদপুর, ঢাকা",
        "expected_thana": "Mohammadpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "bangla_script",
        "spans": [
            ("ফ্ল্যাট ৩বি", "FLOOR_FLAT"),
            ("হোল্ডিং ৪৫", "HOUSE"),
            ("তাজমহল রোড", "ROAD"),
            ("মোহাম্মদপুর", "THANA_UPAZILA"),
            ("ঢাকা", "DISTRICT"),
        ],
    },
    {
        "id": "gold_089",
        "raw": "দোকান নং ৮, নিউ মার্কেট, কোতোয়ালী, চট্টগ্রাম",
        "expected_thana": "Kotwali",
        "expected_district": "Chattogram",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("দোকান নং ৮", "HOUSE"),
            ("নিউ মার্কেট", "LANDMARK"),
            ("কোতোয়ালী", "THANA_UPAZILA"),
            ("চট্টগ্রাম", "DISTRICT"),
        ],
    },
    {
        "id": "gold_090",
        "raw": "বাড়ি নং ৭, সেক্টর ৩, উত্তরা, ঢাকা",
        "expected_thana": "Uttara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("বাড়ি নং ৭", "HOUSE"),
            ("সেক্টর ৩", "BLOCK_SECTOR"),
            ("উত্তরা", "THANA_UPAZILA"),
            ("ঢাকা", "DISTRICT"),
        ],
    },
    {
        "id": "gold_091",
        "raw": "হোল্ডিং ১৫, আম্বরখানা, কোতোয়ালী, সিলেট",
        "expected_thana": "Kotwali",
        "expected_district": "Sylhet",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("হোল্ডিং ১৫", "HOUSE"),
            ("আম্বরখানা", "SUBAREA"),
            ("কোতোয়ালী", "THANA_UPAZILA"),
            ("সিলেট", "DISTRICT"),
        ],
    },
    {
        "id": "gold_092",
        "raw": "গ্রাম: চন্ডিপুর, ডাকঘর: বাঘা, উপজেলা: বাঘা, জেলা: রাজশাহী",
        "expected_thana": "Bagha",
        "expected_district": "Rajshahi",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("চন্ডিপুর", "SUBAREA"),
            ("বাঘা", "SUBAREA"),
            ("বাঘা", "THANA_UPAZILA"),
            ("রাজশাহী", "DISTRICT"),
        ],
    },
    {
        "id": "gold_093",
        "raw": "বাসা নং ২২, কেডিএ এভিনিউ, সোনাডাঙ্গা, খুলনা",
        "expected_thana": "Sonadanga",
        "expected_district": "Khulna",
        "is_homonym_test": False,
        "category": "bangla_script",
        "spans": [
            ("বাসা নং ২২", "HOUSE"),
            ("কেডিএ এভিনিউ", "ROAD"),
            ("সোনাডাঙ্গা", "THANA_UPAZILA"),
            ("খুলনা", "DISTRICT"),
        ],
    },

    # =========================================================================
    # Group 9: Banglish Typos, Landmarks & Relational Anchors (7 addresses)
    # =========================================================================
    {
        "id": "gold_094",
        "raw": "basha 12, rd 4, mirpur dosh, dhaka",
        "expected_thana": "Mirpur",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("basha 12", "HOUSE"),
            ("rd 4", "ROAD"),
            ("mirpur dosh", "SUBAREA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_095",
        "raw": "H# 45/A, rasta char, blok B, uttora, dhaka",
        "expected_thana": "Uttara",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("H# 45/A", "HOUSE"),
            ("rasta char", "ROAD"),
            ("blok B", "BLOCK_SECTOR"),
            ("uttora", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_096",
        "raw": "near shia mosque, tajmahal road, mohammadpur, dhaka",
        "expected_thana": "Mohammadpur",
        "expected_district": "Dhaka",
        "is_homonym_test": True,
        "category": "banglish_typo_landmark",
        "spans": [
            ("near", "LANDMARK_REL"),
            ("shia mosque", "LANDMARK"),
            ("tajmahal road", "ROAD"),
            ("mohammadpur", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_097",
        "raw": "opposite square hospital, panthapath, tejgaon, dhaka",
        "expected_thana": "Tejgaon",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("opposite", "LANDMARK_REL"),
            ("square hospital", "LANDMARK"),
            ("panthapath", "ROAD"),
            ("tejgaon", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_098",
        "raw": "behind bashundhara city, panthapath, kalabagan, dhaka",
        "expected_thana": "Kalabagan",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("behind", "LANDMARK_REL"),
            ("bashundhara city", "LANDMARK"),
            ("panthapath", "ROAD"),
            ("kalabagan", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_099",
        "raw": "beside brack bank, rd 11, banani, dhaka",
        "expected_thana": "Banani",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("beside", "LANDMARK_REL"),
            ("brack bank", "LANDMARK"),
            ("rd 11", "ROAD"),
            ("banani", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
    {
        "id": "gold_100",
        "raw": "opp labaid hospital, road 4, dhanmondi, dhaka",
        "expected_thana": "Dhanmondi",
        "expected_district": "Dhaka",
        "is_homonym_test": False,
        "category": "banglish_typo_landmark",
        "spans": [
            ("opp", "LANDMARK_REL"),
            ("labaid hospital", "LANDMARK"),
            ("road 4", "ROAD"),
            ("dhanmondi", "THANA_UPAZILA"),
            ("dhaka", "DISTRICT"),
        ],
    },
]


def align_tokens_to_spans(
    normalized_tokens: List[str], spans: List[Tuple[str, str]]
) -> List[str]:
    """
    Aligns normalized tokens with annotated entity spans into BIO tags.
    """
    tags = ["O"] * len(normalized_tokens)

    # Normalize span text for fuzzy matching against token sequences
    norm_span_info = []
    for span_text, label in spans:
        span_norm = normalize_address(span_text)
        span_tokens = span_norm["tokens"]
        if not span_tokens:
            continue
        norm_span_info.append(([t.lower() for t in span_tokens], label))

    tokens_lower = [t.lower() for t in normalized_tokens]
    matched_indices = set()

    # Match longer spans first
    norm_span_info.sort(key=lambda s: len(s[0]), reverse=True)

    for span_tokens_l, label in norm_span_info:
        L = len(span_tokens_l)
        found = False
        for i in range(len(tokens_lower) - L + 1):
            if any(idx in matched_indices for idx in range(i, i + L)):
                continue
            # Subsequence match
            if tokens_lower[i : i + L] == span_tokens_l:
                if label == "O":
                    for k in range(i, i + L):
                        tags[k] = "O"
                        matched_indices.add(k)
                else:
                    tags[i] = f"B-{label}"
                    for k in range(i + 1, i + L):
                        tags[k] = f"I-{label}"
                        matched_indices.add(k)
                    matched_indices.add(i)
                found = True
                break

        if not found and L == 1:
            # Single token fallback: check substring inclusion
            target = span_tokens_l[0]
            for i, tok_l in enumerate(tokens_lower):
                if i not in matched_indices and (target == tok_l or target in tok_l or tok_l in target):
                    tags[i] = "O" if label == "O" else f"B-{label}"
                    matched_indices.add(i)
                    break

    return tags


def build_gold_dataset() -> List[Dict[str, Any]]:
    """
    Builds the 100 gold evaluation records.
    """
    records = []
    for spec in RAW_GOLD_SPECS:
        raw_text = spec["raw"]
        norm = normalize_address(raw_text)
        tokens = norm["tokens"]
        tags = align_tokens_to_spans(tokens, spec["spans"])

        record = {
            "id": spec["id"],
            "raw_text": raw_text,
            "normalized_text": norm["normalized_text"],
            "tokens": tokens,
            "tags": tags,
            "expected_thana": spec["expected_thana"],
            "expected_district": spec["expected_district"],
            "is_homonym_test": spec["is_homonym_test"],
            "category": spec["category"],
        }
        records.append(record)

    return records


def main():
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    gold_records = build_gold_dataset()
    output_path = os.path.join(os.path.dirname(__file__), "gold_100.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_records, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(gold_records)} gold evaluation records in:")
    print(f" -> {output_path}")

    # Print summary statistics
    tag_counts: Dict[str, int] = {}
    for r in gold_records:
        for t in r["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    homonym_count = sum(1 for r in gold_records if r["is_homonym_test"])
    print(f"Homonym Disambiguation Samples: {homonym_count} / {len(gold_records)}")
    print(f"Total Tokens: {sum(len(r['tokens']) for r in gold_records)}")
    print("BIO Label Distribution:")
    for tag in sorted(tag_counts.keys()):
        print(f"  {tag:<20}: {tag_counts[tag]}")


if __name__ == "__main__":
    main()
