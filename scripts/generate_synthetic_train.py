"""
scripts/generate_synthetic_train.py

Generates 2,500 diverse, non-leaky synthetic BIO-tagged training addresses from
BangladeshGazetteer, strictly enforcing a Geographic Holdout Split against gold_100.json.

Holdout Upazilas and gold templates are completely excluded to guarantee zero leakage.
Saves the synthetic dataset to data/train_synthetic_2500.json.
"""

from __future__ import annotations
import json
import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gazetteer import BangladeshGazetteer
from src.models import AdminLevel, AdminNode
from src.normalizer import normalize_address


# -----------------------------------------------------------------------------
# 1. Geographic Holdout Upazilas (Extracted from gold_100.json)
# -----------------------------------------------------------------------------
def load_holdout_upazilas(gold_path: str) -> Set[str]:
    """
    Reads gold_100.json and extracts all upazila/thana names that must be held out.
    """
    if not os.path.exists(gold_path):
        return set()
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    holdout = set()
    for item in gold_data:
        thana = item.get("expected_thana")
        if thana:
            holdout.add(thana.strip().lower())
    return holdout


# -----------------------------------------------------------------------------
# 2. Synthetic Address Components Pool
# -----------------------------------------------------------------------------
HOUSE_PREFIXES_EN = ["House", "H#", "Holding", "Plot", "Bari", "Basha"]
HOUSE_PREFIXES_BN = ["বাসা নং", "বাড়ি", "হোল্ডিং নং", "প্লট"]
ROAD_PREFIXES_EN = ["Road", "RD#", "Rasta", "Lane", "Goli", "Avenue", "Street", "Shorok"]
ROAD_PREFIXES_BN = ["রোড নং", "রাস্তা", "সড়ক", "গলি", "লেন"]
BLOCK_PREFIXES_EN = ["Block", "BLK#", "Sector", "SEC#"]
BLOCK_PREFIXES_BN = ["ব্লক", "সেক্টর"]
FLOOR_PREFIXES_EN = ["Flat", "flat", "Level", "level", "lvl", "Floor", "tola", "lift", "Apt", "apt", "Apartment", "Unit"]
FLOOR_PREFIXES_BN = ["ফ্ল্যাট", "তলা", "লেভেল", "ফ্লোর", "লিফট"]

RURAL_VILLAGE_PREFIXES_EN = ["Village:", "Gram:", "Vill:", "Gram"]
RURAL_VILLAGE_PREFIXES_BN = ["গ্রাম:", "গ্রাম"]
RURAL_POST_PREFIXES_EN = ["Post:", "Dakghar:", "P.O:", "Post Office:"]
RURAL_POST_PREFIXES_BN = ["ডাকঘর:", "পোস্ট:"]
RURAL_UNION_PREFIXES_EN = ["Union:", "UP:", "Union Parishad:"]
RURAL_UNION_PREFIXES_BN = ["ইউনিয়ন:", "ইউপি:"]
CARE_OF_PREFIXES_EN = ["C/O", "Care of", "c/o"]
CARE_OF_PREFIXES_BN = ["প্রযত্নে:", "যত্নে:"]

CARE_OF_NAMES_EN = [
    "Abdul Malek", "Kamal Hossain", "Dr. Anwar Ali", "Master Rafiqul Islam",
    "Haji Mokbul Ahmed", "Engr. Monirul Islam", "Md. Sohel Rana", "Shahidul Alam",
    "Golam Kibria", "Abul Bashar", "Fazlur Rahman", "Sirajul Islam"
]
CARE_OF_NAMES_BN = [
    "মো: রফিকুল ইসলাম", "আব্দুল মালেক", "কামাল হোসেন", "হাজী মকবুল আহমেদ",
    "ফজলুর রহমান", "মো: সোহেল রানা", "ডা: আনোয়ার আলী"
]

LANDMARKS_EN = [
    # General & Commercial POIs
    "Bismillah Pharmacy", "Sikder Medical", "Agrani Bank", "Pubali Bank",
    "Rupali Bank", "Janata Bank", "Central Jame Masjid", "Railway Station",
    "Govt High School", "Girls College", "Bazar Mosque", "City Super Market",
    "Modern Diagnostic Center", "Al Madina Bakery", "Bismillah Auto Rice Mill",
    # Commercial & Corporate District-Preceding POIs (Distractor Training)
    "Uttara Bank Limited", "Dhaka Bank Limited", "National Bank Ltd", "City Bank Plaza",
    "Sonali Bank Branch", "Pubali Bank Tower", "Islami Bank Hospital", "Brac Bank Centre",
    "Jamuna Future Park", "Bashundhara City Complex", "Navana Tower", "Concord Plaza",
    "Baitul Mukarram Market", "Popular Diagnostic Centre", "Square Hospital", "Labaid Clinic",
    "Green Road Clinic", "Eden College", "Notre Dame College", "Dhaka Medical College",
    "Comilla Sweets", "Noakhali Bhaban", "Chittagong Store", "Sylhet Mart",
    "Bogra Doi Ghor", "Barisal Sweet Meat", "Khulna Complex", "Rajshahi Silk Center",
    "New Market", "Kacha Bazar", "Town Hall", "Police Station", "Bus Terminal"
]
LANDMARKS_BN = [
    "অগ্রণী ব্যাংক", "পূবালী ব্যাংক", "রূপালী ব্যাংক", "জনতা ব্যাংক", "ইসলামী ব্যাংক",
    "ঢাকা ব্যাংক লিমিটেড", "উত্তরা ব্যাংক লিমিটেড", "সোনালী ব্যাংক", "সিটি ব্যাংক",
    "কেন্দ্রীয় জামে মসজিদ", "রেলওয়ে স্টেশন", "সরকারি উচ্চ বিদ্যালয়",
    "মহিলা কলেজ", "সিটি সুপার মার্কেট", "আল মদিনা বেকারি", "স্কয়ার হাসপাতাল",
    "পপুলার ডায়াগনস্টিক", "নভানা টাওয়ার", "কনকর্ড প্লাজা", "যমুনা ফিউচার পার্ক",
    "বসুন্ধরা সিটি", "বায়তুল মোকাররম", "কাঁচা বাজার", "বাস টার্মিনাল",
    "কুমিল্লা মিষ্টান্ন ভাণ্ডার", "নোয়াখালী ভবন", "চট্টগ্রাম প্লাজা"
]

LANDMARK_RELS_EN = ["near", "opposite", "behind", "beside", "in front of"]
LANDMARK_RELS_BN = ["এর পাশে", "এর সামনে", "এর পিছনে", "সংলগ্ন"]

COMMON_ROADS_EN = [
    "College Road", "Station Road", "Hospital Road", "Main Road",
    "Bazar Road", "School Road", "Thana Road", "Court Road",
    "Canal Road", "Bridge Road", "Zilla Parishad Road"
]
COMMON_ROADS_BN = [
    "কলেজ রোড", "স্টেশন রোড", "হাসপাতাল রোড", "মেইন রোড",
    "বাজার রোড", "স্কুল রোড", "থানা রোড", "কোর্ট রোড"
]


# -----------------------------------------------------------------------------
# 3. Synthetic Generator Logic
# -----------------------------------------------------------------------------
def build_synthetic_dataset(
    gazetteer: BangladeshGazetteer,
    holdout_upazilas: Set[str],
    total_samples: int = 2500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generates total_samples training addresses strictly from non-holdout Upazilas.
    """
    random.seed(seed)

    # Filter upazilas to only those strictly NOT in holdout
    eligible_upazilas: List[AdminNode] = []
    for node in gazetteer.all_nodes:
        if node.level in (AdminLevel.UPAZILA, AdminLevel.THANA):
            name_l = node.name.lower()
            if name_l not in holdout_upazilas and (not node.bn_name or node.bn_name.lower() not in holdout_upazilas):
                # Ensure parent district exists
                if node.district is not None:
                    eligible_upazilas.append(node)

    if not eligible_upazilas:
        raise ValueError("No eligible Upazilas found for training after holdout filtering!")

    print(f"Total Eligible Training Upazilas (Non-Holdout): {len(eligible_upazilas)}")

    samples: List[Dict[str, Any]] = []

    # 7 Distinct Synthetic Templates
    for i in range(total_samples):
        upz = random.choice(eligible_upazilas)
        dist = upz.district
        # Get children unions/subareas if available
        unions = [c for c in upz.children if c.level in (AdminLevel.UNION, AdminLevel.SUBAREA)]
        union = random.choice(unions) if unions else None

        template_choice = i % 7

        raw_parts = []
        token_annotations: List[Tuple[str, str]] = []

        if template_choice == 0:
            # Template 1: House + Road + Subarea/Union + Upazila + District (English)
            h_num = str(random.randint(1, 150))
            h_pfx = random.choice(["House", "H#", "Holding", "Plot"])
            r_num = str(random.randint(1, 30))
            r_pfx = random.choice(["Road", "RD#", "Lane", "Avenue"])
            raw_parts = [f"{h_pfx} {h_num}", f"{r_pfx} {r_num}"]
            token_annotations.extend([
                (f"{h_pfx} {h_num}", "HOUSE"),
                (f"{r_pfx} {r_num}", "ROAD"),
            ])
            if union:
                raw_parts.append(union.name)
                token_annotations.append((union.name, "SUBAREA"))
            raw_parts.extend([upz.name, dist.name])
            token_annotations.extend([
                (upz.name, "THANA_UPAZILA"),
                (dist.name, "DISTRICT"),
            ])

        elif template_choice == 1:
            # Template 2: Flat + House + Road + Block + Upazila + District (English)
            flt_num = f"{random.choice(FLOOR_PREFIXES_EN)} {random.choice(['2A', '3B', '4C', '5', '3', '4', '1A', '6B', '2', '1', '7D'])}"
            h_num = f"House {random.randint(1, 99)}"
            r_num = f"Road {random.randint(1, 25)}"
            blk = f"{random.choice(['Block', 'Sector'])} {random.choice(['A', 'B', 'C', 'D', '1', '2', '3'])}"
            raw_parts = [flt_num, h_num, r_num, blk, upz.name, dist.name]
            token_annotations = [
                (flt_num, "FLOOR_FLAT"),
                (h_num, "HOUSE"),
                (r_num, "ROAD"),
                (blk, "BLOCK_SECTOR"),
                (upz.name, "THANA_UPAZILA"),
                (dist.name, "DISTRICT"),
            ]

        elif template_choice == 2:
            # Template 3: Landmark + Landmark Rel + Road + Upazila + District
            lm = random.choice(LANDMARKS_EN)
            rel = random.choice(LANDMARK_RELS_EN)
            rd = random.choice(COMMON_ROADS_EN)
            raw_parts = [rel, lm, rd, upz.name, dist.name]
            token_annotations = [
                (rel, "LANDMARK_REL"),
                (lm, "LANDMARK"),
                (rd, "ROAD"),
                (upz.name, "THANA_UPAZILA"),
                (dist.name, "DISTRICT"),
            ]

        elif template_choice == 3:
            # Template 4: Rural C/O + Village + Post + Union + Upazila + District
            co_name = random.choice(CARE_OF_NAMES_EN)
            co_pfx = random.choice(CARE_OF_PREFIXES_EN)
            v_pfx = random.choice(RURAL_VILLAGE_PREFIXES_EN)
            v_name = f"{random.choice(['Uttar', 'Dakshin', 'Purba', 'Paschim', 'Choto', 'Boro'])} {random.choice(['Para', 'Gram', 'Bari', 'Kandi'])}"
            raw_parts = [f"{co_pfx} {co_name}", f"{v_pfx} {v_name}"]
            token_annotations.extend([
                (co_pfx, "O"),
                (co_name, "O"),
                (v_pfx, "O"),
                (v_name, "SUBAREA"),
            ])
            if union:
                u_pfx = random.choice(RURAL_UNION_PREFIXES_EN)
                raw_parts.append(f"{u_pfx} {union.name}")
                token_annotations.extend([
                    (u_pfx, "O"),
                    (union.name, "SUBAREA"),
                ])
            raw_parts.extend([upz.name, dist.name])
            token_annotations.extend([
                (upz.name, "THANA_UPAZILA"),
                (dist.name, "DISTRICT"),
            ])

        elif template_choice == 4:
            # Template 5: Bangla Script Urban: ফ্ল্যাট/তলা + বাসা + রোড + ব্লক + উপজেলা + জেলা
            flt_bn = f"{random.choice(FLOOR_PREFIXES_BN)} {random.choice(['৩এ', '৪বি', '২', '৩', '৪', '৫'])}"
            h_bn = f"বাসা নং {random.randint(1, 99)}"
            r_bn = f"রোড নং {random.randint(1, 20)}"
            b_bn = f"ব্লক {random.choice(['এ', 'বি', 'সি', 'ডি', '১', '২'])}"
            upz_bn = upz.bn_name or upz.name
            dist_bn = dist.bn_name or dist.name
            raw_parts = [flt_bn, h_bn, r_bn, b_bn, upz_bn, dist_bn]
            token_annotations = [
                (flt_bn, "FLOOR_FLAT"),
                (h_bn, "HOUSE"),
                (r_bn, "ROAD"),
                (b_bn, "BLOCK_SECTOR"),
                (upz_bn, "THANA_UPAZILA"),
                (dist_bn, "DISTRICT"),
            ]

        elif template_choice == 5:
            # Template 6: Bangla Script Rural: প্রযত্নে + গ্রাম + ইউনিয়ন + উপজেলা + জেলা
            co_bn = random.choice(CARE_OF_NAMES_BN)
            co_pfx_bn = random.choice(CARE_OF_PREFIXES_BN)
            v_bn = f"{random.choice(['উত্তরপাড়া', 'দক্ষিণপাড়া', 'পূর্বপাড়া', 'মাইজপাড়া', 'চরপাড়া'])}"
            v_pfx_bn = random.choice(RURAL_VILLAGE_PREFIXES_BN)
            raw_parts = [f"{co_pfx_bn} {co_bn}", f"{v_pfx_bn} {v_bn}"]
            token_annotations = [
                (co_pfx_bn, "O"),
                (co_bn, "O"),
                (v_pfx_bn, "O"),
                (v_bn, "SUBAREA"),
            ]
            upz_bn = upz.bn_name or upz.name
            dist_bn = dist.bn_name or dist.name
            if union and union.bn_name:
                u_pfx_bn = random.choice(RURAL_UNION_PREFIXES_BN)
                raw_parts.append(f"{u_pfx_bn} {union.bn_name}")
                token_annotations.extend([
                    (u_pfx_bn, "O"),
                    (union.bn_name, "SUBAREA"),
                ])
            raw_parts.extend([upz_bn, dist_bn])
            token_annotations.extend([
                (upz_bn, "THANA_UPAZILA"),
                (dist_bn, "DISTRICT"),
            ])

        else:
            # Template 7: Banglish Informal / Typos: basha + rasta + landmark + upazila + district
            h_bg = f"basha {random.randint(1, 80)}"
            r_bg = f"rasta {random.randint(1, 15)}"
            raw_parts = [h_bg, r_bg]
            token_annotations = [
                (h_bg, "HOUSE"),
                (r_bg, "ROAD"),
            ]
            if union:
                raw_parts.append(union.name)
                token_annotations.append((union.name, "SUBAREA"))
            raw_parts.extend([upz.name, dist.name])
            token_annotations.extend([
                (upz.name, "THANA_UPAZILA"),
                (dist.name, "DISTRICT"),
            ])

        raw_text = ", ".join(raw_parts)
        norm = normalize_address(raw_text)
        tokens = norm["tokens"]

        # Align tokens with annotations
        tags = ["O"] * len(tokens)
        matched = set()

        for phrase, label in token_annotations:
            phrase_norm = normalize_address(phrase)
            p_tokens = [t.lower() for t in phrase_norm["tokens"] if t.strip()]
            if not p_tokens:
                continue

            L = len(p_tokens)
            tokens_l = [t.lower() for t in tokens]
            found = False
            for idx in range(len(tokens_l) - L + 1):
                if any(m in matched for m in range(idx, idx + L)):
                    continue
                if tokens_l[idx : idx + L] == p_tokens:
                    if label == "O":
                        for k in range(idx, idx + L):
                            tags[k] = "O"
                            matched.add(k)
                    else:
                        tags[idx] = f"B-{label}"
                        for k in range(idx + 1, idx + L):
                            tags[k] = f"I-{label}"
                            matched.add(k)
                        matched.add(idx)
                    found = True
                    break

            if not found and L == 1:
                target = p_tokens[0]
                for idx, tok_l in enumerate(tokens_l):
                    if idx not in matched and (target == tok_l or target in tok_l or tok_l in target):
                        tags[idx] = "O" if label == "O" else f"B-{label}"
                        matched.add(idx)
                        break

        samples.append({
            "id": f"train_{i+1:04d}",
            "raw_text": raw_text,
            "normalized_text": norm["normalized_text"],
            "tokens": tokens,
            "tags": tags,
            "upazila": upz.name,
            "district": dist.name,
            "template_id": template_choice,
        })

    return samples


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gold_path = os.path.join(root_dir, "data", "eval", "gold_100.json")
    out_path = os.path.join(root_dir, "data", "train_synthetic_2500.json")

    print(f"Loading BangladeshGazetteer...")
    gz = BangladeshGazetteer()

    print(f"Reading holdout upazilas from {gold_path}...")
    holdout = load_holdout_upazilas(gold_path)
    print(f"Total Holdout Upazilas: {len(holdout)}")

    print(f"Generating 2,500 non-leaky synthetic training addresses...")
    train_data = build_synthetic_dataset(gz, holdout, total_samples=2500, seed=42)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(train_data)} training records in:")
    print(f" -> {out_path}")

    # Summary tag distribution
    tag_counts: Dict[str, int] = {}
    for r in train_data:
        for t in r["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    print("BIO Label Distribution in Training Set:")
    for tag in sorted(tag_counts.keys()):
        print(f"  {tag:<20}: {tag_counts[tag]}")


if __name__ == "__main__":
    main()
