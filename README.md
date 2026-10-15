# Thikana-Engine

A fast, open-source Bangladeshi address parser and administrative gazetteer engine.

## Step 1: Administrative Gazetteer & Hierarchy

### Features
- **Four-Tier Administrative Hierarchy**:
  - `Division` (Tier 1) -> `District` (Tier 2) -> `Upazila` / `Thana` (Tier 3) -> `Union` / `Subarea` (Tier 4)
- **Bilingual Support**: Fast O(1) indexed lookup for both English (`name`) and Bengali (`bn_name`).
- **Homonym Resolution**: Captures administrative entities sharing identical names across districts (e.g., *Mirpur* in Dhaka vs. Kushtia; *Mohammadpur* in Dhaka, Magura, and Noakhali).
- **Metro Seeds**: Supplements rural upazila lists with metropolitan thanas and subareas for major urban centers (Dhaka Metro, Chattogram Metro, Sylhet Metro).
- **Unicode & Alias Normalization**: Handles Unicode NFC normalization (Bengali conjuncts, nuktas) and common historical aliases (e.g., *Chittagong* -> *Chattogram*, *Cumilla* -> *Comilla*, *মহম্মদপুর* <-> *মোহাম্মদপুর*).

---

### Project Structure
```
thikana-engine/
├── .venv/                     # Python virtual environment
├── data/
│   └── raw/                   # Raw administrative datasets
│       ├── divisions.json     # 8 Divisions
│       ├── districts.json     # 64 Districts
│       ├── upazilas.json      # 494 Rural Upazilas
│       ├── unions.json        # 4,540 Rural Unions
│       └── dhaka_ctg_thanas.json # 29 Metro Thanas & 150 Subareas
├── scripts/
│   └── fetch_gazetteer.py     # Download script from nuhil/bangladesh-geocode
├── src/
│   ├── __init__.py
│   ├── models.py              # AdminNode & AdminLevel Pydantic models
│   └── gazetteer.py           # BangladeshGazetteer in-memory hierarchy & search
├── tests/
│   ├── __init__.py
│   └── test_gazetteer.py      # Unit test suite (22 tests)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

### Quickstart

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Fetch Gazetteer Data**:
   ```bash
   python scripts/fetch_gazetteer.py
   ```

3. **Usage Example**:
   ```python
   from src.gazetteer import BangladeshGazetteer, AdminLevel

   gazetteer = BangladeshGazetteer()

   # Find homonyms across Bangladesh
   matches = gazetteer.find_by_exact_name("Mirpur")
   for node in matches:
       print(f"{node.name} ({node.level.value}) in {node.district.name}, {node.division.name}")
       # Output:
       # Mirpur (upazila) in Kushtia, Khulna
       # Mirpur (thana) in Dhaka, Dhaka
       # Mirpur (union) in Habiganj, Sylhet
       # Mirpur (union) in Sunamganj, Sylhet

   # Bilingual Bengali lookup
   results = gazetteer.find_by_exact_name("মোহাম্মদপুর")
   for node in results:
       print(node.full_name_bn)

   # Metro subarea lookup
   m10 = gazetteer.find_by_exact_name("Mirpur 10")[0]
   print(m10.full_name_en)
   # Output: Mirpur 10, Mirpur, Dhaka, Dhaka
   ```

4. **Run Unit Tests**:
   ```bash
   pytest -v
   ```
