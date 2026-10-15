#!/usr/bin/env python3
"""
scripts/fetch_gazetteer.py

Downloads MIT-licensed administrative geocode data from nuhil/bangladesh-geocode repository.
Saves:
  - data/raw/divisions.json
  - data/raw/districts.json
  - data/raw/upazilas.json
  - data/raw/unions.json
"""

import argparse
import logging
from pathlib import Path
import sys
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/nuhil/bangladesh-geocode/master"
RESOURCE_MAP = {
    "divisions.json": f"{BASE_URL}/divisions/divisions.json",
    "districts.json": f"{BASE_URL}/districts/districts.json",
    "upazilas.json": f"{BASE_URL}/upazilas/upazilas.json",
    "unions.json": f"{BASE_URL}/unions/unions.json",
}


def download_file(url: str, dest_path: Path, timeout: int = 30) -> None:
    logger.info("Fetching %s -> %s", url, dest_path)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # Ensure UTF-8 content is written properly
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    logger.info("Saved %s (%d bytes)", dest_path.name, len(response.content))


def fetch_all_gazetteer_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in RESOURCE_MAP.items():
        dest = output_dir / filename
        download_file(url, dest)
    logger.info("All gazetteer files downloaded successfully to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw geocode JSON files for Bangladesh administrative divisions."
    )
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "data" / "raw"
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=default_out,
        help=f"Target directory for downloaded JSON files (default: {default_out})",
    )
    args = parser.parse_args()

    try:
        fetch_all_gazetteer_data(args.output_dir)
    except Exception as e:
        logger.error("Failed to download gazetteer data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
