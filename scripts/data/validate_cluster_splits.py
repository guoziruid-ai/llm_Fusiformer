#!/usr/bin/env python3
"""Read-only validation for a generated cluster master manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cluster_specialists.manifests import (
    read_manifest, validate_manifest)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "runs/cluster_specialists/manifests_seed42/cluster_manifest.csv"
EXPECTED_SAMPLE_COUNT: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = read_manifest(MANIFEST_PATH)
    report = validate_manifest(rows, expected_count=EXPECTED_SAMPLE_COUNT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
