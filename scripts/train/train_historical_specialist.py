#!/usr/bin/env python3
"""Train one configured expert using explicitly verified historical labels.

Run as a module. Configure HISTORICAL_SOURCE and TRAIN_CONFIG in this file;
defaults fail closed until the real historical artifacts are supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from scripts.experiments.cluster_specialists.config import HistoricalSourceConfig
from scripts.train.train_cluster_specialist import CFG, run

HISTORICAL_SOURCE = HistoricalSourceConfig()
TRAIN_CONFIG = replace(CFG, historical_source=HISTORICAL_SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if TRAIN_CONFIG.historical_source is None:
        raise ValueError("historical_source must be configured")
    run(TRAIN_CONFIG, check_config=args.check_config)


if __name__ == "__main__":
    main()
