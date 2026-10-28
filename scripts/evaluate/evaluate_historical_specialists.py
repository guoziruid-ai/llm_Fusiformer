#!/usr/bin/env python3
"""Evaluate configured expert(s) using historical labels, never a router.

Run as a module. Configure HISTORICAL_SOURCE and EVAL_CONFIG in this file;
defaults fail closed until the real historical artifacts are supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.cluster_specialists.config import HistoricalSourceConfig
from scripts.evaluate.evaluate_cluster_specialists import CFG, run

HISTORICAL_SOURCE = HistoricalSourceConfig()
EVAL_CONFIG = replace(CFG, historical_source=HISTORICAL_SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    if EVAL_CONFIG.historical_source is None:
        raise ValueError("historical_source must be configured")
    run(EVAL_CONFIG, check_config=args.check_config)


if __name__ == "__main__":
    main()
