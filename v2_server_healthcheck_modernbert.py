#!/usr/bin/env python3
"""Run existing V2 server healthcheck using ModernBERT cache configuration.

Requires the previously deployed v2_server_healthcheck.py in the same folder.
This wrapper only configures the runtime, it does not modify checkpoints.
"""
import train_fusiformer_v2_modernbert  # noqa: F401  (must import FIRST)
import v2_server_healthcheck as checker

if __name__ == "__main__":
    raise SystemExit(checker.main())
