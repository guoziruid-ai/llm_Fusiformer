"""Negative preflight: known test predictions have no saved same-split routes."""
import json
from dataclasses import replace
from scripts.diagnostics.analyze_frozen_history import preflight
from scripts.diagnostics.frozen_history_preflight_20261028 import CFG

if __name__ == '__main__':
    print(json.dumps(preflight(replace(CFG, require_prediction_routes=True))))
