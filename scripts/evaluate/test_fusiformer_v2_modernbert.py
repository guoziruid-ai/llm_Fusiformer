#!/usr/bin/env python3
"""Independently evaluate a specified Fusiformer V2.1 ModernBERT checkpoint on test."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Edit this entire section to MATCH THE CHECKPOINT'S original training Config.
# The defaults below match train_fusiformer_v2_modernbert.py (no dropout).
MODEL_CONFIG = {
    "model_name": "FusiformerV2.1-ModernBERT",  # actual model/version
    "target": "e_form",                 # JSON regression label and model head
    "stage": "a2",                      # a1/a2/a3, must match checkpoint
    "num_queries": 16,                 # A3 material queries
    "cross_dropout": 0.0,              # A2/A3 fusion dropout
    "model_depth": 6,                  # coGN depth
    "edge_input_dim": 80,              # edge feature size
    "triplet_input_dim": 40,           # angle feature size
    "embed_dim": 128,                  # graph hidden size
    "num_heads": 4,                    # attention heads
    "text_dim": 768,                   # ModernBERT rank-3 cache final dimension
    "text_proj_dim": 64,              # text projection output size
    "text_proj_hidden_dims": (256, 128),  # text projection hidden layers
    "text_proj_dropout": 0.0,          # text projection dropout
    "text_encoder_name": "modernbert",  # required cache encoder
    "text_model_path": ROOT / "external/ModernBERT-base",  # cache provenance; weights are not loaded
    "text_pooling": "mean",             # cached late-chunking path
    "text_preprocessing": "none",       # cached late-chunking path
    "text_mode": "normal",             # normal or bypass (same checkpoint)
}

DATA_CONFIG = {
    "data_path": ROOT / "data/dataset_final_robocrys.json",  # original ordered JSON
    "text_cache_path": ROOT / "data/v2/modernbert_late_robocrys_segments.npy",  # [N,M,768] plus mask/meta
    "text_field": "description",       # cache fingerprint field
    "data_split_ratio": (0.8, 0.1, 0.1),  # training split, seed 7
    "data_split_counts": None,         # use training counts instead of ratio if applicable
    "batch_size": 32,                  # inference batches; last batch is retained
    "num_workers": 8,                  # set 0 for debugging or Windows
    "device_name": "cuda",             # set cpu if CUDA is unavailable
    "pin_memory": True,               # only effective on CUDA
    "use_amp": False,                  # keep False for training-equivalent precision
}

CHECKPOINT_CONFIG = {
    "path": ROOT / "checkpoints/best_model_weights.pt",  # set this to the selected V2.1 checkpoint
    "weight_key": "model_state_dict",  # v4 full and lightweight checkpoint field
    "expected_format_version": 4,     # v4 checkpoint format
    "expected_model_version": "V2.1",  # training_config.fusiformer_version
    "strict": True,                   # never silently skip V2 parameters
}

OUTPUT_CONFIG = {
    "directory": ROOT / "runs/v21_test_eval",  # choose a fresh run-specific directory
    "metrics_filename": "test_metrics.json",
    "predictions_filename": "test_predictions.csv",
    "metadata_filename": "test_run_config.json",
}


def main() -> None:
    from fusiformer_v2.evaluation import run_evaluation
    result = run_evaluation("test", MODEL_CONFIG, DATA_CONFIG, CHECKPOINT_CONFIG, OUTPUT_CONFIG)
    print(result)


if __name__ == "__main__":
    main()
