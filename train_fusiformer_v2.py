#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fusiformer-V2 production launcher (real Abstract branch integration).

Reuses v6's proven connection to v4 optimizer, DGL data transforms, scheduler,
logging, early stopping, and checkpoint format. Only V2 model/cache interfaces
are replaced. The original v4/v6 files remain intact for A0 comparisons.

1. Generate the V2 semantic cache using tools/precompute_semantic_segments_v2.py.
2. Edit V2 configuration below (data, cache, target, stage, output).
3. Run: python train_fusiformer_v2.py

IMPORTANT: Start in SCRATCH mode. A0 checkpoints are NOT compatible by default.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import train_fusiformer_production_editor_v6 as baseline
from fusiformer_v2.dataset import get_dataset_v2
from fusiformer_v2.model import FusiformerV2


# ============================ User configuration ============================
# Use one stage per reproducible experiment: a1, a2, or a3.
V2_STAGE = "a2"
V2_NUM_QUERIES = 16
V2_CROSS_DROPOUT = 0.1

# Original data must use the same ordered JSON from cache generation.
DATA_PATH = "./data/dataset_final_merged_with_robocrys.json"
TEXT_CACHE = "./data/v2/matscibert_segments.npy"   # rank-3 [N,M,D], not legacy rank-2 cache
OUTPUT_ROOT = "./runs/fusiformer_v2_a2_matscibert"
TARGET = "e_form"

# Training: reuse all other v6/v4 settings (batch_size, optimizer, scheduler).
# Recommended initial A2 test: 32 micro-batch, 2 accumulation steps, seed 42.
# For Qwen, change baseline.text_encoder_name, text_model_name and
# text_preprocessing consistently with the generated V2 cache metadata.
# ===========================================================================


if V2_STAGE not in {"a1", "a2", "a3"}:
    raise ValueError("V2_STAGE must be a1/a2/a3")

baseline.data_path = DATA_PATH
baseline.target_name = TARGET
baseline.text_embedding_cache_path = TEXT_CACHE
baseline.text_embedding_meta_path = None
baseline.output_root = Path(OUTPUT_ROOT)
baseline.training_mode = "scratch"
baseline.resume_checkpoint_path = None
baseline.weights_checkpoint_path = None
baseline.eval_checkpoint_path = None
baseline.use_precomputed_text_embeddings = True
baseline.freeze_text_encoder = True
baseline.use_text = True
baseline.SCRIPT_VERSION = f"fusiformer-v2-{V2_STAGE}-native-abstract-20261014"

# v6 already registers custom production boundaries on its v4 engine object.
engine = baseline.engine
_parent_config = baseline.build_runtime_config
_parent_training_config = baseline.build_training_config
_parent_clone = engine.clone_inputs_without_text


def build_runtime_config():
    args = _parent_config()
    args.v2_stage = V2_STAGE
    args.v2_num_queries = int(V2_NUM_QUERIES)
    args.v2_cross_dropout = float(V2_CROSS_DROPOUT)
    if args.mode != "scratch":
        raise ValueError("V2 initially requires training_mode='scratch'; incompatible A0 checkpoints")
    if not args.use_precomputed_text_embeddings:
        raise ValueError("V2 requires a precomputed rank-3 semantic cache")
    return args


def build_model(args, device):
    arr = np.load(args.text_embedding_cache_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError("V2 expects [N,M,D] semantic cache; legacy pooled [N,D] cannot be reused")
    hidden_size = int(arr.shape[-1])
    del arr
    model = FusiformerV2(
        targets=[args.target], depth=args.depth,
        edge_input_dim=args.edge_input_dim,
        triplet_input_dim=args.triplet_input_dim,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        text_dim=hidden_size, text_proj_dim=args.text_proj_dim,
        text_proj_hidden_dims=args.text_proj_hidden_dims,
        text_proj_dropout=args.text_proj_dropout,
        stage=args.v2_stage, num_queries=args.v2_num_queries,
        cross_dropout=args.v2_cross_dropout,
        text_encoder_name=args.text_encoder_name,
        text_model_name=args.text_model_name,
        text_pooling=args.text_pooling,
        text_preprocessing=args.text_preprocessing,
        load_text_encoder=False, freeze_text_encoder=True,
    )
    model.use_precomputed_text_embeddings = True
    return model.to(device)


def build_dataloaders(args, device, logger):
    # Existing v6 function owns graph transforms and DataLoader options; replace
    # only its bound get_dataset factory for the duration of this program.
    return baseline.build_dataloaders(args, device, logger)


def clone_inputs_without_text(inputs):
    copied = _parent_clone(inputs)
    if isinstance(copied, dict):
        copied["text_segments"] = None
        copied["text_segment_mask"] = None
        copied["text_mode"] = "bypass"
    return copied


def parameter_lr_category(name):
    if name.startswith(("cross_modal.", "v2_text_to_graph.", "material_queries.")):
        return "text"
    return baseline.parameter_lr_category(name)


def build_training_config(*args, **kwargs):
    payload = _parent_training_config(*args, **kwargs)
    payload.update({
        "fusiformer_version": "V2",
        "v2_stage": V2_STAGE,
        "v2_num_queries": V2_NUM_QUERIES,
        "v2_cross_dropout": V2_CROSS_DROPOUT,
        "v2_semantic_cache": TEXT_CACHE,
    })
    return payload


# Patch the exact same runtime hooks as v6, before v4.main() executes.
baseline.get_dataset = get_dataset_v2
engine.build_runtime_config = build_runtime_config
engine.build_model = build_model
engine.build_dataloaders = build_dataloaders
engine.clone_inputs_without_text = clone_inputs_without_text
engine.parameter_lr_category = parameter_lr_category
engine.build_training_config = build_training_config
engine.SCRIPT_VERSION = baseline.SCRIPT_VERSION


if __name__ == "__main__":
    engine.main()
