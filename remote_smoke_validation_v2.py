"""Low-cost real-data validation smoke for the configured V2.1 checkpoint.

Run only inside the isolated remote test directory after activating fusiformer.
This is not a full validation score and never reads from the test split.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.evaluate.validate_fusiformer_v2_modernbert import (
    CHECKPOINT_CONFIG, DATA_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG,
)

ROOT = Path(__file__).resolve().parent
SMOKE_COUNT = 2
SMOKE_OUTPUT = ROOT / "runs" / "validation_smoke_2_samples.json"


class PrefixDataset(torch.utils.data.Dataset):
    def __init__(self, base, count: int):
        self.base = base
        self.data = base.data[:count]
        self.original_indices = base.original_indices[:count]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.base[index]

    def collect(self):
        return self.base.collect()


def main() -> None:
    from fusiformer_v2.dataset import get_dataset_v2
    from fusiformer_v2.evaluation import (
        _check_checkpoint, evaluate_loader, select_dataset, validate_settings,
    )
    from fusiformer_v2.model import FusiformerV2
    from loader.loader_cached import GraphCollectingCached
    from loader.transforms import (
        AddAngleFeature, AddEdgeFeature, AddNodeFeature, Compose,
    )
    from train_fusiformer_production_editor_v4 import load_torch_file

    checkpoint_path, _ = validate_settings(
        "validation", MODEL_CONFIG, DATA_CONFIG, CHECKPOINT_CONFIG, OUTPUT_CONFIG,
    )
    if SMOKE_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite smoke result: {SMOKE_OUTPUT}")
    checkpoint = load_torch_file(checkpoint_path, map_location="cpu")
    state = _check_checkpoint(checkpoint, MODEL_CONFIG, DATA_CONFIG, CHECKPOINT_CONFIG)
    model = FusiformerV2(
        targets=[MODEL_CONFIG["target"]], depth=MODEL_CONFIG["model_depth"],
        edge_input_dim=MODEL_CONFIG["edge_input_dim"],
        triplet_input_dim=MODEL_CONFIG["triplet_input_dim"],
        embed_dim=MODEL_CONFIG["embed_dim"], num_heads=MODEL_CONFIG["num_heads"],
        text_dim=MODEL_CONFIG["text_dim"],
        text_proj_dim=MODEL_CONFIG["text_proj_dim"],
        text_proj_hidden_dims=MODEL_CONFIG["text_proj_hidden_dims"],
        text_proj_dropout=MODEL_CONFIG["text_proj_dropout"],
        stage=MODEL_CONFIG["stage"], num_queries=MODEL_CONFIG["num_queries"],
        cross_dropout=MODEL_CONFIG["cross_dropout"],
        text_encoder_name=MODEL_CONFIG["text_encoder_name"],
        text_model_name=MODEL_CONFIG["text_model_path"],
        text_pooling=MODEL_CONFIG["text_pooling"],
        text_preprocessing=MODEL_CONFIG["text_preprocessing"],
        load_text_encoder=False, freeze_text_encoder=True,
    )
    model.use_precomputed_text_embeddings = True
    model.load_state_dict(state, strict=True)
    device = torch.device(DATA_CONFIG["device_name"])
    model = model.to(device).eval()
    transform = Compose([
        AddNodeFeature(), AddEdgeFeature(), AddAngleFeature(),
        GraphCollectingCached(["graph", "line_graph"], [MODEL_CONFIG["target"]]),
    ])
    datasets = get_dataset_v2(
        data_path=DATA_CONFIG["data_path"],
        ratio_train_val_test=DATA_CONFIG["data_split_ratio"],
        n_train_val_test=DATA_CONFIG["data_split_counts"],
        transforms=transform, graph=True, line_graph=True,
        text_embedding_cache_path=DATA_CONFIG["text_cache_path"],
        text_field=DATA_CONFIG["text_field"],
        text_encoder_name=MODEL_CONFIG["text_encoder_name"],
        text_model_name=MODEL_CONFIG["text_model_path"],
    )
    validation = select_dataset(datasets, "validation")
    if len(validation) < SMOKE_COUNT:
        raise ValueError("Validation split is smaller than smoke count")
    selected = PrefixDataset(validation, SMOKE_COUNT)
    loader = torch.utils.data.DataLoader(
        selected, batch_size=SMOKE_COUNT, num_workers=0, shuffle=False,
        drop_last=False, collate_fn=selected.collect(),
    )
    metrics, rows = evaluate_loader(
        model, loader, selected, device,
        text_mode=MODEL_CONFIG["text_mode"],
        amp_enabled=DATA_CONFIG["use_amp"],
    )
    result = {
        "status": "validation_smoke_only_not_full_score",
        "sample_count": len(rows),
        "full_split_sizes": [len(ds) if ds is not None else 0 for ds in datasets],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data_path": str(DATA_CONFIG["data_path"]),
        "text_cache_path": str(DATA_CONFIG["text_cache_path"]),
        "device": str(device),
        "metrics": metrics,
        "predictions": rows,
    }
    SMOKE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with SMOKE_OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"SMOKE_OUTPUT={SMOKE_OUTPUT}")


if __name__ == "__main__":
    main()
