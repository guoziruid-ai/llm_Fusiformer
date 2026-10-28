"""Two-sample in-memory V2 scratch assembly smoke; no checkpoints or training run.

Run only in an isolated environment with the configured real JSON/cache/model.
"""
from __future__ import annotations

import logging

import torch

from scripts.train.train_fusiformer_v2_modernbert import (
    CFG, inspect_cache, setup_training, validate_user_config,
)


SMOKE_BATCH_SIZE = 2


def main() -> int:
    validate_user_config(CFG)
    inspect_cache(CFG)
    engine = setup_training(CFG)
    args = engine.build_runtime_config()
    if args.mode != "scratch" or args.text_encoder_name != "modernbert":
        raise ValueError("Smoke requires the configured ModernBERT scratch mode")
    args.batch_size = SMOKE_BATCH_SIZE
    args.num_workers = 0
    args.pin_mem = False
    args.drop_last = False
    engine.validate_config(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    logger = logging.getLogger("v2_training_assembly_smoke")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    train_dataset, validation, test, train_loader, _, _ = engine.build_dataloaders(
        args, device, logger,
    )
    model = engine.build_model(args, device)
    optimizer = engine.build_optimizer(model, args, actual_lr=float(CFG.absolute_lr))
    model.train()
    inputs, targets = next(iter(train_loader))
    inputs = engine.to_device(inputs, device)
    targets = engine.normalize_targets(targets, device)
    outputs = engine.normalize_outputs(model(inputs))
    if outputs.shape != targets.shape:
        raise ValueError(f"Output/target mismatch: {outputs.shape} != {targets.shape}")
    loss = torch.nn.functional.mse_loss(outputs, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite real-data smoke loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not trainable or not all(torch.isfinite(p.grad).all() for p in trainable):
        raise RuntimeError("Missing or non-finite gradients")
    changed_parameter = next((p for p in trainable if p.grad.detach().abs().sum() > 0), None)
    if changed_parameter is None:
        raise RuntimeError("No nonzero parameter gradient")
    before = changed_parameter.detach().clone()
    optimizer.step()
    if torch.equal(before, changed_parameter.detach()):
        raise RuntimeError("Optimizer step did not update a trainable parameter")
    print(
        "PASS real-data scratch assembly: "
        f"split_sizes=({len(train_dataset)}, {len(validation)}, {len(test)}) "
        f"batch={SMOKE_BATCH_SIZE} loss={loss.item():.8g} "
        f"optimizer_groups={len(optimizer.param_groups)} device={device}"
    )
    print("No training loop, checkpoint or result file was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
