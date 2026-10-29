import argparse
import csv
import json
import logging
import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from loader.loader import get_dataset
from loader.transforms import (
    Compose,
    AddNodeFeature,
    AddEdgeFeature,
    AddAngleFeature,
    GraphCollecting,
)
from engine import to_device

try:
    from models.TextGuidedFusiformer import TextGuidedFusiformer
except ImportError:
    from TextGuidedFusiformer import TextGuidedFusiformer


# =======================
# 基本配置
# =======================
data_path = "./data/my_dataset.json"
target_name = "e_form"

qwen_model_name = "/root/models/Qwen2.5-1.5B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 小样本过拟合测试
n_train = 32
n_val = 0
n_test = 0

batch_size = 2
epochs = 30
lr = 1e-4
weight_decay = 1e-5

use_text = True  # True: 图+文本；False: 只用图

# 日志与可视化配置
base_log_dir = "./runs/dataset_effect"
log_interval = 50


# =======================
# 日志与可视化工具函数
# =======================
def make_run_dir(base_dir):
    """创建本次实验的独立日志目录，避免覆盖旧实验。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir):
    """同时输出到终端和 train.log 文件。"""
    logger = logging.getLogger("dataset_effect")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(run_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_section(logger, title):
    logger.info("")
    logger.info("========== %s ==========" % title)


def save_config(run_dir, config_dict):
    """保存本次实验配置，方便复现实验。"""
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)


def create_csv_writer(csv_path, fieldnames):
    """创建 CSV writer，并写入表头。"""
    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    f.flush()
    return f, writer


def safe_float(value):
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_metrics_from_csv(log_dir):
    """
    根据保存下来的 CSV 日志重新绘图。
    可在训练结束后自动调用，也可用：
        python test_dataset_effect_with_logging.py --plot-only --log-dir ./runs/dataset_effect/某次实验目录
    """
    log_dir = Path(log_dir)
    epoch_csv = log_dir / "epoch_metrics.csv"
    step_csv = log_dir / "step_metrics.csv"

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。可执行：pip install matplotlib")
        return []

    saved_figures = []

    if epoch_csv.exists():
        rows = []
        with open(epoch_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            epochs_list = [int(row["epoch"]) for row in rows]
            train_mse = [safe_float(row.get("train_mse")) for row in rows]
            val_mse = [safe_float(row.get("val_mse")) for row in rows]
            train_mae = [safe_float(row.get("train_mae")) for row in rows]
            val_mae = [safe_float(row.get("val_mae")) for row in rows]
            train_rmse = [safe_float(row.get("train_rmse")) for row in rows]
            val_rmse = [safe_float(row.get("val_rmse")) for row in rows]

            plt.figure()
            plt.plot(epochs_list, train_mse, marker="o", label="train_mse")
            if any(v is not None for v in val_mse):
                plt.plot(epochs_list, val_mse, marker="o", label="val_mse")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE curve")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            mse_fig = log_dir / "mse_curve.png"
            plt.savefig(mse_fig, dpi=200)
            plt.close()
            saved_figures.append(str(mse_fig))

            plt.figure()
            plt.plot(epochs_list, train_mae, marker="o", label="train_mae")
            plt.plot(epochs_list, train_rmse, marker="o", label="train_rmse")
            if any(v is not None for v in val_mae):
                plt.plot(epochs_list, val_mae, marker="o", label="val_mae")
            if any(v is not None for v in val_rmse):
                plt.plot(epochs_list, val_rmse, marker="o", label="val_rmse")
            plt.xlabel("Epoch")
            plt.ylabel("Metric value")
            plt.title("MAE / RMSE curve")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            metric_fig = log_dir / "mae_rmse_curve.png"
            plt.savefig(metric_fig, dpi=200)
            plt.close()
            saved_figures.append(str(metric_fig))

    if step_csv.exists():
        rows = []
        with open(step_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            global_steps = [int(row["global_step"]) for row in rows]
            losses = [safe_float(row.get("loss")) for row in rows]

            plt.figure()
            plt.plot(global_steps, losses, marker="o", label="step_loss")
            plt.xlabel("Global step")
            plt.ylabel("Loss")
            plt.title("Step loss curve")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            step_fig = log_dir / "step_loss_curve.png"
            plt.savefig(step_fig, dpi=200)
            plt.close()
            saved_figures.append(str(step_fig))

    return saved_figures


# =======================
# 训练工具函数
# =======================
def set_qwen_eval_if_frozen(model):
    if getattr(model, "freeze_qwen", False) and hasattr(model, "qwen") and model.qwen is not None:
        model.qwen.eval()


def clone_inputs_without_text(inputs):
    """
    用于图-only 对比实验。
    不深拷贝 DGL graph，避免额外显存开销，只复制 dict 外壳。
    """
    new_inputs = dict(inputs)
    new_inputs["text"] = None
    return new_inputs


def compute_metrics(pred, target):
    """
    pred, target: Tensor [N, 1] 或 [N]
    """
    pred = pred.detach()
    target = target.detach()

    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()

    return mae, rmse


def check_qwen_freeze(model, logger):
    log_section(logger, "Check Qwen freeze")

    if not hasattr(model, "qwen") or model.qwen is None:
        logger.info("model.qwen does not exist.")
        return

    qwen_total = sum(p.numel() for p in model.qwen.parameters())
    qwen_trainable = sum(p.numel() for p in model.qwen.parameters() if p.requires_grad)

    logger.info("qwen total params: %s", qwen_total)
    logger.info("qwen trainable params: %s", qwen_trainable)
    logger.info("qwen training mode: %s", model.qwen.training)

    first_param = next(model.qwen.parameters())
    logger.info("first qwen param requires_grad: %s", first_param.requires_grad)
    logger.info("first qwen param grad is None: %s", first_param.grad is None)


@torch.no_grad()
def evaluate(model, loader, device, use_text=True):
    model.eval()
    set_qwen_eval_if_frozen(model)

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    criterion = torch.nn.MSELoss(reduction="sum")

    for inputs, targets in loader:
        inputs = to_device(inputs, device)
        targets = targets.to(device, non_blocking=True)

        if not use_text:
            inputs = clone_inputs_without_text(inputs)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_n = targets.shape[0]
        total_loss += loss.item()
        total_samples += batch_n

        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    mse = total_loss / max(total_samples, 1)
    mae, rmse = compute_metrics(all_preds, all_targets)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    use_text=True,
    logger=None,
    step_writer=None,
    step_file=None,
    log_interval=50,
    start_global_step=0,
):
    model.train()
    set_qwen_eval_if_frozen(model)

    criterion = torch.nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    global_step = start_global_step

    for step, (inputs, targets) in enumerate(loader):
        global_step += 1

        inputs = to_device(inputs, device)
        targets = targets.to(device, non_blocking=True)

        if not use_text:
            inputs = clone_inputs_without_text(inputs)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Loss is invalid at epoch {epoch}, step {step}: {loss.item()}")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=5.0,
        )

        optimizer.step()

        batch_n = targets.shape[0]
        total_loss += loss.item() * batch_n
        total_samples += batch_n

        all_preds.append(outputs.detach())
        all_targets.append(targets.detach())

        current_lr = optimizer.param_groups[0]["lr"]
        if step_writer is not None:
            step_writer.writerow({
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "loss": loss.item(),
                "lr": current_lr,
            })
            if step_file is not None:
                step_file.flush()

        if logger is not None and step % log_interval == 0:
            logger.info("epoch %03d | step %04d | global_step %06d | loss %.6f | lr %.6g",
                        epoch, step, global_step, loss.item(), current_lr)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / max(total_samples, 1)
    mae, rmse = compute_metrics(all_preds, all_targets)

    return {
        "mse": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "last_global_step": global_step,
    }


# =======================
# 主流程
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="训练时：指定日志目录；不指定则自动创建。--plot-only 时：指定已有日志目录。",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="不重新训练，只读取已有 CSV 日志并重新生成可视化图片。",
    )
    args = parser.parse_args()

    if args.plot_only:
        if args.log_dir is None:
            raise ValueError("使用 --plot-only 时必须指定 --log-dir")
        figures = plot_metrics_from_csv(args.log_dir)
        print("已生成图像：")
        for fig in figures:
            print(fig)
        return

    if args.log_dir is None:
        run_dir = make_run_dir(base_log_dir)
    else:
        run_dir = Path(args.log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("log_dir: %s", run_dir)

    save_config(run_dir, {
        "data_path": data_path,
        "target_name": target_name,
        "qwen_model_name": qwen_model_name,
        "device": str(device),
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "use_text": use_text,
        "log_interval": log_interval,
    })

    step_file, step_writer = create_csv_writer(
        run_dir / "step_metrics.csv",
        fieldnames=["epoch", "step", "global_step", "loss", "lr"],
    )
    epoch_file, epoch_writer = create_csv_writer(
        run_dir / "epoch_metrics.csv",
        fieldnames=[
            "epoch",
            "train_mse",
            "train_mae",
            "train_rmse",
            "val_mse",
            "val_mae",
            "val_rmse",
        ],
    )

    try:
        log_section(logger, "Device info")
        logger.info("selected device: %s", device)
        logger.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("current cuda device: %s", torch.cuda.current_device())
            logger.info("cuda device name: %s", torch.cuda.get_device_name(0))

        log_section(logger, "Build dataset")

        transform = Compose([
            AddNodeFeature(),
            AddEdgeFeature(),
            AddAngleFeature(),
            GraphCollecting(["graph", "line_graph"], [target_name]),
        ])

        train_dataset, val_dataset, _ = get_dataset(
            data_path=data_path,
            ratio_train_val_test=None,
            n_train_val_test=[n_train, n_val, n_test],
            transforms=transform,
            graph=True,
            line_graph=True,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collect(),
        )

        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=val_dataset.collect(),
            )
        else:
            val_loader = None

        logger.info("n_train: %s", len(train_dataset))
        logger.info("n_val: %s", 0 if val_dataset is None else len(val_dataset))
        logger.info("batch_size: %s", batch_size)
        logger.info("use_text: %s", use_text)

        log_section(logger, "Qwen config")

        config = AutoConfig.from_pretrained(
            qwen_model_name,
            local_files_only=True,
            trust_remote_code=True,
        )

        qwen_hidden_size = config.hidden_size

        logger.info("qwen model: %s", qwen_model_name)
        logger.info("qwen hidden_size: %s", qwen_hidden_size)

        log_section(logger, "Build model")

        model = TextGuidedFusiformer(
            targets=[target_name],
            depth=4,
            edge_input_dim=80,
            triplet_input_dim=40,
            embed_dim=128,
            num_heads=4,
            text_dim=qwen_hidden_size,
            qwen_model_name=qwen_model_name,
            freeze_qwen=True,
        )

        model = model.to(device)

        check_qwen_freeze(model, logger)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        log_section(logger, "Start training")

        global_step = 0
        for epoch in range(1, epochs + 1):
            log_section(logger, f"Epoch {epoch}/{epochs}")

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                use_text=use_text,
                logger=logger,
                step_writer=step_writer,
                step_file=step_file,
                log_interval=log_interval,
                start_global_step=global_step,
            )
            global_step = train_metrics["last_global_step"]

            if val_loader is not None:
                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    use_text=use_text,
                )
            else:
                val_metrics = None

            epoch_row = {
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_mse": "" if val_metrics is None else val_metrics["mse"],
                "val_mae": "" if val_metrics is None else val_metrics["mae"],
                "val_rmse": "" if val_metrics is None else val_metrics["rmse"],
            }
            epoch_writer.writerow(epoch_row)
            epoch_file.flush()

            if val_metrics is not None:
                logger.info(
                    "Epoch %03d | train MSE %.6f | train MAE %.6f | train RMSE %.6f | "
                    "val MSE %.6f | val MAE %.6f | val RMSE %.6f",
                    epoch,
                    train_metrics["mse"],
                    train_metrics["mae"],
                    train_metrics["rmse"],
                    val_metrics["mse"],
                    val_metrics["mae"],
                    val_metrics["rmse"],
                )
            else:
                logger.info(
                    "Epoch %03d | train MSE %.6f | train MAE %.6f | train RMSE %.6f",
                    epoch,
                    train_metrics["mse"],
                    train_metrics["mae"],
                    train_metrics["rmse"],
                )

        figures = plot_metrics_from_csv(run_dir)
        for fig in figures:
            logger.info("saved figure: %s", fig)

        logger.info("PASS: dataset effect test completed.")
        logger.info("training log: %s", run_dir / "train.log")
        logger.info("epoch metrics csv: %s", run_dir / "epoch_metrics.csv")
        logger.info("step metrics csv: %s", run_dir / "step_metrics.csv")

    finally:
        step_file.close()
        epoch_file.close()


if __name__ == "__main__":
    main()
