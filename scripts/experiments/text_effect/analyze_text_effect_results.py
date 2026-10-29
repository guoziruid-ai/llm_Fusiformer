#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusiformer 文本作用实验结果汇总与可视化
==========================================

用途
----
自动扫描以下目录中的文本作用实验结果：

    E:\\Python项目\\llm_Fusiformer\\Run_log\\20260722
        ├── normal_seed0
        ├── bypass_seed0
        ├── shuffle_seed0
        └── ...

读取各实验目录中的：

    config.json
    final_result.json
    best_result.json
    epoch_metrics.csv
    train.log
    text_effect_test/text_condition_metrics.json
    text_effect_test/text_condition_samples.csv

然后计算并输出：

1. normal / bypass / shuffle 三种训练条件的测试集 MSE、MAE、RMSE；
2. 文本训练收益：bypass-trained 与 normal-trained 的性能差异；
3. 文本语义训练收益：shuffle-trained 与 normal-trained 的性能差异；
4. normal 训练模型的推理期文本消融收益；
5. normal 训练模型的正确文本与错配文本语义收益；
6. Benefit Rate、Harm Rate、Neutral Rate；
7. Prediction Shift、Text Dependency Index；
8. 训练/验证曲线、泛化差距、最佳 epoch、训练耗时；
9. 数据完整性和实验公平性警告；
10. 单样本最受益和最受损样本。

所有结果保存到：

    E:\\Python项目\\llm_Fusiformer\\Run_log\\20260722\\result

使用方法
--------
1. 检查下方“用户可编辑配置”中的 BASE_DIR。
2. 在当前 Python 环境安装依赖：

       pip install pandas numpy matplotlib

3. 直接运行：

       python analyze_text_effect_results.py

本程序不需要命令行参数。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "缺少依赖，请先执行：pip install pandas numpy matplotlib"
    ) from exc


# =============================================================================
# 用户可编辑配置
# =============================================================================

# 图片中显示的实验日期目录。
BASE_DIR = Path(r"../Run_log/20260725/Wiki")

# 所有汇总结果写入该目录。
RESULT_DIR = BASE_DIR / "result"

# 是否在每次运行前清空旧 result 目录。
# True：避免旧图片和旧 CSV 混入本次分析。
CLEAR_OLD_RESULT = True

# 实验目录名称中允许出现的训练条件。
CONDITION_ORDER = ["normal", "bypass", "shuffle"]

# 单样本排名输出数量。
TOP_N_SAMPLES = 20

# 当存在多个 normal seed 时，用于生成详细单样本图的优先 seed。
# 找不到该 seed 时，会自动选择 seed 最小的 normal 实验。
PRIMARY_NORMAL_SEED = 0

# 图像参数。
FIGURE_DPI = 220

# 数值稳定项。
EPSILON = 1e-12

# 实验公平性检查的浮点容差。
CONFIG_FLOAT_TOLERANCE = 1e-12

# 需要在 normal/bypass/shuffle 之间保持一致的配置字段。
FAIRNESS_CONFIG_KEYS = [
    "data_path",
    "target",
    "split_ratio",
    "split_counts",
    "depth",
    "edge_input_dim",
    "triplet_input_dim",
    "embed_dim",
    "num_heads",
    "freeze_qwen",
    "batch_size",
    "epochs",
    "actual_lr",
    "min_lr",
    "warmup_epochs",
    "weight_decay",
    "betas",
    "clip_grad",
    "amp",
    "drop_last",
    "monitor_requested",
]


# =============================================================================
# 通用工具
# =============================================================================


def configure_matplotlib() -> None:
    """配置 Matplotlib，优先使用 Windows 中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.autolayout"] = True


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"[WARN] 无法读取 JSON：{path} | {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        print(f"[WARN] 无法读取 CSV：{path} | {exc}")
        return pd.DataFrame()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 方便 Windows Excel 直接打开中文。
    df.to_csv(path, index=False, encoding="utf-8-sig")


def nested_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def finite_or_none(value: Any) -> Optional[float]:
    result = to_float(value)
    return result if math.isfinite(result) else None


def safe_divide(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return float("nan")
    if abs(denominator) <= EPSILON:
        return float("nan")
    return numerator / denominator


def format_number(value: Any, digits: int = 6) -> str:
    number = to_float(value)
    return "N/A" if not math.isfinite(number) else f"{number:.{digits}f}"


def format_percent(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    return "N/A" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def parse_seed(name: str, config: dict[str, Any]) -> Optional[int]:
    config_seed = config.get("seed")
    try:
        return int(config_seed)
    except (TypeError, ValueError):
        pass

    match = re.search(r"seed[_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_condition(name: str, config: dict[str, Any], final_result: dict[str, Any]) -> str:
    for source in (
        final_result.get("text_train_condition"),
        config.get("text_train_condition"),
    ):
        if isinstance(source, str) and source.strip().lower() in CONDITION_ORDER:
            return source.strip().lower()

    lower_name = name.lower()
    for condition in CONDITION_ORDER:
        if condition in lower_name:
            return condition
    return "unknown"


def parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def comparable_value(value: Any) -> Any:
    """把嵌套配置转换为可稳定比较的形式。"""
    if isinstance(value, dict):
        return tuple(sorted((k, comparable_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(comparable_value(v) for v in value)
    if isinstance(value, float):
        return round(value, 14)
    return value


def normal_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean_value = float(values.mean())
    if values.size == 1:
        return mean_value, mean_value
    standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
    margin = 1.96 * standard_error
    return mean_value - margin, mean_value + margin


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def prepare_result_dir() -> None:
    if RESULT_DIR.resolve() == BASE_DIR.resolve():
        raise RuntimeError("RESULT_DIR 不能与 BASE_DIR 相同。")
    if CLEAR_OLD_RESULT and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "sample_analysis").mkdir(parents=True, exist_ok=True)


# =============================================================================
# 实验目录发现与读取
# =============================================================================


@dataclass
class RunRecord:
    run_dir: Path
    run_name: str
    condition: str
    seed: Optional[int]
    modified_time: float
    finished_at: Optional[datetime]
    config: dict[str, Any]
    final_result: dict[str, Any]
    test_effect: dict[str, Any]
    validation_effect: dict[str, Any]
    epoch_metrics: pd.DataFrame
    sample_metrics: pd.DataFrame
    warnings: list[str]


def looks_like_run_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"result", ".ipynb_checkpoints"}:
        return False
    marker_paths = [
        path / "config.json",
        path / "final_result.json",
        path / "epoch_metrics.csv",
        path / "text_effect_test" / "text_condition_metrics.json",
    ]
    return any(marker.exists() for marker in marker_paths)


def discover_run_directories(base_dir: Path) -> list[Path]:
    """
    优先扫描 BASE_DIR 的直接子目录。
    同时兼容 condition_seed0_时间戳/ 这一类嵌套或带时间戳目录。
    """
    direct_runs = [path for path in base_dir.iterdir() if looks_like_run_directory(path)]
    if direct_runs:
        return sorted(direct_runs)

    nested_runs: list[Path] = []
    for path in base_dir.rglob("final_result.json"):
        parent = path.parent
        if RESULT_DIR in parent.parents or parent == RESULT_DIR:
            continue
        if looks_like_run_directory(parent):
            nested_runs.append(parent)
    return sorted(set(nested_runs))


def load_effect_json(run_dir: Path, split: str) -> dict[str, Any]:
    directory_name = "text_effect_test" if split == "test" else "text_effect_validation"
    return safe_read_json(run_dir / directory_name / "text_condition_metrics.json")


def load_run_record(run_dir: Path) -> RunRecord:
    config = safe_read_json(run_dir / "config.json")
    final_result = safe_read_json(run_dir / "final_result.json")

    test_effect = final_result.get("test_text_effect")
    if not isinstance(test_effect, dict) or not test_effect:
        test_effect = load_effect_json(run_dir, "test")

    validation_effect = final_result.get("validation_text_effect")
    if not isinstance(validation_effect, dict) or not validation_effect:
        validation_effect = load_effect_json(run_dir, "validation")

    epoch_metrics = safe_read_csv(run_dir / "epoch_metrics.csv")
    sample_metrics = safe_read_csv(
        run_dir / "text_effect_test" / "text_condition_samples.csv"
    )

    condition = infer_condition(run_dir.name, config, final_result)
    seed = parse_seed(run_dir.name, config)
    finished_at = parse_datetime(final_result.get("finished_at"))
    warnings: list[str] = []

    required_files = [
        run_dir / "config.json",
        run_dir / "final_result.json",
        run_dir / "text_effect_test" / "text_condition_metrics.json",
        run_dir / "text_effect_test" / "text_condition_samples.csv",
    ]
    for required in required_files:
        if not required.is_file():
            warnings.append(f"缺少文件：{required.relative_to(run_dir)}")

    if condition == "unknown":
        warnings.append("无法识别 text_train_condition")
    if seed is None:
        warnings.append("无法识别随机种子")

    return RunRecord(
        run_dir=run_dir,
        run_name=run_dir.name,
        condition=condition,
        seed=seed,
        modified_time=run_dir.stat().st_mtime,
        finished_at=finished_at,
        config=config,
        final_result=final_result,
        test_effect=test_effect,
        validation_effect=validation_effect,
        epoch_metrics=epoch_metrics,
        sample_metrics=sample_metrics,
        warnings=warnings,
    )


def extract_condition_metrics(effect: dict[str, Any], condition: str) -> dict[str, float]:
    metrics = nested_get(effect, "conditions", condition, default={})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "mse": to_float(metrics.get("mse")),
        "mae": to_float(metrics.get("mae")),
        "rmse": to_float(metrics.get("rmse")),
        "shuffle_singleton_batches": to_float(
            metrics.get("shuffle_singleton_batches")
        ),
    }


def extract_training_condition_test(record: RunRecord) -> dict[str, float]:
    metrics = record.final_result.get("test_at_training_condition")
    if isinstance(metrics, dict) and metrics:
        result = {
            "mse": to_float(metrics.get("mse")),
            "mae": to_float(metrics.get("mae")),
            "rmse": to_float(metrics.get("rmse")),
        }
        if any(math.isfinite(value) for value in result.values()):
            return result
    return extract_condition_metrics(record.test_effect, record.condition)


def extract_training_condition_validation(record: RunRecord) -> dict[str, float]:
    metrics = record.final_result.get("validation_at_training_condition")
    if isinstance(metrics, dict) and metrics:
        result = {
            "mse": to_float(metrics.get("mse")),
            "mae": to_float(metrics.get("mae")),
            "rmse": to_float(metrics.get("rmse")),
        }
        if any(math.isfinite(value) for value in result.values()):
            return result
    return extract_condition_metrics(record.validation_effect, record.condition)


def parse_gradient_check(run_dir: Path) -> dict[str, float]:
    log_path = run_dir / "train.log"
    result = {
        "gradient_parameters": float("nan"),
        "gradient_not_none": float("nan"),
        "gradient_nonzero": float("nan"),
        "gradient_summed_norm": float("nan"),
    }
    if not log_path.is_file():
        return result

    pattern = re.compile(
        r"text gradient check\s*\|\s*condition=(?P<condition>\w+)\s*\|\s*"
        r"parameters=(?P<parameters>\d+)\s*\|\s*"
        r"grad_not_none=(?P<not_none>\d+)\s*\|\s*"
        r"grad_nonzero=(?P<nonzero>\d+)\s*\|\s*"
        r"summed_grad_norm=(?P<norm>[+\-0-9.eE]+)"
    )

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return result

    matches = list(pattern.finditer(text))
    if not matches:
        return result
    match = matches[-1]
    result.update(
        {
            "gradient_parameters": float(match.group("parameters")),
            "gradient_not_none": float(match.group("not_none")),
            "gradient_nonzero": float(match.group("nonzero")),
            "gradient_summed_norm": float(match.group("norm")),
        }
    )
    return result


def epoch_summary(record: RunRecord) -> dict[str, float]:
    df = record.epoch_metrics.copy()
    result = {
        "epochs_recorded": float("nan"),
        "final_train_mae": float("nan"),
        "final_val_mae": float("nan"),
        "min_val_mae": float("nan"),
        "min_val_mae_epoch": float("nan"),
        "final_train_rmse": float("nan"),
        "final_val_rmse": float("nan"),
        "total_epoch_seconds": float("nan"),
    }
    if df.empty:
        return result

    numeric_columns = [
        "epoch",
        "train_mae",
        "val_mae",
        "train_rmse",
        "val_rmse",
        "epoch_seconds",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    result["epochs_recorded"] = float(len(df))
    last = df.iloc[-1]
    for source, target in [
        ("train_mae", "final_train_mae"),
        ("val_mae", "final_val_mae"),
        ("train_rmse", "final_train_rmse"),
        ("val_rmse", "final_val_rmse"),
    ]:
        if source in df.columns:
            result[target] = to_float(last[source])

    if "val_mae" in df.columns and df["val_mae"].notna().any():
        best_index = df["val_mae"].idxmin()
        result["min_val_mae"] = to_float(df.loc[best_index, "val_mae"])
        result["min_val_mae_epoch"] = to_float(df.loc[best_index, "epoch"])

    if "epoch_seconds" in df.columns:
        result["total_epoch_seconds"] = to_float(df["epoch_seconds"].sum())

    return result


def run_record_to_row(record: RunRecord) -> dict[str, Any]:
    test_training = extract_training_condition_test(record)
    val_training = extract_training_condition_validation(record)
    normal_test = extract_condition_metrics(record.test_effect, "normal")
    bypass_test = extract_condition_metrics(record.test_effect, "bypass")
    shuffle_test = extract_condition_metrics(record.test_effect, "shuffle")
    epoch_info = epoch_summary(record)
    gradient_info = parse_gradient_check(record.run_dir)

    normal_vs_bypass = record.test_effect.get("normal_vs_bypass", {})
    if not isinstance(normal_vs_bypass, dict):
        normal_vs_bypass = {}
    normal_vs_shuffle = record.test_effect.get("normal_vs_shuffle", {})
    if not isinstance(normal_vs_shuffle, dict):
        normal_vs_shuffle = {}

    training_seconds = to_float(record.final_result.get("training_seconds"))
    if not math.isfinite(training_seconds):
        training_seconds = epoch_info["total_epoch_seconds"]

    row = {
        "run_name": record.run_name,
        "run_dir": str(record.run_dir),
        "condition": record.condition,
        "seed": record.seed,
        "finished_at": record.finished_at.isoformat() if record.finished_at else "",
        "best_epoch": record.final_result.get("best_epoch"),
        "monitor": record.final_result.get("monitor", ""),
        "best_metric": to_float(record.final_result.get("best_metric")),
        "training_seconds": training_seconds,
        "training_hours": safe_divide(training_seconds, 3600.0),
        "test_mse_at_training_condition": test_training["mse"],
        "test_mae_at_training_condition": test_training["mae"],
        "test_rmse_at_training_condition": test_training["rmse"],
        "validation_mse_at_training_condition": val_training["mse"],
        "validation_mae_at_training_condition": val_training["mae"],
        "validation_rmse_at_training_condition": val_training["rmse"],
        "test_generalization_gap_mae": (
            test_training["mae"] - val_training["mae"]
            if math.isfinite(test_training["mae"]) and math.isfinite(val_training["mae"])
            else float("nan")
        ),
        "test_normal_mse": normal_test["mse"],
        "test_normal_mae": normal_test["mae"],
        "test_normal_rmse": normal_test["rmse"],
        "test_bypass_mse": bypass_test["mse"],
        "test_bypass_mae": bypass_test["mae"],
        "test_bypass_rmse": bypass_test["rmse"],
        "test_shuffle_mse": shuffle_test["mse"],
        "test_shuffle_mae": shuffle_test["mae"],
        "test_shuffle_rmse": shuffle_test["rmse"],
        "shuffle_singleton_batches": shuffle_test["shuffle_singleton_batches"],
        "inference_absolute_text_gain": to_float(
            normal_vs_bypass.get("absolute_mae_gain")
        ),
        "inference_relative_text_gain": to_float(
            normal_vs_bypass.get("relative_text_gain")
        ),
        "mean_prediction_shift": to_float(
            normal_vs_bypass.get("mean_prediction_shift")
        ),
        "median_prediction_shift": to_float(
            normal_vs_bypass.get("median_prediction_shift")
        ),
        "mean_text_dependency_index": to_float(
            normal_vs_bypass.get("mean_text_dependency_index")
        ),
        "benefit_rate": to_float(normal_vs_bypass.get("benefit_rate")),
        "harm_rate": to_float(normal_vs_bypass.get("harm_rate")),
        "neutral_rate": to_float(normal_vs_bypass.get("neutral_rate")),
        "semantic_absolute_mae_gain": to_float(
            normal_vs_shuffle.get("absolute_semantic_mae_gain")
        ),
        "semantic_relative_gain": to_float(
            normal_vs_shuffle.get("relative_semantic_gain")
        ),
        "semantic_mean_prediction_shift": to_float(
            normal_vs_shuffle.get("mean_prediction_shift")
        ),
        "config_seed": record.config.get("seed"),
        "config_batch_size": record.config.get("batch_size"),
        "config_epochs": record.config.get("epochs"),
        "config_actual_lr": record.config.get("actual_lr"),
        "config_amp": record.config.get("amp"),
        "config_target": record.config.get("target"),
        "config_data_path": record.config.get("data_path"),
        "warning_count": len(record.warnings),
        "warnings": "；".join(record.warnings),
        **epoch_info,
        **gradient_info,
    }
    return row


# =============================================================================
# 选择有效实验和公平性检查
# =============================================================================


def select_latest_per_condition_seed(records: list[RunRecord]) -> list[RunRecord]:
    """同一 condition+seed 出现多个目录时，选择完成时间/修改时间最新者。"""
    grouped: dict[tuple[str, Optional[int]], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.condition, record.seed), []).append(record)

    selected: list[RunRecord] = []
    for group_records in grouped.values():
        group_records.sort(
            key=lambda item: (
                item.finished_at.timestamp() if item.finished_at else item.modified_time,
                item.modified_time,
            ),
            reverse=True,
        )
        selected.append(group_records[0])
    return sorted(
        selected,
        key=lambda item: (
            item.seed if item.seed is not None else 10**9,
            CONDITION_ORDER.index(item.condition)
            if item.condition in CONDITION_ORDER
            else 10**9,
        ),
    )


def check_fairness(records: list[RunRecord]) -> pd.DataFrame:
    """按 seed 检查三种训练条件的关键配置是否一致。"""
    rows: list[dict[str, Any]] = []
    seeds = sorted({record.seed for record in records if record.seed is not None})

    for seed in seeds:
        seed_records = [record for record in records if record.seed == seed]
        by_condition = {record.condition: record for record in seed_records}
        reference = by_condition.get("normal") or (seed_records[0] if seed_records else None)
        if reference is None:
            continue

        missing_conditions = [
            condition for condition in CONDITION_ORDER if condition not in by_condition
        ]
        if missing_conditions:
            rows.append(
                {
                    "seed": seed,
                    "check_type": "missing_condition",
                    "config_key": "",
                    "reference_condition": reference.condition,
                    "other_condition": ",".join(missing_conditions),
                    "reference_value": "",
                    "other_value": "",
                    "status": "WARN",
                    "message": f"缺少训练条件：{missing_conditions}",
                }
            )

        for other_condition, other in by_condition.items():
            if other.run_dir == reference.run_dir:
                continue
            for key in FAIRNESS_CONFIG_KEYS:
                reference_value = reference.config.get(key)
                other_value = other.config.get(key)
                if comparable_value(reference_value) != comparable_value(other_value):
                    rows.append(
                        {
                            "seed": seed,
                            "check_type": "config_mismatch",
                            "config_key": key,
                            "reference_condition": reference.condition,
                            "other_condition": other_condition,
                            "reference_value": json.dumps(
                                reference_value, ensure_ascii=False
                            ),
                            "other_value": json.dumps(other_value, ensure_ascii=False),
                            "status": "WARN",
                            "message": "关键配置不一致，可能影响公平比较",
                        }
                    )

    if not rows:
        rows.append(
            {
                "seed": "all",
                "check_type": "fairness",
                "config_key": "",
                "reference_condition": "",
                "other_condition": "",
                "reference_value": "",
                "other_value": "",
                "status": "PASS",
                "message": "已发现的同 seed 实验关键配置一致",
            }
        )
    return pd.DataFrame(rows)


def build_quality_issues(records: list[RunRecord], run_df: pd.DataFrame) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []

    for record in records:
        for warning in record.warnings:
            issues.append(
                {
                    "severity": "WARN",
                    "run_name": record.run_name,
                    "issue": warning,
                }
            )

    for _, row in run_df.iterrows():
        run_name = str(row["run_name"])
        condition = str(row["condition"])

        singleton = to_float(row.get("shuffle_singleton_batches"))
        if math.isfinite(singleton) and singleton > 0:
            issues.append(
                {
                    "severity": "WARN",
                    "run_name": run_name,
                    "issue": (
                        f"shuffle 测试包含 {int(singleton)} 个 batch size=1，"
                        "这些 batch 无法真正错配文本"
                    ),
                }
            )

        rate_values = [
            to_float(row.get("benefit_rate")),
            to_float(row.get("harm_rate")),
            to_float(row.get("neutral_rate")),
        ]
        if all(math.isfinite(value) for value in rate_values):
            if abs(sum(rate_values) - 1.0) > 1e-5:
                issues.append(
                    {
                        "severity": "WARN",
                        "run_name": run_name,
                        "issue": "benefit/harm/neutral 比例之和不等于 1",
                    }
                )

        gradient_nonzero = to_float(row.get("gradient_nonzero"))
        if condition == "bypass" and math.isfinite(gradient_nonzero):
            if gradient_nonzero != 0:
                issues.append(
                    {
                        "severity": "ERROR",
                        "run_name": run_name,
                        "issue": (
                            f"bypass 实验文本相关参数存在 {int(gradient_nonzero)} 个非零梯度"
                        ),
                    }
                )
        if condition in {"normal", "shuffle"} and math.isfinite(gradient_nonzero):
            if gradient_nonzero == 0:
                issues.append(
                    {
                        "severity": "WARN",
                        "run_name": run_name,
                        "issue": f"{condition} 实验文本相关参数非零梯度数量为 0",
                    }
                )

        test_mae = to_float(row.get("test_mae_at_training_condition"))
        if not math.isfinite(test_mae):
            issues.append(
                {
                    "severity": "ERROR",
                    "run_name": run_name,
                    "issue": "缺少训练条件对应的测试 MAE",
                }
            )

    seed_counts = (
        run_df[run_df["condition"].isin(CONDITION_ORDER)]
        .groupby("condition")["seed"]
        .nunique(dropna=True)
    )
    for condition in CONDITION_ORDER:
        count = int(seed_counts.get(condition, 0))
        if count < 3:
            issues.append(
                {
                    "severity": "INFO",
                    "run_name": condition,
                    "issue": f"仅有 {count} 个随机种子；正式结论建议至少 3 个 seed",
                }
            )

    if not issues:
        issues.append(
            {
                "severity": "PASS",
                "run_name": "all",
                "issue": "未发现明显的数据完整性或实验配置问题",
            }
        )
    return pd.DataFrame(issues)


# =============================================================================
# 跨训练条件计算
# =============================================================================


def paired_seed_comparisons(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid_df = run_df[run_df["condition"].isin(CONDITION_ORDER)].copy()
    valid_df = valid_df[valid_df["seed"].notna()].copy()

    for seed, group in valid_df.groupby("seed"):
        by_condition = {
            row["condition"]: row
            for _, row in group.sort_values("finished_at").iterrows()
        }
        normal = by_condition.get("normal")
        bypass = by_condition.get("bypass")
        shuffle = by_condition.get("shuffle")

        row: dict[str, Any] = {"seed": int(seed)}

        if normal is not None:
            row["normal_test_mae"] = to_float(
                normal["test_mae_at_training_condition"]
            )
            row["normal_test_rmse"] = to_float(
                normal["test_rmse_at_training_condition"]
            )
        if bypass is not None:
            row["bypass_test_mae"] = to_float(
                bypass["test_mae_at_training_condition"]
            )
            row["bypass_test_rmse"] = to_float(
                bypass["test_rmse_at_training_condition"]
            )
        if shuffle is not None:
            row["shuffle_test_mae"] = to_float(
                shuffle["test_mae_at_training_condition"]
            )
            row["shuffle_test_rmse"] = to_float(
                shuffle["test_rmse_at_training_condition"]
            )

        normal_mae = to_float(row.get("normal_test_mae"))
        bypass_mae = to_float(row.get("bypass_test_mae"))
        shuffle_mae = to_float(row.get("shuffle_test_mae"))
        normal_rmse = to_float(row.get("normal_test_rmse"))
        bypass_rmse = to_float(row.get("bypass_test_rmse"))
        shuffle_rmse = to_float(row.get("shuffle_test_rmse"))

        row["training_absolute_text_gain_mae"] = (
            bypass_mae - normal_mae
            if math.isfinite(bypass_mae) and math.isfinite(normal_mae)
            else float("nan")
        )
        row["training_relative_text_gain_mae"] = safe_divide(
            row["training_absolute_text_gain_mae"], bypass_mae
        )
        row["training_absolute_text_gain_rmse"] = (
            bypass_rmse - normal_rmse
            if math.isfinite(bypass_rmse) and math.isfinite(normal_rmse)
            else float("nan")
        )
        row["training_relative_text_gain_rmse"] = safe_divide(
            row["training_absolute_text_gain_rmse"], bypass_rmse
        )

        row["training_absolute_semantic_gain_mae"] = (
            shuffle_mae - normal_mae
            if math.isfinite(shuffle_mae) and math.isfinite(normal_mae)
            else float("nan")
        )
        row["training_relative_semantic_gain_mae"] = safe_divide(
            row["training_absolute_semantic_gain_mae"], shuffle_mae
        )
        row["training_absolute_semantic_gain_rmse"] = (
            shuffle_rmse - normal_rmse
            if math.isfinite(shuffle_rmse) and math.isfinite(normal_rmse)
            else float("nan")
        )
        row["training_relative_semantic_gain_rmse"] = safe_divide(
            row["training_absolute_semantic_gain_rmse"], shuffle_rmse
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()


def aggregate_condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        group = run_df[run_df["condition"] == condition]
        if group.empty:
            continue
        for metric in [
            "test_mse_at_training_condition",
            "test_mae_at_training_condition",
            "test_rmse_at_training_condition",
            "validation_mae_at_training_condition",
            "training_hours",
        ]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "count": int(values.size),
                    "mean": float(values.mean()) if not values.empty else float("nan"),
                    "std": (
                        float(values.std(ddof=1)) if values.size > 1 else 0.0
                    ),
                    "min": float(values.min()) if not values.empty else float("nan"),
                    "max": float(values.max()) if not values.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_pair_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in paired_df.columns
        if column != "seed" and pd.api.types.is_numeric_dtype(paired_df[column])
    ]
    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        values = pd.to_numeric(paired_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        ci_low, ci_high = normal_ci95(values.to_numpy())
        rows.append(
            {
                "metric": metric,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 单样本分析
# =============================================================================


def ensure_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_candidates = [
        "element_index",
        "target",
        "prediction_normal",
        "prediction_bypass",
        "prediction_shuffle",
        "absolute_error_normal",
        "absolute_error_bypass",
        "absolute_error_shuffle",
        "prediction_shift",
        "signed_prediction_shift",
        "mae_gain_from_text",
        "text_dependency_index",
        "semantic_prediction_shift",
        "mae_gain_correct_vs_shuffle",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "absolute_error_normal" not in df.columns and {
        "prediction_normal",
        "target",
    }.issubset(df.columns):
        df["absolute_error_normal"] = (
            df["prediction_normal"] - df["target"]
        ).abs()

    if "absolute_error_bypass" not in df.columns and {
        "prediction_bypass",
        "target",
    }.issubset(df.columns):
        df["absolute_error_bypass"] = (
            df["prediction_bypass"] - df["target"]
        ).abs()

    if "absolute_error_shuffle" not in df.columns and {
        "prediction_shuffle",
        "target",
    }.issubset(df.columns):
        df["absolute_error_shuffle"] = (
            df["prediction_shuffle"] - df["target"]
        ).abs()

    if "prediction_shift" not in df.columns and {
        "prediction_normal",
        "prediction_bypass",
    }.issubset(df.columns):
        df["prediction_shift"] = (
            df["prediction_normal"] - df["prediction_bypass"]
        ).abs()

    if "mae_gain_from_text" not in df.columns and {
        "absolute_error_normal",
        "absolute_error_bypass",
    }.issubset(df.columns):
        df["mae_gain_from_text"] = (
            df["absolute_error_bypass"] - df["absolute_error_normal"]
        )

    if "mae_gain_correct_vs_shuffle" not in df.columns and {
        "absolute_error_normal",
        "absolute_error_shuffle",
    }.issubset(df.columns):
        df["mae_gain_correct_vs_shuffle"] = (
            df["absolute_error_shuffle"] - df["absolute_error_normal"]
        )

    if "semantic_prediction_shift" not in df.columns and {
        "prediction_normal",
        "prediction_shuffle",
    }.issubset(df.columns):
        df["semantic_prediction_shift"] = (
            df["prediction_normal"] - df["prediction_shuffle"]
        ).abs()

    return df


def summarize_samples(record: RunRecord) -> dict[str, Any]:
    df = ensure_sample_columns(record.sample_metrics)
    summary: dict[str, Any] = {
        "run_name": record.run_name,
        "condition": record.condition,
        "seed": record.seed,
        "num_samples": int(len(df)),
    }
    if df.empty:
        return summary

    gain = pd.to_numeric(df.get("mae_gain_from_text"), errors="coerce").dropna()
    semantic_gain = pd.to_numeric(
        df.get("mae_gain_correct_vs_shuffle"), errors="coerce"
    ).dropna()
    shift = pd.to_numeric(df.get("prediction_shift"), errors="coerce").dropna()
    dependency = pd.to_numeric(
        df.get("text_dependency_index"), errors="coerce"
    ).dropna()

    threshold = to_float(record.test_effect.get("neutral_threshold"))
    if not math.isfinite(threshold):
        threshold = to_float(record.config.get("text_effect_neutral_threshold"))
    if not math.isfinite(threshold):
        threshold = 1e-6

    if not gain.empty:
        gain_array = gain.to_numpy(dtype=float)
        benefit_count = int(np.sum(gain_array > threshold))
        harm_count = int(np.sum(gain_array < -threshold))
        neutral_count = int(gain_array.size - benefit_count - harm_count)
        benefit_ci = wilson_interval(benefit_count, gain_array.size)
        gain_ci = normal_ci95(gain_array)
        gain_std = float(np.std(gain_array, ddof=1)) if gain_array.size > 1 else 0.0
        cohens_dz = safe_divide(float(np.mean(gain_array)), gain_std)

        summary.update(
            {
                "neutral_threshold": threshold,
                "mean_mae_gain_from_text": float(np.mean(gain_array)),
                "median_mae_gain_from_text": float(np.median(gain_array)),
                "std_mae_gain_from_text": gain_std,
                "mae_gain_ci95_low": gain_ci[0],
                "mae_gain_ci95_high": gain_ci[1],
                "cohens_dz_text_gain": cohens_dz,
                "benefit_count": benefit_count,
                "harm_count": harm_count,
                "neutral_count": neutral_count,
                "benefit_rate": benefit_count / gain_array.size,
                "harm_rate": harm_count / gain_array.size,
                "neutral_rate": neutral_count / gain_array.size,
                "benefit_rate_ci95_low": benefit_ci[0],
                "benefit_rate_ci95_high": benefit_ci[1],
            }
        )

    if not semantic_gain.empty:
        semantic_array = semantic_gain.to_numpy(dtype=float)
        semantic_ci = normal_ci95(semantic_array)
        summary.update(
            {
                "mean_semantic_mae_gain": float(np.mean(semantic_array)),
                "median_semantic_mae_gain": float(np.median(semantic_array)),
                "semantic_gain_ci95_low": semantic_ci[0],
                "semantic_gain_ci95_high": semantic_ci[1],
                "semantic_benefit_rate": float(np.mean(semantic_array > threshold)),
            }
        )

    if not shift.empty:
        summary.update(
            {
                "mean_prediction_shift": float(shift.mean()),
                "median_prediction_shift": float(shift.median()),
                "p90_prediction_shift": float(shift.quantile(0.90)),
                "p95_prediction_shift": float(shift.quantile(0.95)),
            }
        )

    if not dependency.empty:
        summary.update(
            {
                "mean_text_dependency_index": float(dependency.mean()),
                "median_text_dependency_index": float(dependency.median()),
                "p90_text_dependency_index": float(dependency.quantile(0.90)),
            }
        )

    if "prediction_shift" in df.columns and "mae_gain_from_text" in df.columns:
        summary["correlation_shift_vs_gain"] = pearson_correlation(
            df["prediction_shift"].to_numpy(dtype=float),
            df["mae_gain_from_text"].to_numpy(dtype=float),
        )

    if "text_dependency_index" in df.columns and "mae_gain_from_text" in df.columns:
        summary["correlation_dependency_vs_gain"] = pearson_correlation(
            df["text_dependency_index"].to_numpy(dtype=float),
            df["mae_gain_from_text"].to_numpy(dtype=float),
        )

    return summary


def choose_primary_normal_record(records: list[RunRecord]) -> Optional[RunRecord]:
    normal_records = [record for record in records if record.condition == "normal"]
    if not normal_records:
        return None

    exact = [record for record in normal_records if record.seed == PRIMARY_NORMAL_SEED]
    candidates = exact or normal_records
    candidates.sort(
        key=lambda item: (
            item.seed if item.seed is not None else 10**9,
            -(item.finished_at.timestamp() if item.finished_at else item.modified_time),
        )
    )
    return candidates[0]


def save_top_samples(record: RunRecord) -> tuple[Path, Path]:
    df = ensure_sample_columns(record.sample_metrics)
    output_dir = RESULT_DIR / "sample_analysis"
    benefit_path = output_dir / f"top_benefit_{record.run_name}.csv"
    harm_path = output_dir / f"top_harm_{record.run_name}.csv"

    if df.empty or "mae_gain_from_text" not in df.columns:
        save_csv(pd.DataFrame(), benefit_path)
        save_csv(pd.DataFrame(), harm_path)
        return benefit_path, harm_path

    sort_columns = [
        column
        for column in [
            "element_index",
            "target",
            "prediction_normal",
            "prediction_bypass",
            "prediction_shuffle",
            "absolute_error_normal",
            "absolute_error_bypass",
            "absolute_error_shuffle",
            "mae_gain_from_text",
            "mae_gain_correct_vs_shuffle",
            "prediction_shift",
            "text_dependency_index",
            "text_effect_class",
        ]
        if column in df.columns
    ]

    benefit = df.nlargest(TOP_N_SAMPLES, "mae_gain_from_text")[sort_columns]
    harm = df.nsmallest(TOP_N_SAMPLES, "mae_gain_from_text")[sort_columns]
    save_csv(benefit, benefit_path)
    save_csv(harm, harm_path)
    return benefit_path, harm_path


# =============================================================================
# 可视化
# =============================================================================


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_condition_performance(run_df: pd.DataFrame) -> Optional[Path]:
    rows = []
    for condition in CONDITION_ORDER:
        group = run_df[run_df["condition"] == condition]
        if group.empty:
            continue
        for metric in ["test_mae_at_training_condition", "test_rmse_at_training_condition"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "condition": condition,
                    "metric": "MAE" if "mae" in metric else "RMSE",
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
            )
    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    conditions = [condition for condition in CONDITION_ORDER if condition in plot_df["condition"].values]
    x = np.arange(len(conditions), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    for offset, metric in [(-width / 2, "MAE"), (width / 2, "RMSE")]:
        subset = plot_df[plot_df["metric"] == metric].set_index("condition")
        means = [to_float(subset.loc[c, "mean"]) if c in subset.index else np.nan for c in conditions]
        stds = [to_float(subset.loc[c, "std"]) if c in subset.index else 0.0 for c in conditions]
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=metric)
        ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Test error")
    ax.set_title("Training-condition test performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    path = RESULT_DIR / "figures" / "01_condition_performance.png"
    save_figure(path)
    return path


def plot_per_seed_mae(paired_df: pd.DataFrame) -> Optional[Path]:
    if paired_df.empty or "seed" not in paired_df.columns:
        return None

    available = [
        ("normal_test_mae", "normal"),
        ("bypass_test_mae", "bypass"),
        ("shuffle_test_mae", "shuffle"),
    ]
    if not any(column in paired_df.columns for column, _ in available):
        return None

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    seeds = pd.to_numeric(paired_df["seed"], errors="coerce")
    for column, label in available:
        if column not in paired_df.columns:
            continue
        values = pd.to_numeric(paired_df[column], errors="coerce")
        ax.plot(seeds, values, marker="o", label=label)

    ax.set_xlabel("Seed")
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE by random seed")
    ax.set_xticks(sorted(seeds.dropna().unique()))
    ax.grid(alpha=0.25)
    ax.legend()

    path = RESULT_DIR / "figures" / "02_per_seed_mae.png"
    save_figure(path)
    return path


def plot_training_curves(records: list[RunRecord]) -> list[Path]:
    paths: list[Path] = []

    # 总体图：各 run 的 validation MAE。
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    plotted = False
    for record in records:
        df = record.epoch_metrics.copy()
        if df.empty or "epoch" not in df.columns or "val_mae" not in df.columns:
            continue
        epoch = pd.to_numeric(df["epoch"], errors="coerce")
        val_mae = pd.to_numeric(df["val_mae"], errors="coerce")
        mask = epoch.notna() & val_mae.notna()
        if not mask.any():
            continue
        ax.plot(epoch[mask], val_mae[mask], label=f"{record.condition}-seed{record.seed}")
        plotted = True

    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE")
        ax.set_title("Validation MAE curves")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        path = RESULT_DIR / "figures" / "03_validation_mae_curves.png"
        save_figure(path)
        paths.append(path)
    else:
        plt.close(fig)

    # 每个 run 单独绘制 train/val MAE 与 RMSE，便于检查过拟合。
    curve_dir = RESULT_DIR / "figures" / "training_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        df = record.epoch_metrics.copy()
        if df.empty or "epoch" not in df.columns:
            continue
        for column in ["epoch", "train_mae", "val_mae", "train_rmse", "val_rmse"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        plotted_run = False
        for column, label in [
            ("train_mae", "train MAE"),
            ("val_mae", "validation MAE"),
            ("train_rmse", "train RMSE"),
            ("val_rmse", "validation RMSE"),
        ]:
            if column not in df.columns:
                continue
            mask = df["epoch"].notna() & df[column].notna()
            if mask.any():
                ax.plot(df.loc[mask, "epoch"], df.loc[mask, column], label=label)
                plotted_run = True

        if not plotted_run:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title(f"Training curves: {record.run_name}")
        ax.grid(alpha=0.25)
        ax.legend()
        path = curve_dir / f"{record.run_name}_curves.png"
        save_figure(path)
        paths.append(path)

    return paths


def plot_normal_counterfactual(record: RunRecord) -> Optional[Path]:
    effect = record.test_effect
    rows = []
    for condition in CONDITION_ORDER:
        metrics = extract_condition_metrics(effect, condition)
        if math.isfinite(metrics["mae"]):
            rows.append((condition, metrics["mae"], metrics["rmse"]))
    if not rows:
        return None

    labels = [row[0] for row in rows]
    mae_values = [row[1] for row in rows]
    rmse_values = [row[2] for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    bars_mae = ax.bar(x - width / 2, mae_values, width=width, label="MAE")
    bars_rmse = ax.bar(x + width / 2, rmse_values, width=width, label="RMSE")
    ax.bar_label(bars_mae, fmt="%.5f", padding=3, fontsize=8)
    ax.bar_label(bars_rmse, fmt="%.5f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test error")
    ax.set_title(f"Counterfactual inference: {record.run_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    path = RESULT_DIR / "figures" / "04_normal_model_counterfactual.png"
    save_figure(path)
    return path


def plot_sample_analysis(record: RunRecord) -> list[Path]:
    df = ensure_sample_columns(record.sample_metrics)
    if df.empty:
        return []

    paths: list[Path] = []
    figure_dir = RESULT_DIR / "figures" / "sample_analysis"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 1. normal vs bypass 单样本绝对误差散点图。
    if {"absolute_error_bypass", "absolute_error_normal"}.issubset(df.columns):
        x = df["absolute_error_bypass"].to_numpy(dtype=float)
        y = df["absolute_error_normal"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            limit = float(max(np.max(x[mask]), np.max(y[mask])))
            fig, ax = plt.subplots(figsize=(6.5, 6.2))
            ax.scatter(x[mask], y[mask], alpha=0.55, s=18)
            ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
            ax.set_xlabel("Absolute error: bypass")
            ax.set_ylabel("Absolute error: normal")
            ax.set_title("Per-sample error: normal vs bypass")
            ax.grid(alpha=0.2)
            path = figure_dir / "05_error_scatter_normal_vs_bypass.png"
            save_figure(path)
            paths.append(path)

    # 2. 文本收益分布。
    if "mae_gain_from_text" in df.columns:
        gain = pd.to_numeric(df["mae_gain_from_text"], errors="coerce").dropna()
        if not gain.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            ax.hist(gain, bins="auto")
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.axvline(float(gain.mean()), linestyle="-", linewidth=1.2, label="mean")
            ax.set_xlabel("MAE gain from text (bypass error - normal error)")
            ax.set_ylabel("Sample count")
            ax.set_title("Distribution of per-sample text gain")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            path = figure_dir / "06_text_gain_histogram.png"
            save_figure(path)
            paths.append(path)

    # 3. 文本依赖性与收益关系。
    if {"text_dependency_index", "mae_gain_from_text"}.issubset(df.columns):
        x = pd.to_numeric(df["text_dependency_index"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["mae_gain_from_text"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            correlation = pearson_correlation(x[mask], y[mask])
            fig, ax = plt.subplots(figsize=(7.3, 5.8))
            ax.scatter(x[mask], y[mask], alpha=0.5, s=18)
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xlabel("Text Dependency Index")
            ax.set_ylabel("MAE gain from text")
            ax.set_title(f"Dependency vs gain (Pearson r={correlation:.3f})")
            ax.grid(alpha=0.2)
            path = figure_dir / "07_dependency_vs_gain.png"
            save_figure(path)
            paths.append(path)

    # 4. 三种条件预测值与真实值散点。
    if "target" in df.columns:
        prediction_columns = [
            ("prediction_normal", "normal"),
            ("prediction_bypass", "bypass"),
            ("prediction_shuffle", "shuffle"),
        ]
        available = [(column, label) for column, label in prediction_columns if column in df.columns]
        if available:
            target = pd.to_numeric(df["target"], errors="coerce").to_numpy()
            finite_target = target[np.isfinite(target)]
            if finite_target.size:
                predictions = []
                for column, _ in available:
                    predictions.extend(
                        pd.to_numeric(df[column], errors="coerce").dropna().tolist()
                    )
                all_values = np.asarray(list(finite_target) + predictions, dtype=float)
                all_values = all_values[np.isfinite(all_values)]
                low, high = float(all_values.min()), float(all_values.max())

                fig, ax = plt.subplots(figsize=(7.0, 6.3))
                for column, label in available:
                    prediction = pd.to_numeric(df[column], errors="coerce").to_numpy()
                    mask = np.isfinite(target) & np.isfinite(prediction)
                    ax.scatter(target[mask], prediction[mask], alpha=0.45, s=18, label=label)
                ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
                ax.set_xlabel("Target")
                ax.set_ylabel("Prediction")
                ax.set_title("Prediction vs target")
                ax.grid(alpha=0.2)
                ax.legend()
                path = figure_dir / "08_prediction_vs_target.png"
                save_figure(path)
                paths.append(path)

    # 5. 文本作用样本分类比例。
    if "mae_gain_from_text" in df.columns:
        gain = pd.to_numeric(df["mae_gain_from_text"], errors="coerce").dropna()
        threshold = to_float(record.test_effect.get("neutral_threshold"))
        if not math.isfinite(threshold):
            threshold = 1e-6
        if not gain.empty:
            counts = [
                int((gain > threshold).sum()),
                int((gain < -threshold).sum()),
                int((gain.abs() <= threshold).sum()),
            ]
            labels = ["benefit", "harm", "neutral"]
            fig, ax = plt.subplots(figsize=(7.3, 5.2))
            bars = ax.bar(labels, counts)
            ax.bar_label(bars, labels=[f"{count}\n({count / len(gain):.1%})" for count in counts])
            ax.set_ylabel("Sample count")
            ax.set_title("Text-effect sample classes")
            ax.grid(axis="y", alpha=0.2)
            path = figure_dir / "09_text_effect_classes.png"
            save_figure(path)
            paths.append(path)

    # 6. 正确文本相对错配文本的语义收益分布。
    if "mae_gain_correct_vs_shuffle" in df.columns:
        semantic_gain = pd.to_numeric(
            df["mae_gain_correct_vs_shuffle"], errors="coerce"
        ).dropna()
        if not semantic_gain.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            ax.hist(semantic_gain, bins="auto")
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.axvline(
                float(semantic_gain.mean()), linestyle="-", linewidth=1.2, label="mean"
            )
            ax.set_xlabel("Semantic MAE gain (shuffle error - normal error)")
            ax.set_ylabel("Sample count")
            ax.set_title("Distribution of semantic text gain")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            path = figure_dir / "10_semantic_gain_histogram.png"
            save_figure(path)
            paths.append(path)

    return paths


# =============================================================================
# 报告生成
# =============================================================================


def get_selected_row(run_df: pd.DataFrame, condition: str, seed: Optional[int] = None) -> Optional[pd.Series]:
    subset = run_df[run_df["condition"] == condition]
    if seed is not None:
        exact = subset[subset["seed"] == seed]
        if not exact.empty:
            subset = exact
    if subset.empty:
        return None
    subset = subset.sort_values(["seed", "finished_at"], na_position="last")
    return subset.iloc[0]


def build_summary_payload(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    primary_normal: Optional[RunRecord],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "base_dir": str(BASE_DIR),
        "result_dir": str(RESULT_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "number_of_selected_runs": int(len(run_df)),
        "conditions_found": sorted(run_df["condition"].dropna().unique().tolist()),
        "seeds_found": sorted(
            int(seed) for seed in run_df["seed"].dropna().unique().tolist()
        ),
        "primary_normal_run": primary_normal.run_name if primary_normal else None,
        "runs": run_df.replace({np.nan: None}).to_dict(orient="records"),
        "paired_seed_comparisons": paired_df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "paired_metric_summary": pair_summary_df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
        "sample_level_summary": sample_summary_df.replace({np.nan: None}).to_dict(
            orient="records"
        ),
    }
    return payload


def metric_mean(pair_summary_df: pd.DataFrame, metric: str) -> float:
    if pair_summary_df.empty:
        return float("nan")
    subset = pair_summary_df[pair_summary_df["metric"] == metric]
    return to_float(subset.iloc[0]["mean"]) if not subset.empty else float("nan")


def make_markdown_report(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    primary_normal: Optional[RunRecord],
    figure_paths: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# Fusiformer 文本作用实验结果报告")
    lines.append("")
    lines.append(f"- 数据目录：`{BASE_DIR}`")
    lines.append(f"- 结果目录：`{RESULT_DIR}`")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 有效实验数：{len(run_df)}")
    lines.append("")

    lines.append("## 1. 三种训练条件的核心测试指标")
    lines.append("")
    lines.append("| 条件 | Seed | Test MAE | Test RMSE | Test MSE | Val MAE | Best epoch | 训练小时 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in run_df.sort_values(["seed", "condition"]).iterrows():
        lines.append(
            "| {condition} | {seed} | {mae} | {rmse} | {mse} | {val_mae} | {best_epoch} | {hours} |".format(
                condition=row["condition"],
                seed=row["seed"] if pd.notna(row["seed"]) else "N/A",
                mae=format_number(row["test_mae_at_training_condition"]),
                rmse=format_number(row["test_rmse_at_training_condition"]),
                mse=format_number(row["test_mse_at_training_condition"]),
                val_mae=format_number(row["validation_mae_at_training_condition"]),
                best_epoch=row["best_epoch"] if pd.notna(row["best_epoch"]) else "N/A",
                hours=format_number(row["training_hours"], 3),
            )
        )
    lines.append("")

    lines.append("## 2. 文本训练作用：跨训练模型比较")
    lines.append("")
    if paired_df.empty:
        lines.append("没有找到可按相同 seed 配对的 normal/bypass/shuffle 实验。")
    else:
        lines.append("定义：")
        lines.append("")
        lines.append("- 文本训练绝对收益 = `MAE(bypass-trained) - MAE(normal-trained)`")
        lines.append("- 文本训练相对收益 = `绝对收益 / MAE(bypass-trained)`")
        lines.append("- 语义训练收益 = `MAE(shuffle-trained) - MAE(normal-trained)`")
        lines.append("")
        lines.append("| Seed | normal MAE | bypass MAE | shuffle MAE | 文本训练绝对收益 | 文本训练相对收益 | 语义训练相对收益 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in paired_df.iterrows():
            lines.append(
                "| {seed} | {normal} | {bypass} | {shuffle} | {abs_gain} | {rel_gain} | {semantic_gain} |".format(
                    seed=int(row["seed"]),
                    normal=format_number(row.get("normal_test_mae")),
                    bypass=format_number(row.get("bypass_test_mae")),
                    shuffle=format_number(row.get("shuffle_test_mae")),
                    abs_gain=format_number(row.get("training_absolute_text_gain_mae")),
                    rel_gain=format_percent(row.get("training_relative_text_gain_mae")),
                    semantic_gain=format_percent(row.get("training_relative_semantic_gain_mae")),
                )
            )
        lines.append("")

        mean_text_gain = metric_mean(
            pair_summary_df, "training_relative_text_gain_mae"
        )
        mean_semantic_gain = metric_mean(
            pair_summary_df, "training_relative_semantic_gain_mae"
        )
        lines.append(
            f"- 平均文本训练相对收益：**{format_percent(mean_text_gain)}**。"
        )
        lines.append(
            f"- 平均语义训练相对收益：**{format_percent(mean_semantic_gain)}**。"
        )
        if math.isfinite(mean_text_gain):
            if mean_text_gain > 0:
                lines.append("- 解释：正确文本训练的平均 MAE 优于严格 graph-only 训练。")
            else:
                lines.append("- 注意：正确文本训练没有优于严格 graph-only 训练。")
        if math.isfinite(mean_semantic_gain):
            if mean_semantic_gain > 0:
                lines.append("- 解释：正确匹配文本训练优于错配文本训练，支持文本语义有效。")
            else:
                lines.append("- 注意：正确匹配文本训练没有优于错配文本训练。")
    lines.append("")

    lines.append("## 3. normal 模型的推理期文本消融")
    lines.append("")
    if primary_normal is None:
        lines.append("未找到 normal 训练模型。")
    else:
        primary_row_df = run_df[run_df["run_name"] == primary_normal.run_name]
        if primary_row_df.empty:
            lines.append("无法读取 primary normal 实验的汇总行。")
        else:
            row = primary_row_df.iloc[0]
            lines.append(f"主要分析实验：`{primary_normal.run_name}`")
            lines.append("")
            lines.append("| 推理条件 | Test MAE | Test RMSE |")
            lines.append("|---|---:|---:|")
            for condition in CONDITION_ORDER:
                lines.append(
                    f"| {condition} | {format_number(row.get(f'test_{condition}_mae'))} | "
                    f"{format_number(row.get(f'test_{condition}_rmse'))} |"
                )
            lines.append("")
            lines.append(
                f"- 同一模型移除文本后的相对文本收益：**{format_percent(row.get('inference_relative_text_gain'))}**。"
            )
            lines.append(
                f"- 正确文本相对错配文本的语义收益：**{format_percent(row.get('semantic_relative_gain'))}**。"
            )
            lines.append(
                f"- 平均预测偏移：**{format_number(row.get('mean_prediction_shift'))}**。"
            )
            lines.append(
                f"- 平均 Text Dependency Index：**{format_number(row.get('mean_text_dependency_index'))}**。"
            )
            lines.append(
                f"- Benefit / Harm / Neutral：**{format_percent(row.get('benefit_rate'))} / "
                f"{format_percent(row.get('harm_rate'))} / {format_percent(row.get('neutral_rate'))}**。"
            )
    lines.append("")

    lines.append("## 4. 单样本文本影响")
    lines.append("")
    if sample_summary_df.empty:
        lines.append("未找到可用的单样本结果 CSV。")
    else:
        normal_samples = sample_summary_df[sample_summary_df["condition"] == "normal"]
        if primary_normal is not None:
            exact = normal_samples[normal_samples["run_name"] == primary_normal.run_name]
            if not exact.empty:
                normal_samples = exact
        if normal_samples.empty:
            lines.append("没有 normal 训练模型的单样本结果。")
        else:
            sample = normal_samples.iloc[0]
            lines.append(f"- 样本数：{int(sample.get('num_samples', 0))}")
            lines.append(
                f"- 平均单样本 MAE 收益：**{format_number(sample.get('mean_mae_gain_from_text'))}**，"
                f"95% 近似区间 [{format_number(sample.get('mae_gain_ci95_low'))}, "
                f"{format_number(sample.get('mae_gain_ci95_high'))}]。"
            )
            lines.append(
                f"- Benefit Rate：**{format_percent(sample.get('benefit_rate'))}**，"
                f"95% Wilson 区间 [{format_percent(sample.get('benefit_rate_ci95_low'))}, "
                f"{format_percent(sample.get('benefit_rate_ci95_high'))}]。"
            )
            lines.append(
                f"- Harm Rate：**{format_percent(sample.get('harm_rate'))}**。"
            )
            lines.append(
                f"- Cohen's dz：**{format_number(sample.get('cohens_dz_text_gain'))}**。"
            )
            lines.append(
                f"- 文本依赖性与收益的 Pearson 相关：**{format_number(sample.get('correlation_dependency_vs_gain'))}**。"
            )
            lines.append(
                f"- 正确文本相对错配文本的平均单样本语义收益：**{format_number(sample.get('mean_semantic_mae_gain'))}**。"
            )
    lines.append("")

    lines.append("## 5. 实验公平性和数据质量")
    lines.append("")
    fairness_warnings = fairness_df[fairness_df["status"] == "WARN"]
    quality_problems = quality_df[quality_df["severity"].isin(["WARN", "ERROR"])]
    if fairness_warnings.empty and quality_problems.empty:
        lines.append("- 未发现明显的关键配置不一致或数据完整性问题。")
    else:
        for _, row in fairness_warnings.iterrows():
            lines.append(
                f"- [配置警告] seed={row['seed']}，{row['config_key']}："
                f"{row['reference_condition']}={row['reference_value']}，"
                f"{row['other_condition']}={row['other_value']}。"
            )
        for _, row in quality_problems.iterrows():
            lines.append(
                f"- [{row['severity']}] {row['run_name']}：{row['issue']}。"
            )
    lines.append("")

    lines.append("## 6. 结果解释时必须注意")
    lines.append("")
    lines.append("1. `normal-trained + normal-test` 与 `bypass-trained + bypass-test` 用于判断文本训练的总体作用。")
    lines.append("2. `normal-trained` 模型内部的 `normal-test` 与 `bypass-test` 用于判断推理期文本依赖性。")
    lines.append("3. `normal-trained` 模型的 `normal-test` 与 `shuffle-test` 用于判断正确文本语义是否有效。")
    lines.append("4. `bypass-trained + normal-test` 会启用未训练的文本模块，不应作为主要模型性能。")
    lines.append("5. 只有一个 seed 时不能判断训练随机性的影响；正式论文结果建议至少 3–5 个 seed。")
    lines.append("6. 若 shuffle_singleton_batches > 0，部分单样本 batch 没有真正错配文本，语义结论需要谨慎。")
    lines.append("")

    lines.append("## 7. 已生成图像")
    lines.append("")
    for path in figure_paths:
        try:
            relative = path.relative_to(RESULT_DIR)
        except ValueError:
            relative = path
        lines.append(f"- `{relative}`")
    lines.append("")

    return "\n".join(lines)


def print_console_summary(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    primary_normal: Optional[RunRecord],
    quality_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 88)
    print("Fusiformer 文本作用实验汇总")
    print("=" * 88)
    print(f"输入目录：{BASE_DIR}")
    print(f"结果目录：{RESULT_DIR}")
    print(f"有效实验数：{len(run_df)}")

    print("\n[核心测试指标]")
    display_columns = [
        "run_name",
        "condition",
        "seed",
        "test_mae_at_training_condition",
        "test_rmse_at_training_condition",
        "validation_mae_at_training_condition",
        "best_epoch",
    ]
    existing = [column for column in display_columns if column in run_df.columns]
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(run_df[existing].to_string(index=False))

    if not paired_df.empty:
        print("\n[相同 seed 的训练条件比较]")
        display_columns = [
            "seed",
            "normal_test_mae",
            "bypass_test_mae",
            "shuffle_test_mae",
            "training_relative_text_gain_mae",
            "training_relative_semantic_gain_mae",
        ]
        existing = [column for column in display_columns if column in paired_df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(paired_df[existing].to_string(index=False))

    if primary_normal is not None:
        row = run_df[run_df["run_name"] == primary_normal.run_name]
        if not row.empty:
            row = row.iloc[0]
            print(f"\n[主要 normal 模型：{primary_normal.run_name}]")
            print(
                "normal/bypass/shuffle MAE："
                f"{format_number(row.get('test_normal_mae'))} / "
                f"{format_number(row.get('test_bypass_mae'))} / "
                f"{format_number(row.get('test_shuffle_mae'))}"
            )
            print(
                "推理期相对文本收益："
                f"{format_percent(row.get('inference_relative_text_gain'))}"
            )
            print(
                "相对语义收益："
                f"{format_percent(row.get('semantic_relative_gain'))}"
            )
            print(
                "Benefit/Harm/Neutral："
                f"{format_percent(row.get('benefit_rate'))} / "
                f"{format_percent(row.get('harm_rate'))} / "
                f"{format_percent(row.get('neutral_rate'))}"
            )

    errors = quality_df[quality_df["severity"] == "ERROR"]
    warnings = quality_df[quality_df["severity"] == "WARN"]
    print(f"\n质量检查：ERROR={len(errors)}，WARN={len(warnings)}")
    if not errors.empty:
        for _, row in errors.iterrows():
            print(f"  [ERROR] {row['run_name']}：{row['issue']}")
    if not warnings.empty:
        for _, row in warnings.head(10).iterrows():
            print(f"  [WARN] {row['run_name']}：{row['issue']}")

    print("\n全部表格、报告和图片已写入 result/。")
    print("=" * 88)


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    configure_matplotlib()

    if not BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"BASE_DIR 不存在：{BASE_DIR}\n"
            "请打开脚本并修改顶部的 BASE_DIR。"
        )

    prepare_result_dir()

    print(f"扫描实验目录：{BASE_DIR}")
    run_dirs = discover_run_directories(BASE_DIR)
    if not run_dirs:
        raise FileNotFoundError(
            "没有找到实验目录。实验目录中至少应包含 config.json、"
            "final_result.json、epoch_metrics.csv 或 text_effect_test。"
        )

    print("发现目录：")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")

    all_records = [load_run_record(run_dir) for run_dir in run_dirs]
    selected_records = select_latest_per_condition_seed(all_records)

    all_run_df = pd.DataFrame([run_record_to_row(record) for record in all_records])
    run_df = pd.DataFrame([run_record_to_row(record) for record in selected_records])

    if run_df.empty:
        raise RuntimeError("没有可分析的实验记录。")

    # 稳定排序。
    condition_rank = {condition: index for index, condition in enumerate(CONDITION_ORDER)}
    run_df["_condition_rank"] = run_df["condition"].map(condition_rank).fillna(999)
    run_df = run_df.sort_values(["seed", "_condition_rank", "run_name"]).drop(
        columns=["_condition_rank"]
    )

    # 核心汇总。
    paired_df = paired_seed_comparisons(run_df)
    condition_summary_df = aggregate_condition_summary(run_df)
    pair_summary_df = aggregate_pair_summary(paired_df)
    fairness_df = check_fairness(selected_records)
    quality_df = build_quality_issues(selected_records, run_df)

    # 单样本汇总。
    sample_summary_rows = [
        summarize_samples(record)
        for record in selected_records
        if not record.sample_metrics.empty
    ]
    sample_summary_df = pd.DataFrame(sample_summary_rows)
    primary_normal = choose_primary_normal_record(selected_records)

    # 保存表格。
    save_csv(all_run_df, RESULT_DIR / "tables" / "all_detected_runs.csv")
    save_csv(run_df, RESULT_DIR / "tables" / "selected_run_metrics.csv")
    save_csv(condition_summary_df, RESULT_DIR / "tables" / "condition_summary.csv")
    save_csv(paired_df, RESULT_DIR / "tables" / "paired_seed_comparison.csv")
    save_csv(pair_summary_df, RESULT_DIR / "tables" / "paired_metric_summary.csv")
    save_csv(sample_summary_df, RESULT_DIR / "tables" / "sample_level_summary.csv")
    save_csv(fairness_df, RESULT_DIR / "tables" / "fairness_check.csv")
    save_csv(quality_df, RESULT_DIR / "tables" / "quality_issues.csv")

    # 保存主要 normal 模型单样本表和排名。
    if primary_normal is not None and not primary_normal.sample_metrics.empty:
        primary_samples = ensure_sample_columns(primary_normal.sample_metrics)
        save_csv(
            primary_samples,
            RESULT_DIR
            / "sample_analysis"
            / f"all_samples_{primary_normal.run_name}.csv",
        )
        save_top_samples(primary_normal)

    # 绘图。
    figure_paths: list[Path] = []
    for path in [
        plot_condition_performance(run_df),
        plot_per_seed_mae(paired_df),
        plot_normal_counterfactual(primary_normal) if primary_normal else None,
    ]:
        if path is not None:
            figure_paths.append(path)
    figure_paths.extend(plot_training_curves(selected_records))
    if primary_normal is not None:
        figure_paths.extend(plot_sample_analysis(primary_normal))

    # 总 JSON。
    summary_payload = build_summary_payload(
        run_df,
        paired_df,
        pair_summary_df,
        sample_summary_df,
        primary_normal,
    )
    summary_payload["fairness_check"] = fairness_df.replace({np.nan: None}).to_dict(
        orient="records"
    )
    summary_payload["quality_issues"] = quality_df.replace({np.nan: None}).to_dict(
        orient="records"
    )
    summary_payload["figures"] = [str(path) for path in figure_paths]
    save_json(RESULT_DIR / "summary_metrics.json", summary_payload)

    # Markdown 报告。
    report = make_markdown_report(
        run_df=run_df,
        paired_df=paired_df,
        pair_summary_df=pair_summary_df,
        sample_summary_df=sample_summary_df,
        fairness_df=fairness_df,
        quality_df=quality_df,
        primary_normal=primary_normal,
        figure_paths=figure_paths,
    )
    report_path = RESULT_DIR / "text_effect_report.md"
    report_path.write_text(report, encoding="utf-8")

    # 纯文本版本，便于记事本打开。
    text_report = re.sub(r"`([^`]*)`", r"\1", report)
    text_report = text_report.replace("**", "")
    (RESULT_DIR / "text_effect_report.txt").write_text(
        text_report, encoding="utf-8"
    )

    # 记录本次运行配置。
    save_json(
        RESULT_DIR / "analysis_config.json",
        {
            "base_dir": str(BASE_DIR),
            "result_dir": str(RESULT_DIR),
            "clear_old_result": CLEAR_OLD_RESULT,
            "condition_order": CONDITION_ORDER,
            "top_n_samples": TOP_N_SAMPLES,
            "primary_normal_seed": PRIMARY_NORMAL_SEED,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    )

    print_console_summary(
        run_df=run_df,
        paired_df=paired_df,
        sample_summary_df=sample_summary_df,
        primary_normal=primary_normal,
        quality_df=quality_df,
    )


if __name__ == "__main__":
    main()
