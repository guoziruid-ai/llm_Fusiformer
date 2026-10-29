#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fusiformer Robo / Wiki 双数据集文本作用实验分析
=================================================

目录布局
--------
默认扫描：

    ../Run_log/20260725/
        ├── Robo/
        │   ├── normal_seed0[/或带时间戳]
        │   ├── normal_seed42[/或带时间戳]
        │   ├── normal_seed2026[/或带时间戳]
        │   ├── bypass_seed0 ...
        │   └── shuffle_seed2026 ...
        └── Wiki/
            └── 同样的 normal / bypass / shuffle × seed 结构

其中：
- Robo、Wiki：两组不同文本数据集；
- normal、bypass、shuffle：三种文本训练模式；
- seed：0、42、2026。

分析目标
--------
1. 分别分析 Robo 和 Wiki 内部的文本训练作用；
2. 比较 normal / bypass / shuffle 在同一数据集、同一 seed 下的测试集表现；
3. 比较 Robo 与 Wiki 的：
   - normal 模型测试 MAE；
   - 相对 graph-only 的文本训练收益；
   - 正确文本相对 shuffle 的语义收益；
   - normal 模型推理期文本依赖和语义敏感性；
   - Benefit / Harm 比例与跨 seed 稳定性；
4. 检查两数据集内部实验公平性、跨数据集配置可比性、seed 完整性和文本梯度。

正式结论口径
------------
- 每个 run 使用验证集指标选择最佳 checkpoint；
- 文本训练作用的正式比较使用最佳 checkpoint 在独立测试集上的 MAE；
- 验证集 MAE 只用于最佳 epoch 选择和训练诊断；
- Robo 与 Wiki 的“优劣”同时从绝对预测性能和相对文本贡献两方面判断。

输出目录
--------

    ../Run_log/20260725/result/

运行方法
--------
1. 检查“用户可编辑配置”；
2. 安装依赖：pip install pandas numpy matplotlib
3. 运行：python analyze_text_effect_results_robo_wiki.py
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
from typing import Any, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError("缺少依赖，请先执行：pip install pandas numpy matplotlib") from exc


# =============================================================================
# 用户可编辑配置
# =============================================================================

# 该目录的直接子目录应包含 Robo/ 与 Wiki/。
BASE_DIR = Path(r"../Run_log/20260725")

# 数据集目录。需要新增数据集时，只需在这里继续添加。
DATASET_DIRS: dict[str, Path] = {
    "Robo": BASE_DIR / "Robo",
    "Wiki": BASE_DIR / "Wiki",
}
DATASET_ORDER = ["Robo", "Wiki"]

# 双数据集统一汇总结果写入日期目录下的 result/。
RESULT_DIR = BASE_DIR / "result"
CLEAR_OLD_RESULT = True

CONDITION_ORDER = ["normal", "bypass", "shuffle"]
EXPECTED_SEEDS = [0, 42, 2026]
PRIMARY_NORMAL_SEED = 0
TOP_N_SAMPLES = 20
FIGURE_DPI = 220
EPSILON = 1e-12

# MAE 多来源一致性检查。
MAE_CONSISTENCY_ATOL = 1e-8
MAE_CONSISTENCY_RTOL = 1e-6

# 数据集内部：三种文本条件必须一致的训练配置。
FAIRNESS_CONFIG_KEYS = [
    "data_path",
    "target",
    "split_ratio",
    "split_counts",
    "qwen_model_name",
    "depth",
    "edge_input_dim",
    "triplet_input_dim",
    "embed_dim",
    "num_heads",
    "freeze_qwen",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "epochs",
    "blr",
    "actual_lr",
    "min_lr",
    "warmup_epochs",
    "weight_decay",
    "betas",
    "clip_grad",
    "amp",
    "drop_last",
    "monitor_requested",
    "monitor",
    "text_test_conditions",
    "shuffle_singleton_policy",
]

# 跨数据集比较时 data_path 本来就应不同；其余关键训练设置应尽量一致。
CROSS_DATASET_CONFIG_KEYS = [
    key for key in FAIRNESS_CONFIG_KEYS if key not in {"data_path"}
]


# =============================================================================
# 通用工具
# =============================================================================


def configure_matplotlib() -> None:
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


def parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def parse_seed(name: str, config: dict[str, Any]) -> Optional[int]:
    config_seed = config.get("seed")
    try:
        return int(config_seed)
    except (TypeError, ValueError):
        pass

    match = re.search(r"seed[_-]?(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def infer_condition(
    name: str,
    config: dict[str, Any],
    final_result: dict[str, Any],
) -> str:
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


def comparable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, comparable_value(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(comparable_value(item) for item in value)
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
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
        )
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


def mean_std(values: pd.Series) -> tuple[int, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0, float("nan"), float("nan")
    std_value = float(numeric.std(ddof=1)) if numeric.size > 1 else 0.0
    return int(numeric.size), float(numeric.mean()), std_value


def prepare_result_dir() -> None:
    if RESULT_DIR.resolve() == BASE_DIR.resolve():
        raise RuntimeError("RESULT_DIR 不能与 BASE_DIR 相同。")
    if CLEAR_OLD_RESULT and RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)
    for directory in [
        RESULT_DIR,
        RESULT_DIR / "tables",
        RESULT_DIR / "figures",
        RESULT_DIR / "figures" / "training_curves",
        RESULT_DIR / "figures" / "sample_analysis",
        RESULT_DIR / "sample_analysis",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 实验目录发现与读取
# =============================================================================


@dataclass
class RunRecord:
    dataset: str
    run_dir: Path
    run_name: str
    condition: str
    seed: Optional[int]
    modified_time: float
    finished_at: Optional[datetime]
    config: dict[str, Any]
    final_result: dict[str, Any]
    best_result: dict[str, Any]
    test_effect: dict[str, Any]
    validation_effect: dict[str, Any]
    epoch_metrics: pd.DataFrame
    test_samples: pd.DataFrame
    validation_samples: pd.DataFrame
    warnings: list[str]


def is_ignored_path(path: Path) -> bool:
    ignored = {"result", ".ipynb_checkpoints", "tensorboard", "checkpoints"}
    return any(part in ignored for part in path.parts)


def looks_like_run_directory(path: Path) -> bool:
    if not path.is_dir() or is_ignored_path(path):
        return False
    markers = [
        path / "config.json",
        path / "final_result.json",
        path / "epoch_metrics.csv",
        path / "text_effect_test" / "text_condition_metrics.json",
    ]
    return any(marker.exists() for marker in markers)


def discover_dataset_run_directories() -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    for dataset in DATASET_ORDER:
        dataset_dir = DATASET_DIRS[dataset]
        if not dataset_dir.is_dir():
            print(f"[WARN] 数据集目录不存在：{dataset_dir}")
            continue

        direct = [
            path
            for path in dataset_dir.iterdir()
            if looks_like_run_directory(path)
        ]
        if direct:
            discovered.extend((dataset, path) for path in sorted(direct))
            continue

        nested: set[Path] = set()
        for marker_name in (
            "final_result.json",
            "config.json",
            "epoch_metrics.csv",
        ):
            for marker in dataset_dir.rglob(marker_name):
                parent = marker.parent
                if looks_like_run_directory(parent):
                    nested.add(parent)
        discovered.extend((dataset, path) for path in sorted(nested))
    return discovered


def load_effect_json(run_dir: Path, split: str) -> dict[str, Any]:
    directory = "text_effect_test" if split == "test" else "text_effect_validation"
    return safe_read_json(run_dir / directory / "text_condition_metrics.json")


def load_samples_csv(run_dir: Path, split: str) -> pd.DataFrame:
    directory = "text_effect_test" if split == "test" else "text_effect_validation"
    return safe_read_csv(run_dir / directory / "text_condition_samples.csv")


def load_run_record(dataset: str, run_dir: Path) -> RunRecord:
    config = safe_read_json(run_dir / "config.json")
    final_result = safe_read_json(run_dir / "final_result.json")
    best_result = safe_read_json(run_dir / "best_result.json")

    test_effect = final_result.get("test_text_effect")
    if not isinstance(test_effect, dict) or not test_effect:
        test_effect = load_effect_json(run_dir, "test")

    validation_effect = final_result.get("validation_text_effect")
    if not isinstance(validation_effect, dict) or not validation_effect:
        validation_effect = load_effect_json(run_dir, "validation")

    epoch_metrics = safe_read_csv(run_dir / "epoch_metrics.csv")
    test_samples = load_samples_csv(run_dir, "test")
    validation_samples = load_samples_csv(run_dir, "validation")

    condition = infer_condition(run_dir.name, config, final_result)
    seed = parse_seed(run_dir.name, config)
    finished_at = parse_datetime(final_result.get("finished_at"))
    warnings: list[str] = []

    if not (run_dir / "config.json").is_file():
        warnings.append("缺少文件：config.json")
    if not (run_dir / "final_result.json").is_file():
        warnings.append("缺少文件：final_result.json")
    if not (run_dir / "epoch_metrics.csv").is_file():
        warnings.append("缺少文件：epoch_metrics.csv")
    if not (run_dir / "text_effect_test" / "text_condition_metrics.json").is_file():
        warnings.append("缺少文件：text_effect_test/text_condition_metrics.json")
    if not (run_dir / "text_effect_test" / "text_condition_samples.csv").is_file():
        warnings.append("缺少文件：text_effect_test/text_condition_samples.csv")

    if condition == "unknown":
        warnings.append("无法识别 text_train_condition")
    if seed is None:
        warnings.append("无法识别随机种子")
    elif seed not in EXPECTED_SEEDS:
        warnings.append(f"随机种子 {seed} 不在预期集合 {EXPECTED_SEEDS} 中")

    config_condition = str(config.get("text_train_condition", "")).strip().lower()
    if config_condition and condition != "unknown" and config_condition != condition:
        warnings.append(
            f"目录/最终结果识别条件={condition}，但 config 条件={config_condition}"
        )

    config_seed = config.get("seed")
    try:
        parsed_config_seed = int(config_seed)
    except (TypeError, ValueError):
        parsed_config_seed = None
    if seed is not None and parsed_config_seed is not None and seed != parsed_config_seed:
        warnings.append(f"目录 seed={seed}，但 config seed={parsed_config_seed}")

    return RunRecord(
        dataset=dataset,
        run_dir=run_dir,
        run_name=run_dir.name,
        condition=condition,
        seed=seed,
        modified_time=run_dir.stat().st_mtime,
        finished_at=finished_at,
        config=config,
        final_result=final_result,
        best_result=best_result,
        test_effect=test_effect,
        validation_effect=validation_effect,
        epoch_metrics=epoch_metrics,
        test_samples=test_samples,
        validation_samples=validation_samples,
        warnings=warnings,
    )


# =============================================================================
# 指标选择与来源审计
# =============================================================================


def effect_for_split(record: RunRecord, split: str) -> dict[str, Any]:
    return record.test_effect if split == "test" else record.validation_effect


def samples_for_split(record: RunRecord, split: str) -> pd.DataFrame:
    return record.test_samples if split == "test" else record.validation_samples


def extract_condition_metrics(
    effect: dict[str, Any],
    condition: str,
) -> dict[str, float]:
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

    for condition in CONDITION_ORDER:
        error_column = f"absolute_error_{condition}"
        prediction_column = f"prediction_{condition}"
        if error_column not in df.columns and {
            prediction_column,
            "target",
        }.issubset(df.columns):
            df[error_column] = (df[prediction_column] - df["target"]).abs()

    if "prediction_shift" not in df.columns and {
        "prediction_normal",
        "prediction_bypass",
    }.issubset(df.columns):
        df["prediction_shift"] = (
            df["prediction_normal"] - df["prediction_bypass"]
        ).abs()

    if "signed_prediction_shift" not in df.columns and {
        "prediction_normal",
        "prediction_bypass",
    }.issubset(df.columns):
        df["signed_prediction_shift"] = (
            df["prediction_normal"] - df["prediction_bypass"]
        )

    if "mae_gain_from_text" not in df.columns and {
        "absolute_error_normal",
        "absolute_error_bypass",
    }.issubset(df.columns):
        df["mae_gain_from_text"] = (
            df["absolute_error_bypass"] - df["absolute_error_normal"]
        )

    if "semantic_prediction_shift" not in df.columns and {
        "prediction_normal",
        "prediction_shuffle",
    }.issubset(df.columns):
        df["semantic_prediction_shift"] = (
            df["prediction_normal"] - df["prediction_shuffle"]
        ).abs()

    if "mae_gain_correct_vs_shuffle" not in df.columns and {
        "absolute_error_normal",
        "absolute_error_shuffle",
    }.issubset(df.columns):
        df["mae_gain_correct_vs_shuffle"] = (
            df["absolute_error_shuffle"] - df["absolute_error_normal"]
        )
    return df


def recompute_mae_from_samples(
    record: RunRecord,
    split: str,
    inference_condition: str,
) -> tuple[float, str]:
    df = ensure_sample_columns(samples_for_split(record, split))
    if df.empty:
        return float("nan"), ""

    error_column = f"absolute_error_{inference_condition}"
    if error_column in df.columns:
        values = pd.to_numeric(df[error_column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean()), f"mean({error_column})"

    prediction_column = f"prediction_{inference_condition}"
    if {prediction_column, "target"}.issubset(df.columns):
        prediction = pd.to_numeric(df[prediction_column], errors="coerce")
        target = pd.to_numeric(df["target"], errors="coerce")
        mask = prediction.notna() & target.notna()
        if mask.any():
            value = float((prediction[mask] - target[mask]).abs().mean())
            return value, f"mean(abs({prediction_column}-target))"
    return float("nan"), ""


def collect_mae_candidates(
    record: RunRecord,
    split: str,
    inference_condition: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # 训练条件对应的正式 final_result 指标，只有推理条件等于训练条件时可使用。
    if inference_condition == record.condition:
        key = f"{split}_at_training_condition"
        metrics = record.final_result.get(key)
        if isinstance(metrics, dict):
            value = to_float(metrics.get("mae"))
            if math.isfinite(value):
                candidates.append(
                    {
                        "source": "final_result",
                        "path": f"final_result.json::{key}.mae",
                        "value": value,
                    }
                )

    # 反事实多条件汇总。
    effect = effect_for_split(record, split)
    value = to_float(
        nested_get(effect, "conditions", inference_condition, "mae", default=None)
    )
    if math.isfinite(value):
        directory = "text_effect_test" if split == "test" else "text_effect_validation"
        candidates.append(
            {
                "source": "condition_metrics",
                "path": (
                    f"{directory}/text_condition_metrics.json::"
                    f"conditions.{inference_condition}.mae"
                ),
                "value": value,
            }
        )

    # 单样本独立重算。
    recomputed, formula = recompute_mae_from_samples(
        record,
        split,
        inference_condition,
    )
    if math.isfinite(recomputed):
        directory = "text_effect_test" if split == "test" else "text_effect_validation"
        candidates.append(
            {
                "source": "sample_recomputed",
                "path": f"{directory}/text_condition_samples.csv::{formula}",
                "value": recomputed,
            }
        )
    return candidates


def select_mae(
    record: RunRecord,
    split: str,
    inference_condition: str,
) -> dict[str, Any]:
    candidates = collect_mae_candidates(record, split, inference_condition)
    if not candidates:
        return {
            "value": float("nan"),
            "source": "",
            "path": "",
            "candidate_count": 0,
            "consistent": False,
            "mismatch_detail": "无可用来源",
            "candidates": [],
        }

    selected = candidates[0]
    reference = float(selected["value"])
    mismatch_items: list[str] = []
    consistent = True
    for candidate in candidates[1:]:
        value = float(candidate["value"])
        if not math.isclose(
            reference,
            value,
            rel_tol=MAE_CONSISTENCY_RTOL,
            abs_tol=MAE_CONSISTENCY_ATOL,
        ):
            consistent = False
            mismatch_items.append(
                f"{selected['source']}={reference:.12g} vs "
                f"{candidate['source']}={value:.12g}"
            )

    return {
        "value": reference,
        "source": selected["source"],
        "path": selected["path"],
        "candidate_count": len(candidates),
        "consistent": consistent,
        "mismatch_detail": "；".join(mismatch_items),
        "candidates": candidates,
    }


def extract_training_condition_other_metrics(
    record: RunRecord,
    split: str,
) -> dict[str, float]:
    key = f"{split}_at_training_condition"
    metrics = record.final_result.get(key)
    if isinstance(metrics, dict):
        result = {
            "mse": to_float(metrics.get("mse")),
            "rmse": to_float(metrics.get("rmse")),
        }
        if any(math.isfinite(value) for value in result.values()):
            return result
    effect_metrics = extract_condition_metrics(effect_for_split(record, split), record.condition)
    return {"mse": effect_metrics["mse"], "rmse": effect_metrics["rmse"]}


def parse_gradient_check(run_dir: Path) -> dict[str, float]:
    result = {
        "gradient_parameters": float("nan"),
        "gradient_not_none": float("nan"),
        "gradient_nonzero": float("nan"),
        "gradient_summed_norm": float("nan"),
    }
    log_path = run_dir / "train.log"
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
    return {
        "gradient_parameters": float(match.group("parameters")),
        "gradient_not_none": float(match.group("not_none")),
        "gradient_nonzero": float(match.group("nonzero")),
        "gradient_summed_norm": float(match.group("norm")),
    }


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
        index = df["val_mae"].idxmin()
        result["min_val_mae"] = to_float(df.loc[index, "val_mae"])
        result["min_val_mae_epoch"] = to_float(df.loc[index, "epoch"])

    if "epoch_seconds" in df.columns:
        result["total_epoch_seconds"] = to_float(df["epoch_seconds"].sum())
    return result


def run_record_to_row(record: RunRecord) -> dict[str, Any]:
    test_training_mae = select_mae(record, "test", record.condition)
    val_training_mae = select_mae(record, "validation", record.condition)
    test_other = extract_training_condition_other_metrics(record, "test")
    val_other = extract_training_condition_other_metrics(record, "validation")

    selected_test_mae = {
        condition: select_mae(record, "test", condition)
        for condition in CONDITION_ORDER
    }
    test_condition_metrics = {
        condition: extract_condition_metrics(record.test_effect, condition)
        for condition in CONDITION_ORDER
    }

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

    best_epoch = record.final_result.get("best_epoch", record.best_result.get("best_epoch"))
    monitor = record.final_result.get("monitor", record.best_result.get("monitor", ""))
    best_metric = to_float(
        record.final_result.get("best_metric", record.best_result.get("best_metric"))
    )

    normal_mae = selected_test_mae["normal"]["value"]
    bypass_mae = selected_test_mae["bypass"]["value"]
    shuffle_mae = selected_test_mae["shuffle"]["value"]

    # 优先使用实验程序已保存的汇总；缺失时由三种 MAE 重算。
    inference_absolute_text_gain = to_float(normal_vs_bypass.get("absolute_mae_gain"))
    if not math.isfinite(inference_absolute_text_gain):
        inference_absolute_text_gain = (
            bypass_mae - normal_mae
            if math.isfinite(bypass_mae) and math.isfinite(normal_mae)
            else float("nan")
        )
    inference_relative_text_gain = to_float(normal_vs_bypass.get("relative_text_gain"))
    if not math.isfinite(inference_relative_text_gain):
        inference_relative_text_gain = safe_divide(
            inference_absolute_text_gain,
            bypass_mae,
        )

    semantic_absolute_gain = to_float(
        normal_vs_shuffle.get("absolute_semantic_mae_gain")
    )
    if not math.isfinite(semantic_absolute_gain):
        semantic_absolute_gain = (
            shuffle_mae - normal_mae
            if math.isfinite(shuffle_mae) and math.isfinite(normal_mae)
            else float("nan")
        )
    semantic_relative_gain = to_float(normal_vs_shuffle.get("relative_semantic_gain"))
    if not math.isfinite(semantic_relative_gain):
        semantic_relative_gain = safe_divide(semantic_absolute_gain, shuffle_mae)

    return {
        "dataset": record.dataset,
        "run_name": record.run_name,
        "run_dir": str(record.run_dir),
        "condition": record.condition,
        "seed": record.seed,
        "finished_at": record.finished_at.isoformat() if record.finished_at else "",
        "best_epoch": best_epoch,
        "monitor": monitor,
        "best_metric": best_metric,
        "training_seconds": training_seconds,
        "training_hours": safe_divide(training_seconds, 3600.0),
        "test_mse_at_training_condition": test_other["mse"],
        "test_mae_at_training_condition": test_training_mae["value"],
        "test_rmse_at_training_condition": test_other["rmse"],
        "test_mae_source_at_training_condition": test_training_mae["source"],
        "test_mae_path_at_training_condition": test_training_mae["path"],
        "test_mae_sources_consistent": test_training_mae["consistent"],
        "test_mae_mismatch_detail": test_training_mae["mismatch_detail"],
        "validation_mse_at_training_condition": val_other["mse"],
        "validation_mae_at_training_condition": val_training_mae["value"],
        "validation_rmse_at_training_condition": val_other["rmse"],
        "validation_mae_source_at_training_condition": val_training_mae["source"],
        "validation_mae_sources_consistent": val_training_mae["consistent"],
        "validation_mae_mismatch_detail": val_training_mae["mismatch_detail"],
        "test_generalization_gap_mae": (
            test_training_mae["value"] - val_training_mae["value"]
            if math.isfinite(test_training_mae["value"])
            and math.isfinite(val_training_mae["value"])
            else float("nan")
        ),
        **{
            f"test_{condition}_mae": selected_test_mae[condition]["value"]
            for condition in CONDITION_ORDER
        },
        **{
            f"test_{condition}_mae_source": selected_test_mae[condition]["source"]
            for condition in CONDITION_ORDER
        },
        **{
            f"test_{condition}_mse": test_condition_metrics[condition]["mse"]
            for condition in CONDITION_ORDER
        },
        **{
            f"test_{condition}_rmse": test_condition_metrics[condition]["rmse"]
            for condition in CONDITION_ORDER
        },
        "shuffle_singleton_batches": test_condition_metrics["shuffle"][
            "shuffle_singleton_batches"
        ],
        "inference_absolute_text_gain": inference_absolute_text_gain,
        "inference_relative_text_gain": inference_relative_text_gain,
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
        "semantic_absolute_mae_gain": semantic_absolute_gain,
        "semantic_relative_gain": semantic_relative_gain,
        "semantic_mean_prediction_shift": to_float(
            normal_vs_shuffle.get("mean_prediction_shift")
        ),
        "config_seed": record.config.get("seed"),
        "config_batch_size": record.config.get("batch_size"),
        "config_gradient_accumulation_steps": record.config.get(
            "gradient_accumulation_steps"
        ),
        "config_effective_batch_size": record.config.get("effective_batch_size"),
        "config_epochs": record.config.get("epochs"),
        "config_actual_lr": record.config.get("actual_lr"),
        "config_amp": record.config.get("amp"),
        "config_target": record.config.get("target"),
        "config_data_path": record.config.get("data_path"),
        "config_qwen_model_name": record.config.get("qwen_model_name"),
        "warning_count": len(record.warnings),
        "warnings": "；".join(record.warnings),
        **epoch_info,
        **gradient_info,
    }


# =============================================================================
# 实验选择、覆盖率与公平性
# =============================================================================


def select_latest_per_dataset_condition_seed(
    records: list[RunRecord],
) -> list[RunRecord]:
    grouped: dict[tuple[str, str, Optional[int]], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.dataset, record.condition, record.seed),
            [],
        ).append(record)

    selected: list[RunRecord] = []
    for group_records in grouped.values():
        group_records.sort(
            key=lambda item: (
                1 if item.final_result else 0,
                1 if item.test_effect else 0,
                item.finished_at.timestamp() if item.finished_at else item.modified_time,
                item.modified_time,
            ),
            reverse=True,
        )
        selected.append(group_records[0])

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    condition_rank = {name: index for index, name in enumerate(CONDITION_ORDER)}
    return sorted(
        selected,
        key=lambda item: (
            dataset_rank.get(item.dataset, 999),
            item.seed if item.seed is not None else 10**9,
            condition_rank.get(item.condition, 999),
        ),
    )


def build_seed_coverage(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        for seed in EXPECTED_SEEDS:
            for condition in CONDITION_ORDER:
                matches = [
                    record
                    for record in records
                    if record.dataset == dataset
                    and record.seed == seed
                    and record.condition == condition
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "condition": condition,
                        "present": bool(matches),
                        "run_name": matches[0].run_name if matches else "",
                        "status": "PASS" if matches else "MISSING",
                    }
                )
    return pd.DataFrame(rows)


def check_within_dataset_fairness(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_records = [record for record in records if record.dataset == dataset]
        for seed in EXPECTED_SEEDS:
            seed_records = [record for record in dataset_records if record.seed == seed]
            by_condition = {record.condition: record for record in seed_records}
            reference = by_condition.get("normal")
            if reference is None:
                rows.append(
                    {
                        "scope": "within_dataset",
                        "dataset": dataset,
                        "seed": seed,
                        "condition": "normal",
                        "config_key": "",
                        "reference_value": "",
                        "other_value": "",
                        "status": "WARN",
                        "message": "缺少 normal 参考实验",
                    }
                )
                continue

            for condition in CONDITION_ORDER:
                if condition not in by_condition:
                    rows.append(
                        {
                            "scope": "within_dataset",
                            "dataset": dataset,
                            "seed": seed,
                            "condition": condition,
                            "config_key": "",
                            "reference_value": "",
                            "other_value": "",
                            "status": "WARN",
                            "message": f"缺少训练条件 {condition}",
                        }
                    )

            for condition, other in by_condition.items():
                if condition == "normal":
                    continue
                for key in FAIRNESS_CONFIG_KEYS:
                    reference_value = reference.config.get(key)
                    other_value = other.config.get(key)
                    if comparable_value(reference_value) != comparable_value(other_value):
                        rows.append(
                            {
                                "scope": "within_dataset",
                                "dataset": dataset,
                                "seed": seed,
                                "condition": condition,
                                "config_key": key,
                                "reference_value": json.dumps(
                                    reference_value,
                                    ensure_ascii=False,
                                ),
                                "other_value": json.dumps(
                                    other_value,
                                    ensure_ascii=False,
                                ),
                                "status": "WARN",
                                "message": "同数据集三种训练条件关键配置不一致",
                            }
                        )

    if not rows:
        rows.append(
            {
                "scope": "within_dataset",
                "dataset": "all",
                "seed": "all",
                "condition": "all",
                "config_key": "",
                "reference_value": "",
                "other_value": "",
                "status": "PASS",
                "message": "Robo 与 Wiki 内部同 seed 三种条件关键配置一致",
            }
        )
    return pd.DataFrame(rows)


def check_cross_dataset_comparability(records: list[RunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(DATASET_ORDER) < 2:
        return pd.DataFrame()

    reference_dataset = DATASET_ORDER[0]
    other_datasets = DATASET_ORDER[1:]
    for seed in EXPECTED_SEEDS:
        for condition in CONDITION_ORDER:
            reference = next(
                (
                    record
                    for record in records
                    if record.dataset == reference_dataset
                    and record.seed == seed
                    and record.condition == condition
                ),
                None,
            )
            if reference is None:
                continue
            for other_dataset in other_datasets:
                other = next(
                    (
                        record
                        for record in records
                        if record.dataset == other_dataset
                        and record.seed == seed
                        and record.condition == condition
                    ),
                    None,
                )
                if other is None:
                    continue
                for key in CROSS_DATASET_CONFIG_KEYS:
                    reference_value = reference.config.get(key)
                    other_value = other.config.get(key)
                    if comparable_value(reference_value) != comparable_value(other_value):
                        rows.append(
                            {
                                "scope": "cross_dataset",
                                "seed": seed,
                                "condition": condition,
                                "config_key": key,
                                "reference_dataset": reference_dataset,
                                "other_dataset": other_dataset,
                                "reference_value": json.dumps(
                                    reference_value,
                                    ensure_ascii=False,
                                ),
                                "other_value": json.dumps(
                                    other_value,
                                    ensure_ascii=False,
                                ),
                                "status": "WARN",
                                "message": "跨数据集关键训练配置不一致，优劣比较需谨慎",
                            }
                        )

    if not rows:
        rows.append(
            {
                "scope": "cross_dataset",
                "seed": "all",
                "condition": "all",
                "config_key": "",
                "reference_dataset": DATASET_ORDER[0],
                "other_dataset": ",".join(DATASET_ORDER[1:]),
                "reference_value": "",
                "other_value": "",
                "status": "PASS",
                "message": "除 data_path 外，Robo 与 Wiki 的关键训练配置一致",
            }
        )
    return pd.DataFrame(rows)


def build_quality_issues(
    records: list[RunRecord],
    run_df: pd.DataFrame,
    seed_coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []

    for record in records:
        for warning in record.warnings:
            issues.append(
                {
                    "severity": "WARN",
                    "dataset": record.dataset,
                    "run_name": record.run_name,
                    "issue": warning,
                }
            )

    missing = seed_coverage_df[~seed_coverage_df["present"]]
    for _, row in missing.iterrows():
        issues.append(
            {
                "severity": "ERROR",
                "dataset": row["dataset"],
                "run_name": f"{row['condition']}_seed{row['seed']}",
                "issue": "缺少预期实验目录",
            }
        )

    for _, row in run_df.iterrows():
        dataset = str(row["dataset"])
        run_name = str(row["run_name"])
        condition = str(row["condition"])

        singleton = to_float(row.get("shuffle_singleton_batches"))
        if math.isfinite(singleton) and singleton > 0:
            issues.append(
                {
                    "severity": "WARN",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": (
                        f"shuffle 测试包含 {int(singleton)} 个 batch size=1，"
                        "这些 batch 无法真正错配文本"
                    ),
                }
            )

        rates = [
            to_float(row.get("benefit_rate")),
            to_float(row.get("harm_rate")),
            to_float(row.get("neutral_rate")),
        ]
        if all(math.isfinite(value) for value in rates) and abs(sum(rates) - 1.0) > 1e-5:
            issues.append(
                {
                    "severity": "WARN",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": "Benefit/Harm/Neutral 比例之和不等于 1",
                }
            )

        gradient_nonzero = to_float(row.get("gradient_nonzero"))
        if condition == "bypass" and math.isfinite(gradient_nonzero) and gradient_nonzero != 0:
            issues.append(
                {
                    "severity": "ERROR",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": (
                        f"bypass 实验文本相关参数存在 {int(gradient_nonzero)} 个非零梯度"
                    ),
                }
            )
        if condition in {"normal", "shuffle"} and math.isfinite(gradient_nonzero) and gradient_nonzero == 0:
            issues.append(
                {
                    "severity": "WARN",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": f"{condition} 实验文本相关参数非零梯度数量为 0",
                }
            )

        test_mae = to_float(row.get("test_mae_at_training_condition"))
        if not math.isfinite(test_mae):
            issues.append(
                {
                    "severity": "ERROR",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": "缺少训练条件对应的测试 MAE",
                }
            )

        if row.get("test_mae_sources_consistent") is False:
            issues.append(
                {
                    "severity": "WARN",
                    "dataset": dataset,
                    "run_name": run_name,
                    "issue": (
                        "测试 MAE 多来源不一致："
                        f"{row.get('test_mae_mismatch_detail', '')}"
                    ),
                }
            )

        monitor = str(row.get("monitor", "")).strip().lower()
        best_epoch = to_float(row.get("best_epoch"))
        min_val_epoch = to_float(row.get("min_val_mae_epoch"))
        if monitor == "val_mae" and math.isfinite(best_epoch) and math.isfinite(min_val_epoch):
            if int(best_epoch) != int(min_val_epoch):
                issues.append(
                    {
                        "severity": "WARN",
                        "dataset": dataset,
                        "run_name": run_name,
                        "issue": (
                            f"best_epoch={int(best_epoch)} 与最低 val_mae epoch="
                            f"{int(min_val_epoch)} 不一致"
                        ),
                    }
                )

    if not issues:
        issues.append(
            {
                "severity": "PASS",
                "dataset": "all",
                "run_name": "all",
                "issue": "未发现明显的数据完整性或实验程序问题",
            }
        )
    return pd.DataFrame(issues)


# =============================================================================
# 数据集内部配对与双数据集汇总
# =============================================================================


def paired_seed_comparisons(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = run_df[
        run_df["condition"].isin(CONDITION_ORDER) & run_df["seed"].notna()
    ].copy()

    for (dataset, seed), group in valid.groupby(["dataset", "seed"]):
        by_condition = {
            row["condition"]: row
            for _, row in group.sort_values("finished_at").iterrows()
        }
        normal = by_condition.get("normal")
        bypass = by_condition.get("bypass")
        shuffle = by_condition.get("shuffle")
        row: dict[str, Any] = {"dataset": dataset, "seed": int(seed)}

        for condition, source in [
            ("normal", normal),
            ("bypass", bypass),
            ("shuffle", shuffle),
        ]:
            if source is None:
                continue
            row[f"{condition}_test_mae"] = to_float(
                source["test_mae_at_training_condition"]
            )
            row[f"{condition}_test_rmse"] = to_float(
                source["test_rmse_at_training_condition"]
            )
            row[f"{condition}_validation_mae"] = to_float(
                source["validation_mae_at_training_condition"]
            )
            row[f"{condition}_best_epoch"] = to_float(source["best_epoch"])
            row[f"{condition}_training_hours"] = to_float(source["training_hours"])

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
            row["training_absolute_text_gain_mae"],
            bypass_mae,
        )
        row["training_absolute_semantic_gain_mae"] = (
            shuffle_mae - normal_mae
            if math.isfinite(shuffle_mae) and math.isfinite(normal_mae)
            else float("nan")
        )
        row["training_relative_semantic_gain_mae"] = safe_divide(
            row["training_absolute_semantic_gain_mae"],
            shuffle_mae,
        )
        row["training_absolute_text_gain_rmse"] = (
            bypass_rmse - normal_rmse
            if math.isfinite(bypass_rmse) and math.isfinite(normal_rmse)
            else float("nan")
        )
        row["training_relative_text_gain_rmse"] = safe_divide(
            row["training_absolute_text_gain_rmse"],
            bypass_rmse,
        )
        row["training_absolute_semantic_gain_rmse"] = (
            shuffle_rmse - normal_rmse
            if math.isfinite(shuffle_rmse) and math.isfinite(normal_rmse)
            else float("nan")
        )
        row["training_relative_semantic_gain_rmse"] = safe_divide(
            row["training_absolute_semantic_gain_rmse"],
            shuffle_rmse,
        )

        # 验证集仅作诊断，不用于正式训练作用结论。
        normal_val = to_float(row.get("normal_validation_mae"))
        bypass_val = to_float(row.get("bypass_validation_mae"))
        shuffle_val = to_float(row.get("shuffle_validation_mae"))
        row["diagnostic_validation_text_gain_mae"] = (
            bypass_val - normal_val
            if math.isfinite(bypass_val) and math.isfinite(normal_val)
            else float("nan")
        )
        row["diagnostic_validation_semantic_gain_mae"] = (
            shuffle_val - normal_val
            if math.isfinite(shuffle_val) and math.isfinite(normal_val)
            else float("nan")
        )
        row["complete_triplet"] = all(
            math.isfinite(value)
            for value in [normal_mae, bypass_mae, shuffle_mae]
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    result = pd.DataFrame(rows)
    result["_dataset_rank"] = result["dataset"].map(dataset_rank).fillna(999)
    return result.sort_values(["_dataset_rank", "seed"]).drop(columns=["_dataset_rank"])


def aggregate_condition_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "test_mse_at_training_condition",
        "test_mae_at_training_condition",
        "test_rmse_at_training_condition",
        "validation_mae_at_training_condition",
        "test_generalization_gap_mae",
        "training_hours",
    ]
    for dataset in DATASET_ORDER:
        for condition in CONDITION_ORDER:
            group = run_df[
                (run_df["dataset"] == dataset)
                & (run_df["condition"] == condition)
            ]
            for metric in metrics:
                count, mean_value, std_value = mean_std(group.get(metric, pd.Series(dtype=float)))
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "metric": metric,
                        "count": count,
                        "mean": mean_value,
                        "std": std_value,
                        "min": to_float(pd.to_numeric(group.get(metric), errors="coerce").min())
                        if not group.empty
                        else float("nan"),
                        "max": to_float(pd.to_numeric(group.get(metric), errors="coerce").max())
                        if not group.empty
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_pair_summary(paired_df: pd.DataFrame) -> pd.DataFrame:
    if paired_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metric_columns = [
        column
        for column in paired_df.columns
        if column not in {"dataset", "seed", "complete_triplet"}
        and pd.api.types.is_numeric_dtype(paired_df[column])
    ]
    for dataset in DATASET_ORDER:
        group = paired_df[paired_df["dataset"] == dataset]
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            ci_low, ci_high = normal_ci95(values.to_numpy())
            rows.append(
                {
                    "dataset": dataset,
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


def summarize_samples(record: RunRecord) -> dict[str, Any]:
    df = ensure_sample_columns(record.test_samples)
    summary: dict[str, Any] = {
        "dataset": record.dataset,
        "run_name": record.run_name,
        "condition": record.condition,
        "seed": record.seed,
        "num_samples": int(len(df)),
    }
    if df.empty:
        return summary

    threshold = to_float(record.test_effect.get("neutral_threshold"))
    if not math.isfinite(threshold):
        threshold = to_float(record.config.get("text_effect_neutral_threshold"))
    if not math.isfinite(threshold):
        threshold = 1e-6

    gain = pd.to_numeric(df.get("mae_gain_from_text"), errors="coerce").dropna()
    semantic_gain = pd.to_numeric(
        df.get("mae_gain_correct_vs_shuffle"),
        errors="coerce",
    ).dropna()
    shift = pd.to_numeric(df.get("prediction_shift"), errors="coerce").dropna()
    dependency = pd.to_numeric(
        df.get("text_dependency_index"),
        errors="coerce",
    ).dropna()

    if not gain.empty:
        values = gain.to_numpy(dtype=float)
        benefit_count = int(np.sum(values > threshold))
        harm_count = int(np.sum(values < -threshold))
        neutral_count = int(values.size - benefit_count - harm_count)
        gain_ci = normal_ci95(values)
        benefit_ci = wilson_interval(benefit_count, values.size)
        gain_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.update(
            {
                "neutral_threshold": threshold,
                "mean_mae_gain_from_text": float(np.mean(values)),
                "median_mae_gain_from_text": float(np.median(values)),
                "std_mae_gain_from_text": gain_std,
                "mae_gain_ci95_low": gain_ci[0],
                "mae_gain_ci95_high": gain_ci[1],
                "cohens_dz_text_gain": safe_divide(float(np.mean(values)), gain_std),
                "benefit_count": benefit_count,
                "harm_count": harm_count,
                "neutral_count": neutral_count,
                "benefit_rate": benefit_count / values.size,
                "harm_rate": harm_count / values.size,
                "neutral_rate": neutral_count / values.size,
                "benefit_rate_ci95_low": benefit_ci[0],
                "benefit_rate_ci95_high": benefit_ci[1],
            }
        )

    if not semantic_gain.empty:
        values = semantic_gain.to_numpy(dtype=float)
        semantic_ci = normal_ci95(values)
        summary.update(
            {
                "mean_semantic_mae_gain": float(np.mean(values)),
                "median_semantic_mae_gain": float(np.median(values)),
                "semantic_gain_ci95_low": semantic_ci[0],
                "semantic_gain_ci95_high": semantic_ci[1],
                "semantic_benefit_rate": float(np.mean(values > threshold)),
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

    if {"prediction_shift", "mae_gain_from_text"}.issubset(df.columns):
        summary["correlation_shift_vs_gain"] = pearson_correlation(
            df["prediction_shift"].to_numpy(dtype=float),
            df["mae_gain_from_text"].to_numpy(dtype=float),
        )
    if {"text_dependency_index", "mae_gain_from_text"}.issubset(df.columns):
        summary["correlation_dependency_vs_gain"] = pearson_correlation(
            df["text_dependency_index"].to_numpy(dtype=float),
            df["mae_gain_from_text"].to_numpy(dtype=float),
        )
    return summary


def build_dataset_summary(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    normal_rows = run_df[run_df["condition"] == "normal"]

    for dataset in DATASET_ORDER:
        dataset_runs = run_df[run_df["dataset"] == dataset]
        dataset_paired = paired_df[paired_df["dataset"] == dataset]
        dataset_normal = normal_rows[normal_rows["dataset"] == dataset]
        row: dict[str, Any] = {
            "dataset": dataset,
            "selected_run_count": int(len(dataset_runs)),
            "seed_count": int(dataset_runs["seed"].nunique(dropna=True)),
            "complete_triplet_seed_count": int(
                dataset_paired.get("complete_triplet", pd.Series(dtype=bool)).fillna(False).sum()
            )
            if not dataset_paired.empty
            else 0,
            "data_paths": " | ".join(
                sorted(
                    {
                        str(item)
                        for item in dataset_runs.get(
                            "config_data_path",
                            pd.Series(dtype=str),
                        ).dropna()
                    }
                )
            ),
        }

        for condition in CONDITION_ORDER:
            group = dataset_runs[dataset_runs["condition"] == condition]
            for metric, short_name in [
                ("test_mae_at_training_condition", "test_mae"),
                ("test_rmse_at_training_condition", "test_rmse"),
                ("validation_mae_at_training_condition", "validation_mae"),
                ("training_hours", "training_hours"),
            ]:
                count, mean_value, std_value = mean_std(group.get(metric, pd.Series(dtype=float)))
                row[f"{condition}_{short_name}_count"] = count
                row[f"{condition}_{short_name}_mean"] = mean_value
                row[f"{condition}_{short_name}_std"] = std_value

        for metric in [
            "training_absolute_text_gain_mae",
            "training_relative_text_gain_mae",
            "training_absolute_semantic_gain_mae",
            "training_relative_semantic_gain_mae",
            "training_absolute_text_gain_rmse",
            "training_relative_text_gain_rmse",
            "training_absolute_semantic_gain_rmse",
            "training_relative_semantic_gain_rmse",
        ]:
            count, mean_value, std_value = mean_std(
                dataset_paired.get(metric, pd.Series(dtype=float))
            )
            row[f"{metric}_count"] = count
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value

        for metric in [
            "test_normal_mae",
            "test_bypass_mae",
            "test_shuffle_mae",
            "inference_absolute_text_gain",
            "inference_relative_text_gain",
            "semantic_absolute_mae_gain",
            "semantic_relative_gain",
            "mean_prediction_shift",
            "mean_text_dependency_index",
            "benefit_rate",
            "harm_rate",
            "neutral_rate",
        ]:
            count, mean_value, std_value = mean_std(
                dataset_normal.get(metric, pd.Series(dtype=float))
            )
            row[f"normal_model_{metric}_count"] = count
            row[f"normal_model_{metric}_mean"] = mean_value
            row[f"normal_model_{metric}_std"] = std_value

        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_comparison(dataset_summary_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary_df.empty or len(DATASET_ORDER) < 2:
        return pd.DataFrame()
    indexed = dataset_summary_df.set_index("dataset")
    if any(dataset not in indexed.index for dataset in DATASET_ORDER[:2]):
        return pd.DataFrame()

    first, second = DATASET_ORDER[:2]
    definitions = [
        (
            "normal_test_mae",
            "normal_test_mae_mean",
            "lower",
            "完整文本模型的绝对测试性能；越低越好",
        ),
        (
            "bypass_test_mae",
            "bypass_test_mae_mean",
            "lower",
            "纯图基线测试性能；越低通常表示该数据集结构预测更容易",
        ),
        (
            "shuffle_test_mae",
            "shuffle_test_mae_mean",
            "lower",
            "错配文本训练模型测试性能；越低越好",
        ),
        (
            "relative_text_training_gain",
            "training_relative_text_gain_mae_mean",
            "higher",
            "normal 相对 bypass 的 MAE 降幅；越高越支持文本训练有效",
        ),
        (
            "relative_semantic_training_gain",
            "training_relative_semantic_gain_mae_mean",
            "higher",
            "normal 相对 shuffle 的 MAE 降幅；越高越支持正确语义有效",
        ),
        (
            "inference_relative_text_gain",
            "normal_model_inference_relative_text_gain_mean",
            "higher",
            "同一 normal 模型移除文本后的相对退化；越高表示文本依赖更强",
        ),
        (
            "inference_relative_semantic_gain",
            "normal_model_semantic_relative_gain_mean",
            "higher",
            "同一 normal 模型正确文本相对错配文本的收益；越高越好",
        ),
        (
            "benefit_rate",
            "normal_model_benefit_rate_mean",
            "higher",
            "normal 模型中受文本帮助的样本比例；越高越好",
        ),
        (
            "harm_rate",
            "normal_model_harm_rate_mean",
            "lower",
            "normal 模型中被文本损害的样本比例；越低越好",
        ),
        (
            "normal_mae_seed_std",
            "normal_test_mae_std",
            "lower",
            "normal 模型跨 seed 的 MAE 波动；越低越稳定",
        ),
        (
            "normal_training_hours",
            "normal_training_hours_mean",
            "lower",
            "normal 模型平均训练时长；越低计算效率越高",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for metric_name, column, direction, meaning in definitions:
        first_value = to_float(indexed.loc[first].get(column))
        second_value = to_float(indexed.loc[second].get(column))
        winner = "N/A"
        if math.isfinite(first_value) and math.isfinite(second_value):
            if math.isclose(first_value, second_value, rel_tol=1e-12, abs_tol=1e-12):
                winner = "tie"
            elif direction == "lower":
                winner = first if first_value < second_value else second
            else:
                winner = first if first_value > second_value else second
        rows.append(
            {
                "metric": metric_name,
                "source_column": column,
                f"{first}_value": first_value,
                f"{second}_value": second_value,
                f"{second}_minus_{first}": (
                    second_value - first_value
                    if math.isfinite(first_value) and math.isfinite(second_value)
                    else float("nan")
                ),
                "better_direction": direction,
                "better_dataset": winner,
                "meaning": meaning,
            }
        )
    return pd.DataFrame(rows)


def build_cross_dataset_seed_comparison(run_df: pd.DataFrame) -> pd.DataFrame:
    if len(DATASET_ORDER) < 2:
        return pd.DataFrame()
    first, second = DATASET_ORDER[:2]
    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for condition in CONDITION_ORDER:
            first_rows = run_df[
                (run_df["dataset"] == first)
                & (run_df["seed"] == seed)
                & (run_df["condition"] == condition)
            ]
            second_rows = run_df[
                (run_df["dataset"] == second)
                & (run_df["seed"] == seed)
                & (run_df["condition"] == condition)
            ]
            first_value = (
                to_float(first_rows.iloc[0]["test_mae_at_training_condition"])
                if not first_rows.empty
                else float("nan")
            )
            second_value = (
                to_float(second_rows.iloc[0]["test_mae_at_training_condition"])
                if not second_rows.empty
                else float("nan")
            )
            winner = "N/A"
            if math.isfinite(first_value) and math.isfinite(second_value):
                if math.isclose(first_value, second_value, rel_tol=1e-12, abs_tol=1e-12):
                    winner = "tie"
                else:
                    winner = first if first_value < second_value else second
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    f"{first}_test_mae": first_value,
                    f"{second}_test_mae": second_value,
                    f"{second}_minus_{first}": (
                        second_value - first_value
                        if math.isfinite(first_value) and math.isfinite(second_value)
                        else float("nan")
                    ),
                    "lower_mae_dataset": winner,
                }
            )
    return pd.DataFrame(rows)


def choose_primary_normal_records(
    records: list[RunRecord],
) -> dict[str, RunRecord]:
    selected: dict[str, RunRecord] = {}
    for dataset in DATASET_ORDER:
        candidates = [
            record
            for record in records
            if record.dataset == dataset and record.condition == "normal"
        ]
        if not candidates:
            continue
        exact = [record for record in candidates if record.seed == PRIMARY_NORMAL_SEED]
        candidates = exact or candidates
        candidates.sort(
            key=lambda item: (
                item.seed if item.seed is not None else 10**9,
                -(item.finished_at.timestamp() if item.finished_at else item.modified_time),
            )
        )
        selected[dataset] = candidates[0]
    return selected


def save_top_samples(record: RunRecord) -> tuple[Path, Path]:
    df = ensure_sample_columns(record.test_samples)
    output_dir = RESULT_DIR / "sample_analysis" / record.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    benefit_path = output_dir / f"top_benefit_{record.run_name}.csv"
    harm_path = output_dir / f"top_harm_{record.run_name}.csv"

    if df.empty or "mae_gain_from_text" not in df.columns:
        save_csv(pd.DataFrame(), benefit_path)
        save_csv(pd.DataFrame(), harm_path)
        return benefit_path, harm_path

    columns = [
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
    save_csv(df.nlargest(TOP_N_SAMPLES, "mae_gain_from_text")[columns], benefit_path)
    save_csv(df.nsmallest(TOP_N_SAMPLES, "mae_gain_from_text")[columns], harm_path)
    return benefit_path, harm_path


# =============================================================================
# 可视化
# =============================================================================


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_dataset_condition_mae(run_df: pd.DataFrame) -> Optional[Path]:
    if run_df.empty:
        return None
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    width = 0.36 if len(DATASET_ORDER) == 2 else 0.8 / max(len(DATASET_ORDER), 1)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plotted = False
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        means: list[float] = []
        stds: list[float] = []
        for condition in CONDITION_ORDER:
            group = run_df[
                (run_df["dataset"] == dataset)
                & (run_df["condition"] == condition)
            ]
            _, mean_value, std_value = mean_std(
                group.get("test_mae_at_training_condition", pd.Series(dtype=float))
            )
            means.append(mean_value)
            stds.append(std_value if math.isfinite(std_value) else 0.0)
        offset = (dataset_index - (len(DATASET_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=dataset,
        )
        ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
        plotted = plotted or any(math.isfinite(value) for value in means)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel("Test MAE (mean ± seed std)")
    ax.set_title("Robo vs Wiki: training-condition test MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = RESULT_DIR / "figures" / "01_dataset_condition_test_mae.png"
    save_figure(path)
    return path


def plot_dataset_text_gains(paired_df: pd.DataFrame) -> Optional[Path]:
    if paired_df.empty:
        return None
    metrics = [
        ("training_relative_text_gain_mae", "Text gain vs bypass"),
        ("training_relative_semantic_gain_mae", "Semantic gain vs shuffle"),
    ]
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    plotted = False
    for offset, (metric, label) in zip([-width / 2, width / 2], metrics):
        means: list[float] = []
        stds: list[float] = []
        for dataset in DATASET_ORDER:
            group = paired_df[paired_df["dataset"] == dataset]
            _, mean_value, std_value = mean_std(group.get(metric, pd.Series(dtype=float)))
            means.append(mean_value)
            stds.append(std_value if math.isfinite(std_value) else 0.0)
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=label)
        ax.bar_label(
            bars,
            labels=[format_percent(value) for value in means],
            padding=3,
            fontsize=8,
        )
        plotted = plotted or any(math.isfinite(value) for value in means)
    if not plotted:
        plt.close(fig)
        return None
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Relative MAE gain")
    ax.set_title("Text training contribution by dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = RESULT_DIR / "figures" / "02_dataset_relative_text_gains.png"
    save_figure(path)
    return path


def plot_per_seed_mae(run_df: pd.DataFrame) -> Optional[Path]:
    if run_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plotted = False
    for dataset in DATASET_ORDER:
        for condition in CONDITION_ORDER:
            group = run_df[
                (run_df["dataset"] == dataset)
                & (run_df["condition"] == condition)
            ].copy()
            if group.empty:
                continue
            group["seed"] = pd.to_numeric(group["seed"], errors="coerce")
            group["test_mae_at_training_condition"] = pd.to_numeric(
                group["test_mae_at_training_condition"],
                errors="coerce",
            )
            group = group.dropna(subset=["seed", "test_mae_at_training_condition"])
            if group.empty:
                continue
            group = group.sort_values("seed")
            ax.plot(
                group["seed"],
                group["test_mae_at_training_condition"],
                marker="o",
                label=f"{dataset}-{condition}",
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(EXPECTED_SEEDS)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Test MAE")
    ax.set_title("Test MAE across seeds: Robo and Wiki")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path = RESULT_DIR / "figures" / "03_per_seed_test_mae.png"
    save_figure(path)
    return path


def plot_normal_counterfactual_by_dataset(run_df: pd.DataFrame) -> Optional[Path]:
    normal_rows = run_df[run_df["condition"] == "normal"]
    if normal_rows.empty:
        return None
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    width = 0.36 if len(DATASET_ORDER) == 2 else 0.8 / max(len(DATASET_ORDER), 1)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    plotted = False
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        group = normal_rows[normal_rows["dataset"] == dataset]
        means: list[float] = []
        stds: list[float] = []
        for inference_condition in CONDITION_ORDER:
            _, mean_value, std_value = mean_std(
                group.get(f"test_{inference_condition}_mae", pd.Series(dtype=float))
            )
            means.append(mean_value)
            stds.append(std_value if math.isfinite(std_value) else 0.0)
        offset = (dataset_index - (len(DATASET_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=dataset,
        )
        ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
        plotted = plotted or any(math.isfinite(value) for value in means)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x)
    ax.set_xticklabels([f"{item} inference" for item in CONDITION_ORDER])
    ax.set_ylabel("Test MAE of normal-trained model")
    ax.set_title("Counterfactual text inference by dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = RESULT_DIR / "figures" / "04_normal_counterfactual_by_dataset.png"
    save_figure(path)
    return path


def plot_benefit_harm_by_dataset(run_df: pd.DataFrame) -> Optional[Path]:
    normal_rows = run_df[run_df["condition"] == "normal"]
    if normal_rows.empty:
        return None
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.28
    metrics = [
        ("benefit_rate", "Benefit"),
        ("harm_rate", "Harm"),
        ("neutral_rate", "Neutral"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    plotted = False
    for metric_index, (metric, label) in enumerate(metrics):
        means: list[float] = []
        stds: list[float] = []
        for dataset in DATASET_ORDER:
            group = normal_rows[normal_rows["dataset"] == dataset]
            _, mean_value, std_value = mean_std(group.get(metric, pd.Series(dtype=float)))
            means.append(mean_value)
            stds.append(std_value if math.isfinite(std_value) else 0.0)
        offset = (metric_index - 1) * width
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=label)
        ax.bar_label(
            bars,
            labels=[format_percent(value) for value in means],
            padding=3,
            fontsize=8,
        )
        plotted = plotted or any(math.isfinite(value) for value in means)
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("Sample proportion")
    ax.set_title("Normal-model text effect classes by dataset")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = RESULT_DIR / "figures" / "05_benefit_harm_by_dataset.png"
    save_figure(path)
    return path


def plot_training_curves(records: list[RunRecord]) -> list[Path]:
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
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
        ax.plot(
            epoch[mask],
            val_mae[mask],
            label=f"{record.dataset}-{record.condition}-seed{record.seed}",
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE")
        ax.set_title("Validation MAE curves: Robo and Wiki")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
        path = RESULT_DIR / "figures" / "06_all_validation_mae_curves.png"
        save_figure(path)
        paths.append(path)
    else:
        plt.close(fig)

    base_dir = RESULT_DIR / "figures" / "training_curves"
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
        ax.set_title(f"{record.dataset}: {record.run_name}")
        ax.grid(alpha=0.25)
        ax.legend()
        path = base_dir / record.dataset / f"{record.run_name}_curves.png"
        save_figure(path)
        paths.append(path)
    return paths


def plot_sample_analysis(record: RunRecord) -> list[Path]:
    df = ensure_sample_columns(record.test_samples)
    if df.empty:
        return []
    paths: list[Path] = []
    figure_dir = RESULT_DIR / "figures" / "sample_analysis" / record.dataset
    figure_dir.mkdir(parents=True, exist_ok=True)

    if {"absolute_error_bypass", "absolute_error_normal"}.issubset(df.columns):
        x = pd.to_numeric(df["absolute_error_bypass"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["absolute_error_normal"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            limit = float(max(np.max(x[mask]), np.max(y[mask])))
            fig, ax = plt.subplots(figsize=(6.5, 6.2))
            ax.scatter(x[mask], y[mask], alpha=0.55, s=18)
            ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
            ax.set_xlabel("Absolute error: bypass")
            ax.set_ylabel("Absolute error: normal")
            ax.set_title(f"{record.dataset}: normal vs bypass")
            ax.grid(alpha=0.2)
            path = figure_dir / "error_scatter_normal_vs_bypass.png"
            save_figure(path)
            paths.append(path)

    if "mae_gain_from_text" in df.columns:
        gain = pd.to_numeric(df["mae_gain_from_text"], errors="coerce").dropna()
        if not gain.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            ax.hist(gain, bins="auto")
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.axvline(float(gain.mean()), linewidth=1.2, label="mean")
            ax.set_xlabel("MAE gain from text")
            ax.set_ylabel("Sample count")
            ax.set_title(f"{record.dataset}: per-sample text gain")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            path = figure_dir / "text_gain_histogram.png"
            save_figure(path)
            paths.append(path)

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
            ax.set_title(f"{record.dataset}: dependency vs gain (r={correlation:.3f})")
            ax.grid(alpha=0.2)
            path = figure_dir / "dependency_vs_gain.png"
            save_figure(path)
            paths.append(path)

    if "mae_gain_correct_vs_shuffle" in df.columns:
        gain = pd.to_numeric(
            df["mae_gain_correct_vs_shuffle"],
            errors="coerce",
        ).dropna()
        if not gain.empty:
            fig, ax = plt.subplots(figsize=(8.2, 5.2))
            ax.hist(gain, bins="auto")
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.axvline(float(gain.mean()), linewidth=1.2, label="mean")
            ax.set_xlabel("Semantic MAE gain")
            ax.set_ylabel("Sample count")
            ax.set_title(f"{record.dataset}: correct text vs shuffle")
            ax.grid(axis="y", alpha=0.2)
            ax.legend()
            path = figure_dir / "semantic_gain_histogram.png"
            save_figure(path)
            paths.append(path)
    return paths


# =============================================================================
# 报告
# =============================================================================


def metric_summary_value(
    pair_summary_df: pd.DataFrame,
    dataset: str,
    metric: str,
    field: str = "mean",
) -> float:
    if pair_summary_df.empty:
        return float("nan")
    subset = pair_summary_df[
        (pair_summary_df["dataset"] == dataset)
        & (pair_summary_df["metric"] == metric)
    ]
    return to_float(subset.iloc[0][field]) if not subset.empty else float("nan")


def make_markdown_report(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    dataset_comparison_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    cross_fairness_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    seed_coverage_df: pd.DataFrame,
    primary_normals: dict[str, RunRecord],
    figure_paths: list[Path],
) -> str:
    lines: list[str] = []
    lines.append("# Fusiformer Robo / Wiki 双数据集文本作用实验报告")
    lines.append("")
    lines.append(f"- 根目录：`{BASE_DIR}`")
    lines.append(f"- 输出目录：`{RESULT_DIR}`")
    lines.append(f"- 数据集：{', '.join(DATASET_ORDER)}")
    lines.append(f"- 预期随机种子：{EXPECTED_SEEDS}")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 纳入分析 run 数：{len(run_df)}")
    lines.append("")

    lines.append("## 1. 实验覆盖情况")
    lines.append("")
    lines.append("| 数据集 | Seed | normal | bypass | shuffle |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DATASET_ORDER:
        for seed in EXPECTED_SEEDS:
            subset = seed_coverage_df[
                (seed_coverage_df["dataset"] == dataset)
                & (seed_coverage_df["seed"] == seed)
            ]
            statuses = {
                row["condition"]: "✓" if row["present"] else "缺失"
                for _, row in subset.iterrows()
            }
            lines.append(
                f"| {dataset} | {seed} | {statuses.get('normal', '缺失')} | "
                f"{statuses.get('bypass', '缺失')} | {statuses.get('shuffle', '缺失')} |"
            )
    lines.append("")

    lines.append("## 2. 每个 run 的正式测试指标")
    lines.append("")
    lines.append(
        "验证集用于选择最佳 checkpoint；以下 Test MAE/RMSE/MSE 用于正式判断训练作用。"
    )
    lines.append("")
    lines.append(
        "| 数据集 | 条件 | Seed | Test MAE | Test RMSE | Val MAE | Best epoch | 训练小时 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in run_df.sort_values(["dataset", "seed", "condition"]).iterrows():
        lines.append(
            "| {dataset} | {condition} | {seed} | {mae} | {rmse} | {val} | {epoch} | {hours} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                seed=int(row["seed"]) if pd.notna(row["seed"]) else "N/A",
                mae=format_number(row["test_mae_at_training_condition"]),
                rmse=format_number(row["test_rmse_at_training_condition"]),
                val=format_number(row["validation_mae_at_training_condition"]),
                epoch=row["best_epoch"] if pd.notna(row["best_epoch"]) else "N/A",
                hours=format_number(row["training_hours"], 3),
            )
        )
    lines.append("")

    lines.append("## 3. Robo 与 Wiki 的跨 seed 汇总")
    lines.append("")
    lines.append(
        "| 数据集 | normal Test MAE | bypass Test MAE | shuffle Test MAE | 文本训练相对收益 | 语义训练相对收益 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in dataset_summary_df.iterrows():
        lines.append(
            "| {dataset} | {normal} ± {normal_std} | {bypass} ± {bypass_std} | "
            "{shuffle} ± {shuffle_std} | {text_gain} | {semantic_gain} |".format(
                dataset=row["dataset"],
                normal=format_number(row.get("normal_test_mae_mean")),
                normal_std=format_number(row.get("normal_test_mae_std")),
                bypass=format_number(row.get("bypass_test_mae_mean")),
                bypass_std=format_number(row.get("bypass_test_mae_std")),
                shuffle=format_number(row.get("shuffle_test_mae_mean")),
                shuffle_std=format_number(row.get("shuffle_test_mae_std")),
                text_gain=format_percent(
                    row.get("training_relative_text_gain_mae_mean")
                ),
                semantic_gain=format_percent(
                    row.get("training_relative_semantic_gain_mae_mean")
                ),
            )
        )
    lines.append("")
    lines.append("定义：")
    lines.append("")
    lines.append("- 文本训练收益 = `MAE(bypass-trained) - MAE(normal-trained)`；正值表示正确文本训练更好。")
    lines.append("- 语义训练收益 = `MAE(shuffle-trained) - MAE(normal-trained)`；正值表示正确匹配文本优于错配文本。")
    lines.append("")

    lines.append("## 4. 每个 seed 的文本训练作用")
    lines.append("")
    lines.append(
        "| 数据集 | Seed | normal MAE | bypass MAE | shuffle MAE | 文本相对收益 | 语义相对收益 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in paired_df.iterrows():
        lines.append(
            "| {dataset} | {seed} | {normal} | {bypass} | {shuffle} | {text} | {semantic} |".format(
                dataset=row["dataset"],
                seed=int(row["seed"]),
                normal=format_number(row.get("normal_test_mae")),
                bypass=format_number(row.get("bypass_test_mae")),
                shuffle=format_number(row.get("shuffle_test_mae")),
                text=format_percent(row.get("training_relative_text_gain_mae")),
                semantic=format_percent(row.get("training_relative_semantic_gain_mae")),
            )
        )
    lines.append("")

    lines.append("## 5. normal 模型的推理期反事实文本作用")
    lines.append("")
    lines.append(
        "| 数据集 | normal inference MAE | bypass inference MAE | shuffle inference MAE | 推理期文本相对收益 | 推理期语义相对收益 | Benefit | Harm |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in dataset_summary_df.iterrows():
        lines.append(
            "| {dataset} | {normal} | {bypass} | {shuffle} | {text} | {semantic} | {benefit} | {harm} |".format(
                dataset=row["dataset"],
                normal=format_number(row.get("normal_model_test_normal_mae_mean")),
                bypass=format_number(row.get("normal_model_test_bypass_mae_mean")),
                shuffle=format_number(row.get("normal_model_test_shuffle_mae_mean")),
                text=format_percent(
                    row.get("normal_model_inference_relative_text_gain_mean")
                ),
                semantic=format_percent(
                    row.get("normal_model_semantic_relative_gain_mean")
                ),
                benefit=format_percent(row.get("normal_model_benefit_rate_mean")),
                harm=format_percent(row.get("normal_model_harm_rate_mean")),
            )
        )
    lines.append("")
    lines.append(
        "该部分比较同一个 normal-trained 模型在 normal/bypass/shuffle 三种推理输入下的结果，"
        "用于判断模型是否使用文本；它不能替代跨独立训练模型的训练作用比较。"
    )
    lines.append("")

    lines.append("## 6. Robo 与 Wiki 优劣比较")
    lines.append("")
    if dataset_comparison_df.empty:
        lines.append("双数据集指标不完整，无法生成比较表。")
    else:
        first, second = DATASET_ORDER[:2]
        lines.append(
            f"| 比较维度 | {first} | {second} | 更优数据集 | 判定方向 |"
        )
        lines.append("|---|---:|---:|---|---|")
        for _, row in dataset_comparison_df.iterrows():
            direction = "越低越好" if row["better_direction"] == "lower" else "越高越好"
            lines.append(
                f"| {row['metric']} | {format_number(row.get(f'{first}_value'))} | "
                f"{format_number(row.get(f'{second}_value'))} | "
                f"{row['better_dataset']} | {direction} |"
            )
    lines.append("")
    lines.append("解释原则：")
    lines.append("")
    lines.append("1. `normal_test_mae` 更低，表示该数据集上的完整模型绝对预测误差更低。")
    lines.append("2. `relative_text_training_gain` 更高，表示文本相对纯图基线带来的增益更强，是评价文本数据质量的核心指标。")
    lines.append("3. `relative_semantic_training_gain` 更高，表示正确文本—晶体匹配关系更有价值。")
    lines.append("4. 推理期文本收益较高只说明 normal 模型更依赖文本，不必然说明跨模型泛化更好。")
    lines.append("5. 若 Robo 与 Wiki 的结构样本、目标分布或划分不完全一致，绝对 MAE 只能作描述性比较；相对文本收益更适合判断文本数据优劣。")
    lines.append("")

    lines.append("## 7. 单样本分析")
    lines.append("")
    normal_samples = sample_summary_df[sample_summary_df["condition"] == "normal"]
    if normal_samples.empty:
        lines.append("未找到 normal 模型单样本结果。")
    else:
        lines.append(
            "| 数据集 | Seed | 样本数 | 平均文本 MAE 收益 | Benefit Rate | Harm Rate | Cohen's dz | 语义平均收益 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in normal_samples.sort_values(["dataset", "seed"]).iterrows():
            lines.append(
                "| {dataset} | {seed} | {count} | {gain} | {benefit} | {harm} | {dz} | {semantic} |".format(
                    dataset=row["dataset"],
                    seed=int(row["seed"]) if pd.notna(row["seed"]) else "N/A",
                    count=int(row.get("num_samples", 0)),
                    gain=format_number(row.get("mean_mae_gain_from_text")),
                    benefit=format_percent(row.get("benefit_rate")),
                    harm=format_percent(row.get("harm_rate")),
                    dz=format_number(row.get("cohens_dz_text_gain")),
                    semantic=format_number(row.get("mean_semantic_mae_gain")),
                )
            )
    lines.append("")

    lines.append("## 8. 公平性与数据质量")
    lines.append("")
    within_warnings = fairness_df[fairness_df["status"] == "WARN"]
    cross_warnings = cross_fairness_df[cross_fairness_df["status"] == "WARN"]
    quality_problems = quality_df[quality_df["severity"].isin(["WARN", "ERROR"])]
    if within_warnings.empty and cross_warnings.empty and quality_problems.empty:
        lines.append("- 未发现明显的配置不一致、实验缺失或程序异常。")
    else:
        for _, row in within_warnings.iterrows():
            lines.append(
                f"- [数据集内部配置] {row['dataset']} seed={row['seed']} "
                f"{row['condition']} {row['config_key']}：{row['message']}。"
            )
        for _, row in cross_warnings.iterrows():
            lines.append(
                f"- [跨数据集配置] seed={row['seed']} {row['condition']} "
                f"{row['config_key']}：{row['message']}。"
            )
        for _, row in quality_problems.iterrows():
            lines.append(
                f"- [{row['severity']}] {row['dataset']}/{row['run_name']}：{row['issue']}。"
            )
    lines.append("")

    lines.append("## 9. 结果解释注意事项")
    lines.append("")
    lines.append("1. 正式训练作用必须比较各自最佳验证 checkpoint 的独立测试集 MAE。")
    lines.append("2. Robo 与 Wiki 各自都应有 normal/bypass/shuffle × 0/42/2026 共 9 个 run。")
    lines.append("3. 同 seed 配对用于降低随机划分和初始化差异，但 3 个 seed 的统计功效仍有限。")
    lines.append("4. bypass 的文本相关参数非零梯度必须为 0；normal 与 shuffle 应存在非零文本梯度。")
    lines.append("5. shuffle_singleton_batches 应为 0，否则部分样本没有真正错配文本。")
    lines.append("6. 比较文本数据优劣时，应优先看相对文本收益和语义收益，再参考绝对 MAE。")
    lines.append("")

    lines.append("## 10. 已生成图像")
    lines.append("")
    for path in figure_paths:
        try:
            relative = path.relative_to(RESULT_DIR)
        except ValueError:
            relative = path
        lines.append(f"- `{relative}`")
    lines.append("")
    return "\n".join(lines)


def build_summary_payload(
    run_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    condition_summary_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    dataset_comparison_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    fairness_df: pd.DataFrame,
    cross_fairness_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    seed_coverage_df: pd.DataFrame,
    figure_paths: list[Path],
) -> dict[str, Any]:
    def records(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        return df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "base_dir": str(BASE_DIR),
        "dataset_dirs": {name: str(path) for name, path in DATASET_DIRS.items()},
        "result_dir": str(RESULT_DIR),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": DATASET_ORDER,
        "expected_seeds": EXPECTED_SEEDS,
        "conditions": CONDITION_ORDER,
        "formal_training_effect_split": "test",
        "runs": records(run_df),
        "paired_seed_comparisons": records(paired_df),
        "condition_summary": records(condition_summary_df),
        "paired_metric_summary": records(pair_summary_df),
        "dataset_summary": records(dataset_summary_df),
        "dataset_comparison": records(dataset_comparison_df),
        "sample_level_summary": records(sample_summary_df),
        "within_dataset_fairness": records(fairness_df),
        "cross_dataset_comparability": records(cross_fairness_df),
        "quality_issues": records(quality_df),
        "seed_coverage": records(seed_coverage_df),
        "figures": [str(path) for path in figure_paths],
    }


def print_console_summary(
    run_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    dataset_comparison_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 100)
    print("Fusiformer Robo / Wiki 双数据集文本作用实验汇总")
    print("=" * 100)
    print(f"输入根目录：{BASE_DIR}")
    print(f"结果目录：{RESULT_DIR}")
    print(f"有效 run 数：{len(run_df)} / 预期 {len(DATASET_ORDER) * len(CONDITION_ORDER) * len(EXPECTED_SEEDS)}")

    display = [
        "dataset",
        "condition",
        "seed",
        "test_mae_at_training_condition",
        "test_rmse_at_training_condition",
        "validation_mae_at_training_condition",
        "best_epoch",
    ]
    print("\n[全部有效 run]")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(run_df[[column for column in display if column in run_df.columns]].to_string(index=False))

    print("\n[数据集汇总]")
    summary_columns = [
        "dataset",
        "normal_test_mae_mean",
        "bypass_test_mae_mean",
        "shuffle_test_mae_mean",
        "training_relative_text_gain_mae_mean",
        "training_relative_semantic_gain_mae_mean",
        "normal_model_inference_relative_text_gain_mean",
        "normal_model_semantic_relative_gain_mean",
        "normal_model_benefit_rate_mean",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(
            dataset_summary_df[
                [column for column in summary_columns if column in dataset_summary_df.columns]
            ].to_string(index=False)
        )

    if not dataset_comparison_df.empty:
        print("\n[Robo vs Wiki 指标优劣]")
        display_columns = [
            "metric",
            f"{DATASET_ORDER[0]}_value",
            f"{DATASET_ORDER[1]}_value",
            "better_dataset",
        ]
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(dataset_comparison_df[display_columns].to_string(index=False))

    errors = quality_df[quality_df["severity"] == "ERROR"]
    warnings = quality_df[quality_df["severity"] == "WARN"]
    print(f"\n质量检查：ERROR={len(errors)}，WARN={len(warnings)}")
    for _, row in pd.concat([errors, warnings]).head(20).iterrows():
        print(f"  [{row['severity']}] {row['dataset']}/{row['run_name']}：{row['issue']}")
    print("\n全部报告、表格和图像已写入 result/。")
    print("=" * 100)


# =============================================================================
# 主程序
# =============================================================================


def main() -> None:
    configure_matplotlib()
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(
            f"BASE_DIR 不存在：{BASE_DIR}\n请修改脚本顶部的 BASE_DIR。"
        )

    prepare_result_dir()
    discovered = discover_dataset_run_directories()
    if not discovered:
        raise FileNotFoundError(
            "未在 Robo/ 或 Wiki/ 中找到实验目录。实验目录至少应包含 "
            "config.json、final_result.json、epoch_metrics.csv 或 text_effect_test。"
        )

    print("发现实验目录：")
    for dataset, run_dir in discovered:
        print(f"  - {dataset}/{run_dir.name}")

    all_records = [load_run_record(dataset, run_dir) for dataset, run_dir in discovered]
    selected_records = select_latest_per_dataset_condition_seed(all_records)

    all_run_df = pd.DataFrame([run_record_to_row(record) for record in all_records])
    run_df = pd.DataFrame([run_record_to_row(record) for record in selected_records])
    if run_df.empty:
        raise RuntimeError("没有可分析的有效实验记录。")

    dataset_rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    condition_rank = {name: index for index, name in enumerate(CONDITION_ORDER)}
    run_df["_dataset_rank"] = run_df["dataset"].map(dataset_rank).fillna(999)
    run_df["_condition_rank"] = run_df["condition"].map(condition_rank).fillna(999)
    run_df = run_df.sort_values(
        ["_dataset_rank", "seed", "_condition_rank", "run_name"]
    ).drop(columns=["_dataset_rank", "_condition_rank"])

    seed_coverage_df = build_seed_coverage(selected_records)
    paired_df = paired_seed_comparisons(run_df)
    condition_summary_df = aggregate_condition_summary(run_df)
    pair_summary_df = aggregate_pair_summary(paired_df)
    dataset_summary_df = build_dataset_summary(run_df, paired_df)
    dataset_comparison_df = build_dataset_comparison(dataset_summary_df)
    cross_seed_comparison_df = build_cross_dataset_seed_comparison(run_df)

    fairness_df = check_within_dataset_fairness(selected_records)
    cross_fairness_df = check_cross_dataset_comparability(selected_records)
    quality_df = build_quality_issues(selected_records, run_df, seed_coverage_df)

    sample_summary_rows = [
        summarize_samples(record)
        for record in selected_records
        if not record.test_samples.empty
    ]
    sample_summary_df = pd.DataFrame(sample_summary_rows)
    primary_normals = choose_primary_normal_records(selected_records)

    # MAE 来源审计。
    audit_rows: list[dict[str, Any]] = []
    for record in selected_records:
        for split in ["test", "validation"]:
            for inference_condition in CONDITION_ORDER:
                selection = select_mae(record, split, inference_condition)
                audit_rows.append(
                    {
                        "dataset": record.dataset,
                        "run_name": record.run_name,
                        "training_condition": record.condition,
                        "seed": record.seed,
                        "split": split,
                        "inference_condition": inference_condition,
                        "selected_mae": selection["value"],
                        "selected_source": selection["source"],
                        "selected_path": selection["path"],
                        "candidate_count": selection["candidate_count"],
                        "sources_consistent": selection["consistent"],
                        "mismatch_detail": selection["mismatch_detail"],
                    }
                )
    mae_audit_df = pd.DataFrame(audit_rows)

    # 保存表格。
    tables_dir = RESULT_DIR / "tables"
    save_csv(all_run_df, tables_dir / "all_detected_runs.csv")
    save_csv(run_df, tables_dir / "selected_run_metrics.csv")
    save_csv(seed_coverage_df, tables_dir / "seed_coverage.csv")
    save_csv(condition_summary_df, tables_dir / "dataset_condition_summary.csv")
    save_csv(paired_df, tables_dir / "paired_seed_comparison.csv")
    save_csv(pair_summary_df, tables_dir / "dataset_paired_metric_summary.csv")
    save_csv(dataset_summary_df, tables_dir / "dataset_text_effect_summary.csv")
    save_csv(dataset_comparison_df, tables_dir / "robo_vs_wiki_comparison.csv")
    save_csv(cross_seed_comparison_df, tables_dir / "cross_dataset_seed_mae.csv")
    save_csv(sample_summary_df, tables_dir / "sample_level_summary.csv")
    save_csv(fairness_df, tables_dir / "within_dataset_fairness.csv")
    save_csv(cross_fairness_df, tables_dir / "cross_dataset_comparability.csv")
    save_csv(quality_df, tables_dir / "quality_issues.csv")
    save_csv(mae_audit_df, tables_dir / "mae_selection_audit.csv")

    # 每个数据集分别保存 primary normal 的全部样本和 Top 样本。
    for dataset, record in primary_normals.items():
        if record.test_samples.empty:
            continue
        output_dir = RESULT_DIR / "sample_analysis" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(
            ensure_sample_columns(record.test_samples),
            output_dir / f"all_samples_{record.run_name}.csv",
        )
        save_top_samples(record)

    # 绘图。
    figure_paths: list[Path] = []
    for path in [
        plot_dataset_condition_mae(run_df),
        plot_dataset_text_gains(paired_df),
        plot_per_seed_mae(run_df),
        plot_normal_counterfactual_by_dataset(run_df),
        plot_benefit_harm_by_dataset(run_df),
    ]:
        if path is not None:
            figure_paths.append(path)
    figure_paths.extend(plot_training_curves(selected_records))
    for record in primary_normals.values():
        figure_paths.extend(plot_sample_analysis(record))

    summary_payload = build_summary_payload(
        run_df=run_df,
        paired_df=paired_df,
        condition_summary_df=condition_summary_df,
        pair_summary_df=pair_summary_df,
        dataset_summary_df=dataset_summary_df,
        dataset_comparison_df=dataset_comparison_df,
        sample_summary_df=sample_summary_df,
        fairness_df=fairness_df,
        cross_fairness_df=cross_fairness_df,
        quality_df=quality_df,
        seed_coverage_df=seed_coverage_df,
        figure_paths=figure_paths,
    )
    save_json(RESULT_DIR / "summary_metrics.json", summary_payload)

    report = make_markdown_report(
        run_df=run_df,
        paired_df=paired_df,
        pair_summary_df=pair_summary_df,
        dataset_summary_df=dataset_summary_df,
        dataset_comparison_df=dataset_comparison_df,
        sample_summary_df=sample_summary_df,
        fairness_df=fairness_df,
        cross_fairness_df=cross_fairness_df,
        quality_df=quality_df,
        seed_coverage_df=seed_coverage_df,
        primary_normals=primary_normals,
        figure_paths=figure_paths,
    )
    (RESULT_DIR / "text_effect_report.md").write_text(report, encoding="utf-8")
    text_report = re.sub(r"`([^`]*)`", r"\1", report).replace("**", "")
    (RESULT_DIR / "text_effect_report.txt").write_text(text_report, encoding="utf-8")

    save_json(
        RESULT_DIR / "analysis_config.json",
        {
            "base_dir": str(BASE_DIR),
            "dataset_dirs": {name: str(path) for name, path in DATASET_DIRS.items()},
            "result_dir": str(RESULT_DIR),
            "datasets": DATASET_ORDER,
            "conditions": CONDITION_ORDER,
            "expected_seeds": EXPECTED_SEEDS,
            "primary_normal_seed": PRIMARY_NORMAL_SEED,
            "formal_training_effect_split": "test",
            "clear_old_result": CLEAR_OLD_RESULT,
            "top_n_samples": TOP_N_SAMPLES,
            "mae_consistency_atol": MAE_CONSISTENCY_ATOL,
            "mae_consistency_rtol": MAE_CONSISTENCY_RTOL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    )

    print_console_summary(
        run_df=run_df,
        dataset_summary_df=dataset_summary_df,
        dataset_comparison_df=dataset_comparison_df,
        quality_df=quality_df,
    )


if __name__ == "__main__":
    main()
