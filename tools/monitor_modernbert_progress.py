#!/usr/bin/env python3
"""Monitor the Fusiformer-V2 ModernBERT precompute checkpoint without modifying it.

Usage (from /root/llm_fusiformer):
    python tools/monitor_modernbert_progress.py
    python tools/monitor_modernbert_progress.py --interval 3

The primary source of truth is the atomic *.progress.json file, updated after
--checkpoint-every documents. Final completion requires the *.meta.json file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path


def read_speed(log_path: Path):
    if not log_path.exists():
        return None
    # Log is small (~3k progress lines for 143k rows with checkpoint=50).
    # Read its tail only, so this remains inexpensive on long-running jobs.
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(size - 32_768, 0))
        content = f.read().decode("utf-8", errors="replace")
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\s+docs/s\b", content)
    return float(matches[-1]) if matches else None


def process_running(pid_path: Path):
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Live progress bar for ModernBERT cache creation")
    parser.add_argument("--output", default="data/v2/modernbert_late_robocrys_segments.npy",
                        help="Exactly the same --output path as precompute script")
    parser.add_argument("--log", default="logs/modernbert_late_full.log")
    parser.add_argument("--pid", default="logs/modernbert_late_full.pid")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    output = Path(args.output)
    state_path = output.with_name(output.name + ".progress.json")
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    log_path = Path(args.log)
    pid_path = Path(args.pid)
    bar_width = 35
    old_size = 0
    while True:
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                count = int(metadata.get("num_samples", metadata.get("shape", [0])[0]))
                print("\n[SUCCESS] 完整缓存已经生成并发布。总文档:", f"{count:,}")
            except (ValueError, OSError, KeyError, TypeError):
                print("\n[SUCCESS] 检测到最终元数据文件:", meta_path)
            return 0

        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                done = int(state["next_row"])
                total = int(state["signature"]["num_samples"])
                if total <= 0 or not (0 <= done <= total):
                    raise ValueError("invalid progress counters")
                fraction = done / total
                ticks = round(fraction * bar_width)
                bar = "#" * ticks + "-" * (bar_width - ticks)
                rate = read_speed(log_path)
                speed = f"{rate:.2f} docs/s" if rate and rate > 0 else "测速中"
                eta = str(dt.timedelta(seconds=int((total - done) / rate))) if rate and rate > 0 else "计算中"
                line = (f"[{bar}] {fraction:6.2%} | {done:,}/{total:,} "
                        f"| {speed} | 预计剩余 {eta}")
                print("\r" + line + " " * max(0, old_size - len(line)), end="", flush=True)
                old_size = len(line)
            except (json.JSONDecodeError, ValueError, KeyError, OSError) as exc:
                print("\n[WARN] 当前进度文件尚不可读取:", exc, file=sys.stderr)
        else:
            print("\r等待进度文件生成...", end="", flush=True)

        running = process_running(pid_path)
        if running is False:
            if meta_path.exists():
                continue
            print("\n[STOPPED] 预计算进程已退出，但未检测到最终缓存元数据。")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                print("\n".join(lines[-15:]))
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
