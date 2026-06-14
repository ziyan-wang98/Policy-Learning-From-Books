#!/usr/bin/env python3
"""Summarize CIQL/d3rlpy logs and propose checkpoint keep sets.

The script is intentionally read-only. It parses d3rlpy CSV logs and `.d3`
checkpoint names, then emits a JSON summary that can be used before manual or
scripted curation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_PATTERNS = (
    re.compile(r"step_(?P<step>\d+)\.d3$"),
    re.compile(r"model_(?P<step>\d+)\.d3$"),
)
REWARD_PATTERN = re.compile(r"model_rew_(?P<reward>-?\d+(?:\.\d+)?)&step_(?P<step>\d+)\.d3$")


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    step: int
    value: float


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    name: str
    step: int | None
    score_from_name: float | None
    size_bytes: int
    sha256: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize d3rlpy CIQL logs and propose checkpoint keep sets."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="A d3rlpy log directory or a parent directory containing one or more log dirs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the JSON summary.",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        action="append",
        default=[],
        help="Checkpoint step that must be retained. Can be repeated.",
    )
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Compute SHA256 for checkpoint files. This can be slow for many large checkpoints.",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=1,
        help="Number of top checkpoints to keep per primary metric. Default: 1.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON to stdout.",
    )
    return parser.parse_args()


def read_csv_points(path: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    points.append(MetricPoint(int(row[0]), int(row[1]), float(row[2])))
                except ValueError:
                    continue
    except OSError:
        return []
    points.sort(key=lambda p: (p.step, p.epoch))
    return points


def metric_name(log_dir: Path, csv_path: Path) -> str:
    rel = csv_path.relative_to(log_dir).as_posix()
    return rel[:-4] if rel.endswith(".csv") else rel


def collect_metric_files(log_dir: Path) -> dict[str, list[MetricPoint]]:
    metrics: dict[str, list[MetricPoint]] = {}
    for csv_path in sorted(log_dir.rglob("*.csv")):
        points = read_csv_points(csv_path)
        if points:
            metrics[metric_name(log_dir, csv_path)] = points
    return metrics


def metric_summary(points: list[MetricPoint]) -> dict[str, Any]:
    best_max = max(points, key=lambda p: (p.value, p.step, p.epoch))
    best_min = min(points, key=lambda p: (p.value, -p.step, -p.epoch))
    latest = max(points, key=lambda p: (p.step, p.epoch))
    first = min(points, key=lambda p: (p.step, p.epoch))
    return {
        "count": len(points),
        "first": asdict(first),
        "latest": asdict(latest),
        "best_max": asdict(best_max),
        "best_min": asdict(best_min),
    }


def checkpoint_step_and_score(path: Path) -> tuple[int | None, float | None]:
    reward_match = REWARD_PATTERN.search(path.name)
    if reward_match:
        return int(reward_match.group("step")), float(reward_match.group("reward"))
    for pattern in CHECKPOINT_PATTERNS:
        match = pattern.search(path.name)
        if match:
            return int(match.group("step")), None
    return None, None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_checkpoints(log_dir: Path, hash_checkpoints: bool) -> list[CheckpointInfo]:
    checkpoints: list[CheckpointInfo] = []
    for path in sorted(log_dir.rglob("*.d3")):
        step, score = checkpoint_step_and_score(path)
        checkpoints.append(
            CheckpointInfo(
                path=path.as_posix(),
                name=path.name,
                step=step,
                score_from_name=score,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path) if hash_checkpoints else None,
            )
        )
    checkpoints.sort(key=lambda c: (-1 if c.step is None else c.step, c.name))
    return checkpoints


def find_log_dirs(root: Path) -> list[Path]:
    if (root / "params.json").is_file() or any(root.glob("*.d3")) or any(root.glob("*.csv")):
        return [root]
    dirs = [p for p in root.rglob("params.json") if p.is_file()]
    log_dirs = sorted({p.parent for p in dirs})
    if log_dirs:
        return log_dirs
    return [root]


def steps_for_metric(points: list[MetricPoint], top_k: int) -> list[int]:
    ordered = sorted(points, key=lambda p: (p.value, p.step, p.epoch), reverse=True)
    steps: list[int] = []
    for point in ordered:
        if point.step not in steps:
            steps.append(point.step)
        if len(steps) >= top_k:
            break
    return steps


def preferred_metric_names(metrics: dict[str, list[MetricPoint]]) -> list[str]:
    preferred = [
        "eval-environment/win",
        "eval-environment/reward",
        "eval-environment/rew",
        "eval-top_3/win",
        "eval-top_3/reward",
        "eval-top_3/rew",
    ]
    return [name for name in preferred if name in metrics]


def checkpoint_by_step(checkpoints: Iterable[CheckpointInfo]) -> dict[int, list[CheckpointInfo]]:
    by_step: dict[int, list[CheckpointInfo]] = {}
    for checkpoint in checkpoints:
        if checkpoint.step is not None:
            by_step.setdefault(checkpoint.step, []).append(checkpoint)
    return by_step


def summarize_log_dir(
    log_dir: Path,
    target_steps: list[int],
    hash_checkpoints: bool,
    keep_top_k: int,
) -> dict[str, Any]:
    metrics = collect_metric_files(log_dir)
    checkpoints = collect_checkpoints(log_dir, hash_checkpoints)
    by_step = checkpoint_by_step(checkpoints)
    metric_summaries = {name: metric_summary(points) for name, points in sorted(metrics.items())}

    keep_reasons: dict[int, list[str]] = {}
    for step in target_steps:
        keep_reasons.setdefault(step, []).append("target-step")

    for metric in preferred_metric_names(metrics):
        for step in steps_for_metric(metrics[metric], max(1, keep_top_k)):
            keep_reasons.setdefault(step, []).append(f"best:{metric}")

    known_steps = [checkpoint.step for checkpoint in checkpoints if checkpoint.step is not None]
    if known_steps:
        keep_reasons.setdefault(max(known_steps), []).append("latest-checkpoint")

    keep_checkpoints: list[dict[str, Any]] = []
    missing_keep_steps: list[int] = []
    for step, reasons in sorted(keep_reasons.items()):
        matches = by_step.get(step, [])
        if not matches:
            missing_keep_steps.append(step)
            continue
        for checkpoint in matches:
            item = asdict(checkpoint)
            item["keep_reasons"] = sorted(set(reasons))
            keep_checkpoints.append(item)

    keep_paths = {item["path"] for item in keep_checkpoints}
    removable = [asdict(c) for c in checkpoints if c.path not in keep_paths]

    params_path = log_dir / "params.json"
    return {
        "log_dir": log_dir.as_posix(),
        "params_json": params_path.as_posix() if params_path.is_file() else None,
        "metric_count": len(metrics),
        "checkpoint_count": len(checkpoints),
        "metrics": metric_summaries,
        "preferred_metrics_found": preferred_metric_names(metrics),
        "checkpoints": [asdict(c) for c in checkpoints],
        "recommended_keep_steps": [
            {"step": step, "reasons": sorted(set(reasons))}
            for step, reasons in sorted(keep_reasons.items())
        ],
        "recommended_keep_checkpoints": keep_checkpoints,
        "missing_keep_steps": missing_keep_steps,
        "removable_checkpoint_candidates": removable,
    }


def main() -> int:
    args = parse_args()
    root = args.run_dir.resolve()
    if not root.exists():
        raise SystemExit(f"run dir does not exist: {root}")
    log_dirs = find_log_dirs(root)
    summary = {
        "run_dir": root.as_posix(),
        "log_dir_count": len(log_dirs),
        "target_steps": args.target_step,
        "hash_checkpoints": args.hash_checkpoints,
        "keep_top_k": args.keep_top_k,
        "log_dirs": [
            summarize_log_dir(d, args.target_step, args.hash_checkpoints, args.keep_top_k)
            for d in log_dirs
        ],
    }
    text = json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
