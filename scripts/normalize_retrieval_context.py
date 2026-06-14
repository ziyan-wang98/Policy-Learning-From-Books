#!/usr/bin/env python3
"""Normalize Stage 1 aggregate JSONL into Stage 2 retrieval context files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUTS = {
    "policy": Path("Policy/best/agg-best.jsonl"),
    "transition": Path("Dynamics/best/agg-best.jsonl"),
    "reward": Path("Reward/multi/agg-level-2.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Stage 1 URI aggregate JSONL records into retained-style "
            "book_derived/retrieval JSONL files with code/code_title fields."
        )
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        required=True,
        help="Root containing Policy/, Dynamics/, and Reward/ aggregate outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root to create policy/, reward/, and transition/ JSONL files under.",
    )
    parser.add_argument("--policy-input", type=Path, default=DEFAULT_INPUTS["policy"])
    parser.add_argument("--reward-input", type=Path, default=DEFAULT_INPUTS["reward"])
    parser.add_argument("--transition-input", type=Path, default=DEFAULT_INPUTS["transition"])
    parser.add_argument("--limit", type=int, default=None, help="Optional max records per stage.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONL files.")
    return parser.parse_args()


def as_lines(value: Any) -> list[str] | None:
    if isinstance(value, list):
        lines = [str(item) for item in value if str(item).strip()]
        return lines or None
    if isinstance(value, str):
        lines = [line for line in value.splitlines() if line.strip()]
        return lines or None
    return None


def candidate_code_fields(record: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    if "code" in record:
        yield "code", record["code"]
    for key, value in record.items():
        if key in {"code", "code_title", "embedding"}:
            continue
        yield key, value


def normalize_record(record: dict[str, Any], source_label: str, line_no: int) -> dict[str, Any] | None:
    for key, value in candidate_code_fields(record):
        lines = as_lines(value)
        if not lines:
            continue
        title = record.get("code_title")
        if not isinstance(title, str) or not title.strip():
            title = key if key != "code" else f"{Path(source_label).stem}:{line_no}"
        return {
            "code": lines,
            "code_title": title,
            "source_stage1_file": source_label,
            "source_stage1_line": line_no,
        }
    return None


def normalize_file(
    input_path: Path,
    output_path: Path,
    limit: int | None,
    dry_run: bool,
    overwrite: bool,
    source_label: str,
) -> int:
    if not input_path.is_file():
        raise FileNotFoundError(f"missing Stage 1 input: {input_path}")
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    normalized: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if limit is not None and len(normalized) >= limit:
                break
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, str):
                value = {input_path.stem: value}
            if not isinstance(value, dict):
                continue
            record = normalize_record(value, source_label, line_no)
            if record is not None:
                normalized.append(record)

    if not normalized:
        raise ValueError(f"no code records found in {input_path}")

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in normalized:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(normalized)


def main() -> int:
    args = parse_args()
    inputs = {
        "policy": args.policy_input,
        "reward": args.reward_input,
        "transition": args.transition_input,
    }
    for stage, rel_input in inputs.items():
        input_path = rel_input if rel_input.is_absolute() else args.stage1_root / rel_input
        output_path = args.output_root / stage / f"{stage}.jsonl"
        try:
            source_label = input_path.relative_to(args.stage1_root).as_posix()
        except ValueError:
            source_label = input_path.name
        count = normalize_file(
            input_path,
            output_path,
            args.limit,
            args.dry_run,
            args.overwrite,
            source_label,
        )
        action = "would write" if args.dry_run else "wrote"
        print(f"{action} {count} {stage} records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
