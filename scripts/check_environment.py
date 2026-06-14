#!/usr/bin/env python3
"""Environment and artifact preflight checks for the public PLfB release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

EXPECTED_FINAL_SHA256 = "625a387b8701295838ff10beb631dd5052d1bb8eafb9b01af77947164119cd67"
EXPECTED_PARAMS_SHA256 = "730e032e8252ffee361e105fa75a12c7be710ff325356024cbe76f2330af2258"
EXPECTED_MERGED_SHA256 = "882dfa8171601f878078874df2c1a52eb72445b08c58a5adbb3d6eaa71734efc"
EXPECTED_STRICT_FIRST_STAGE_SHA256 = "9f092d090df9797b4566e397529969ffe1f6f3d453c92c41f745fbc29c781851"

FINAL_MODEL = Path("artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3")
FINAL_PARAMS = Path("artifacts/football/final_uri_best/params.json")
MERGED_CACHE = Path("football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz")
STRICT_FIRST_STAGE = Path("artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3")

CORE_IMPORTS = [
    "numpy",
    "yaml",
    "openai",
]
FOOTBALL_IMPORTS = [
    "torch",
    "d3rlpy",
    "gfootball.env",
    "gfootball_engine",
    "kaggle_environments",
]
LLM_REGEN_IMPORTS = [
    "llama_index",
]


@dataclass
class Check:
    name: str
    status: str
    detail: str


class State:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.checks.append(Check(name, "ok", detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.checks.append(Check(name, "warn", detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.checks.append(Check(name, "fail", detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def print_text(self) -> None:
        for check in self.checks:
            marker = {"ok": "[ok]", "warn": "[warn]", "fail": "[fail]"}[check.status]
            detail = f": {check.detail}" if check.detail else ""
            print(f"{marker} {check.name}{detail}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_import(module: str, state: State, required: bool) -> None:
    if importlib.util.find_spec(module) is None:
        if required:
            state.fail(f"import {module}", "module not found")
        else:
            state.warn(f"import {module}", "module not found; needed only for optional stages")
        return
    state.ok(f"import {module}")


def check_file_sha(root: Path, rel: Path, expected: str, state: State) -> None:
    path = root / rel
    if not path.is_file():
        state.fail(str(rel), f"missing at {path}")
        return
    actual = sha256_file(path)
    if actual == expected:
        state.ok(str(rel), actual)
    else:
        state.fail(str(rel), f"sha256 mismatch: {actual}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts")))
    parser.add_argument("--quick", action="store_true", help="Run package/import checks only.")
    parser.add_argument("--strict-football", action="store_true", help="Require the GFootball/d3rlpy runtime imports.")
    parser.add_argument("--llm-regeneration", action="store_true", help="Warn-check packages used only for LLM-backed regeneration.")
    parser.add_argument("--check-artifacts", action="store_true", help="Verify downloaded release artifact checksums.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    state = State()
    repo_root = args.repo_root.resolve()
    artifact_root = args.artifact_root.resolve()

    py = sys.version_info
    if py.major == 3 and py.minor >= 10:
        state.ok("python", platform.python_version())
    else:
        state.fail("python", f"expected Python >=3.10 for environment-universal.yml, got {platform.python_version()}")

    if (repo_root / "football_llm").is_dir() and (repo_root / "plfb-uri").is_dir():
        state.ok("repo layout", str(repo_root))
    else:
        state.fail("repo layout", f"missing football_llm or plfb-uri under {repo_root}")

    for module in CORE_IMPORTS:
        check_import(module, state, required=True)
    for module in FOOTBALL_IMPORTS:
        check_import(module, state, required=args.strict_football)
    if args.llm_regeneration:
        for module in LLM_REGEN_IMPORTS:
            check_import(module, state, required=False)

    if args.check_artifacts:
        check_file_sha(artifact_root, FINAL_MODEL, EXPECTED_FINAL_SHA256, state)
        check_file_sha(artifact_root, FINAL_PARAMS, EXPECTED_PARAMS_SHA256, state)
        check_file_sha(artifact_root, MERGED_CACHE, EXPECTED_MERGED_SHA256, state)
        check_file_sha(artifact_root, STRICT_FIRST_STAGE, EXPECTED_STRICT_FIRST_STAGE_SHA256, state)

    if args.json:
        print(json.dumps({"checks": [asdict(c) for c in state.checks]}, indent=2, sort_keys=True))
    else:
        state.print_text()
    return 1 if state.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
