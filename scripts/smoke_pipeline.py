#!/usr/bin/env python3
"""Smoke checks for the public PLfB football release."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import py_compile
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EXPECTED_FINAL_SHA256 = "625a387b8701295838ff10beb631dd5052d1bb8eafb9b01af77947164119cd67"
EXPECTED_STRICT_FIRST_STAGE_SHA256 = "9f092d090df9797b4566e397529969ffe1f6f3d453c92c41f745fbc29c781851"
EXPECTED_0204_MERGED_SHA256 = "882dfa8171601f878078874df2c1a52eb72445b08c58a5adbb3d6eaa71734efc"
FINAL_MODEL_REL = Path("artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3")
STRICT_FIRST_STAGE_MODEL_REL = Path("artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3")
PAPER_0204_MERGED_REL = Path("football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz")

SOURCE_FILES = [
    "plfb-uri/main_understanding.py",
    "plfb-uri/main_introspecting.py",
    "plfb-uri/understanding/book_reader.py",
    "plfb-uri/understanding/prompt_templete.py",
    "football_llm/book_scripts/filter_books_v2.py",
    "football_llm/book_scripts/filter_pi_t_obj.py",
    "football_llm/llm/generate_main.py",
    "football_llm/llm/imaginary_data_generation.py",
    "football_llm/llm/utils/index.py",
    "football_llm/llm/utils/llama_index_compat.py",
    "football_llm/llm/utils/obs2text.py",
    "football_llm/llm/utils/imaginary.py",
    "football_llm/retrieval/retrieval_module.py",
    "football_llm/retrieval/state_filed_retrieval.py",
    "football_llm/learning/data_loader.py",
    "football_llm/learning/uncertainty_predictor.py",
    "football_llm/learning/imaginaryRL_v2.py",
    "football_llm/learning/load_and_eval_v2.py",
    "football_llm/learning/utils.py",
    "scripts/normalize_retrieval_context.py",
    "scripts/summarize_ciql_run.py",
]

PUBLIC_ARTIFACT_DIRS = [
    "book_derived/retrieval",
    "book_derived/uri_text_results",
    "book_derived/v4-gpt-3.5-turbo-1106-level-strict",
    "football/generated_llm_results",
    "football/imaginary_dataset_0204",
    "football/imaginary_dataset_0204/merged_data",
    "artifacts/football/final_uri_best",
    "artifacts/football/strict_repro_first_stage_ba0e02e",
]

SOURCE_DATA_DIRS = [
    "generated_llm_results",
    "imaginary_dataset_0204",
    "imaginary_dataset_0204/merged_data",
]

REPORT_FILES = [
    "reports/final_ciql_release_report.json",
]

API_COMPAT_FILES = [
    "football_llm/book_scripts/filter_pi_t_obj.py",
    "football_llm/book_scripts/utils.py",
    "football_llm/understanding/utils.py",
    "plfb-uri/understanding/utils.py",
    "football_llm/llm/openai_server.py",
    "football_llm/llm/generate_main.py",
    "football_llm/llm/imaginary_data_generation.py",
    "football_llm/llm/utils/openai_query.py",
    "football_llm/llm/utils/openai_compat.py",
    "football_llm/llm/utils/index.py",
    "football_llm/llm/utils/llama_index_compat.py",
    "football_llm/llm_v2/utils/index.py",
    "football_llm/retrieval/retrieval_module.py",
    "football_llm/retrieval/state_filed_retrieval.py",
    "football_llm/retrieval/finetune_retrieval.py",
    "football_llm/d3rlpy/d3rlpy/metrics/utility.py",
    "football_llm/learning/imaginaryRL.py",
    "football_llm/learning/llm_finetune.py",
    "football_llm/llm/algo/baseline_v1_multi_agent.py",
    "football_llm/llm/algo/baseline_v1_single_agent.py",
    "football_llm/llm/gen_finetune_data.py",
    "football_llm/llm/test.py",
    "football_llm/llm_v2/algo/baseline_v1_multi_agent.py",
    "football_llm/llm_v2/algo/baseline_v1_single_agent.py",
    "football_llm/rehearsing/corpus_gen.py",
    "football_llm/rehearsing/main-v2.py",
    "football_llm/rehearsing/main.py",
    "football_llm/retrieval/evaluation.py",
    "football_llm/tictactoe/ttt_gym_player.py",
    "football_llm/ult/test_llama_index.py",
    "plfb-uri/d3rlpy/metrics/utility.py",
]

LEGACY_OPENAI_PATTERNS = [
    "openai." + "ChatCompletion",
    "openai." + "Completion",
    "openai." + "Embedding",
    "ChatCompletion" + ".create",
    "Completion" + ".create",
    "Embedding" + ".create",
    "text-embedding-ada-002",
    'OpenAI(model="gpt-3.5-turbo")',
    "OpenAI(model='gpt-3.5-turbo')",
    'OpenAI(model="gpt-4")',
    "OpenAI(model='gpt-4')",
]

LEGACY_LLAMA_INDEX_PATTERNS = [
    "from llama_index import",
    "from llama_index.",
    "import llama_index",
    "from llama_index.llms import",
    "from llama_index.embeddings import",
    "from llama_index.schema import",
    "from llama_index.readers.base import",
    "from llama_index.prompts import",
    "from llama_index.vector_stores.types import",
    "from llama_index.indices.service_context import",
    "from llama_index.indices.struct_store import",
    "ServiceContext.from_defaults",
    "VectorStoreIndex.from_documents",
    "VectorStoreIndex(nodes,",
]

LLAMA_INDEX_PATTERN_ALLOWLIST = {
    "football_llm/llm/utils/llama_index_compat.py",
    "scripts/smoke_pipeline.py",
}


GFOOTBALL_REQUIRED_ASSETS = [
    "football_llm/setup/football/third_party/gfootball_engine/data/media/objects/stadiums/test/pitchonly.object",
    "football_llm/setup/football/third_party/gfootball_engine/data/media/objects/stadiums/test/test.object",
    "football_llm/setup/football/third_party/gfootball_engine/data/media/objects/stadiums/test/pitch.ase",
    "football_llm/setup/football/third_party/gfootball_engine/data/media/objects/stadiums/test/test.ase",
]

API_COMPAT_SUFFIXES = (".py", ".md", ".sh", ".sbatch", ".yml", ".yaml")


@dataclass(frozen=True)
class PublicHygienePattern:
    pattern: re.Pattern[str]
    label: str


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    files: tuple[str, ...]
    artifact_dirs: tuple[str, ...] = ()
    source_dirs: tuple[str, ...] = ()
    release_files: tuple[str, ...] = ()


PUBLIC_HYGIENE_PATTERNS = [
    PublicHygienePattern(re.compile(r"local_config"), "local private config reference"),
    PublicHygienePattern(re.compile("/" + "scratch/"), "private absolute scratch path"),
    PublicHygienePattern(re.compile("/" + "Users/"), "private absolute macOS path"),
    PublicHygienePattern(re.compile("/" + "home/"), "private absolute home path"),
    PublicHygienePattern(re.compile("K" + "CL"), "institution-specific scheduler wording"),
    PublicHygienePattern(re.compile(r"api\.openai-[a-z0-9-]+\.com"), "private OpenAI-compatible endpoint"),
    PublicHygienePattern(re.compile(r"OPENAI_[A-Z0-9]+_(?:API_KEY|BASE_URL)"), "private OpenAI-compatible environment variable"),
    PublicHygienePattern(re.compile(r"football_llm/data/v[0-9]+-"), "repo-local historical data default"),
    PublicHygienePattern(re.compile(r"football_llm/data(?:/|$)"), "repo-local data default"),
    PublicHygienePattern(re.compile(r"\.\./gfootball_res"), "repo-parent log default"),
    PublicHygienePattern(re.compile(r"hf " + "upload [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"), "maintainer-specific artifact publication command"),
]


MODULE_SPECS = [
    ModuleSpec(
        name="stage-0-environment-and-contract",
        files=(
            "environment-universal.yml",
            "environment.yml",
            "environment-gfootball.yml",
            "requirements.txt",
            "docs/data_release.md",
            "docs/dataset_contract.md",
            "scripts/smoke_pipeline.py",
            "scripts/plfb_common.sh",
            "scripts/download_artifacts.sh",
            "scripts/smoke_stage.sh",
        ),
    ),
    ModuleSpec(
        name="stage-1-understanding",
        files=(
            "plfb-uri/main_understanding.py",
            "plfb-uri/understanding/book_reader.py",
            "plfb-uri/understanding/prompt_templete.py",
            "plfb-uri/configs/conf.yaml",
            "football_llm/book_scripts/filter_books_v2.py",
            "football_llm/book_scripts/filter_pi_t_obj.py",
            "football_llm/book_scripts/prompt_templete.py",
            "scripts/book_understanding.sh",
        ),
        artifact_dirs=(
            "book_derived/v4-gpt-3.5-turbo-1106-level-strict",
            "book_derived/uri_text_results/understanding",
        ),
        source_dirs=(),
    ),
    ModuleSpec(
        name="stage-2-retrieval-and-code-instantiation",
        files=(
            "football_llm/llm/utils/index.py",
            "football_llm/llm/utils/llama_index_compat.py",
            "football_llm/retrieval/retrieval_module.py",
            "football_llm/retrieval/state_filed_retrieval.py",
            "football_llm/retrieval/finetune_retrieval.py",
            "football_llm/llm/config/gen_main_parser.py",
            "scripts/normalize_retrieval_context.py",
            "scripts/prepare_retrieval_context.sh",
        ),
        artifact_dirs=("book_derived/retrieval", "book_derived/uri_text_results/rehearsing"),
        source_dirs=(),
    ),
    ModuleSpec(
        name="stage-3-rehearsing-imagined-trajectories",
        files=(
            "football_llm/llm/generate_main.py",
            "football_llm/llm/imaginary_data_generation.py",
            "football_llm/llm/utils/obs2text.py",
            "football_llm/llm/utils/imaginary.py",
            "football_llm/rehearsing/rehearse.py",
            "scripts/generate_imagined_trajectories.sh",
        ),
        artifact_dirs=(
            "football/generated_llm_results",
            "football/imaginary_dataset_0204",
        ),
        source_dirs=(
            "generated_llm_results",
            "imaginary_dataset_0204",
        ),
    ),
    ModuleSpec(
        name="stage-4-introspecting-uncertainty-rewards",
        files=(
            "plfb-uri/main_introspecting.py",
            "plfb-uri/introspecting/data_loader.py",
            "plfb-uri/introspecting/uncertainty_predictor.py",
            "football_llm/learning/data_loader.py",
            "football_llm/learning/uncertainty_predictor.py",
            "scripts/introspect_uncertainty.sh",
        ),
        artifact_dirs=("football/imaginary_dataset_0204",),
        source_dirs=("imaginary_dataset_0204",),
        release_files=(
            "artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3",
        ),
    ),
    ModuleSpec(
        name="stage-5-ciql-training",
        files=(
            "football_llm/learning/imaginaryRL_v2.py",
            "football_llm/learning/data_loader.py",
            "football_llm/learning/uncertainty_predictor.py",
            "football_llm/d3rlpy/d3rlpy/algos/qlearning/cql.py",
            "football_llm/d3rlpy/d3rlpy/algos/qlearning/torch/cql_impl.py",
            "examples/slurm/train_ciql.sbatch",
            "scripts/train_ciql.sh",
        ),
        artifact_dirs=("football/imaginary_dataset_0204",),
        source_dirs=("imaginary_dataset_0204",),
    ),
    ModuleSpec(
        name="stage-6-final-model-selection-and-evaluation",
        files=(
            "football_llm/learning/load_and_eval_v2.py",
            "football_llm/learning/utils.py",
            "docs/final_ciql_model.md",
            "docs/final_ciql_traceability.json",
            "examples/slurm/eval_ciql.sbatch",
            "scripts/eval_ciql.sh",
            "scripts/summarize_ciql_run.py",
        ),
        release_files=(
            "reports/final_ciql_release_report.json",
            "reports/final_uri_best_eval_log_summary.json",
            "artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3",
        ),
    ),
]

IMPORT_MODULES = [
    "numpy",
    "torch",
    "d3rlpy",
    "gfootball.env",
    "openai",
    "kaggle_environments.envs.football.helpers",
]
OPTIONAL_IMPORTS = [("llama_index.core", "modern llama-index package"), ("llama_index", "legacy llama-index package")]


@dataclass
class CheckState:
    failures: int = 0
    warnings: int = 0

    def ok(self, message: str) -> None:
        print(f"[ok] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[warn] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[fail] {message}")


def readable_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_eval_report(root: Path) -> Path | None:
    candidates = [root / "reports" / "final_ciql_release_report.json", root / "final_ciql_release_report.json"]
    return next((path for path in candidates if path.is_file()), None)


def find_fresh_eval_json(root: Path) -> Path | None:
    candidates = list((root / "reports" / "eval_result").glob("CIQL_eval_res_final_uri_best_40_*.json"))
    candidates += list((root / "eval_result").glob("CIQL_eval_res_final_uri_best_40_*.json"))
    return sorted(candidates)[-1] if candidates else None


def add_pythonpath(repo_root: Path) -> None:
    paths = [repo_root / "football_llm" / "d3rlpy", repo_root / "football_llm" / "setup" / "football", repo_root / "football_llm", repo_root / "plfb-uri"]
    for path in reversed(paths):
        text = str(path.resolve())
        sys.path = [entry for entry in sys.path if entry != text]
        sys.path.insert(0, text)


def iter_source_check_files() -> list[str]:
    files = set(SOURCE_FILES)
    for spec in MODULE_SPECS:
        files.update(rel for rel in spec.files if rel.endswith(".py"))
    return sorted(files)


def check_source(repo_root: Path, state: CheckState) -> None:
    for rel in iter_source_check_files():
        path = repo_root / rel
        if not path.is_file():
            state.fail(f"missing source file: {rel}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            state.fail(f"syntax check failed: {rel}: {exc.msg}")
        else:
            state.ok(f"syntax check: {rel}")
    for rel in GFOOTBALL_REQUIRED_ASSETS:
        path = repo_root / rel
        if path.is_file() and path.stat().st_size > 0:
            state.ok(f"GFootball runtime asset present: {rel}")
        else:
            state.fail(f"missing GFootball runtime asset: {rel}")


def check_layout(artifact_root: Path, dataset_root: Path | None, release_root: Path | None, state: CheckState) -> None:
    if dataset_root:
        for rel in SOURCE_DATA_DIRS:
            path = dataset_root / rel
            if path.is_dir():
                state.ok(f"source data directory present: {path}")
            else:
                state.fail(f"missing source data directory: {path}")
    else:
        for rel in PUBLIC_ARTIFACT_DIRS:
            path = artifact_root / rel
            if path.is_dir():
                state.ok(f"artifact directory present: {path}")
            else:
                state.fail(f"missing artifact directory: {path}")

    report_root = release_root or artifact_root
    for rel in REPORT_FILES:
        path = report_root / rel
        if path.is_file():
            state.ok(f"artifact file present: {path}")
        else:
            state.fail(f"missing artifact file: {path}")

    model_path = report_root / FINAL_MODEL_REL
    if not model_path.is_file():
        state.fail(f"missing final model: {model_path}")
        return
    digest = sha256_file(model_path)
    if digest != EXPECTED_FINAL_SHA256:
        state.fail(f"final model sha256 mismatch: {digest}")
    else:
        state.ok(f"final model sha256 ok: {readable_size(model_path.stat().st_size)}")

    strict_first_stage_path = report_root / STRICT_FIRST_STAGE_MODEL_REL
    if not strict_first_stage_path.is_file():
        state.fail(f"missing strict first-stage model: {strict_first_stage_path}")
        return
    first_stage_digest = sha256_file(strict_first_stage_path)
    if first_stage_digest != EXPECTED_STRICT_FIRST_STAGE_SHA256:
        state.fail(f"strict first-stage model sha256 mismatch: {first_stage_digest}")
    else:
        state.ok(f"strict first-stage model sha256 ok: {readable_size(strict_first_stage_path.stat().st_size)}")

    merged_path = report_root / PAPER_0204_MERGED_REL
    if not merged_path.is_file():
        state.fail(f"missing paper-aligned 0204 merged cache: {merged_path}")
        return
    merged_digest = sha256_file(merged_path)
    if merged_digest != EXPECTED_0204_MERGED_SHA256:
        state.fail(f"paper-aligned 0204 merged cache sha256 mismatch: {merged_digest}")
    else:
        state.ok(f"paper-aligned 0204 merged cache sha256 ok: {readable_size(merged_path.stat().st_size)}")


def check_eval_report(root: Path, state: CheckState) -> None:
    report_path = existing_eval_report(root)
    if not report_path:
        state.fail(f"missing final_ciql_release_report.json under {root}")
        return
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    final_model = report.get("final_model", {})
    historical = report.get("historical_training_log_metrics", {})
    fresh = report.get("fresh_validation_eval", {})
    if final_model.get("sha256") == EXPECTED_FINAL_SHA256:
        state.ok("release report final model checksum matches expected")
    else:
        state.fail("release report final model checksum is missing or mismatched")
    env_metrics = historical.get("eval-environment", {}).get("checkpoint_step_48000", {})
    win = env_metrics.get("win", {}).get("value")
    if win is not None:
        state.ok(f"historical eval-environment win at step 48000: {win:.4f}")
    else:
        state.fail("historical eval-environment win metric missing")
    fresh_summary = fresh.get("summary", {})
    if "win" in fresh_summary:
        state.ok(f"fresh validation win: {fresh_summary['win']:.4f}")
    else:
        state.warn("fresh validation summary missing from release report")

    fresh_path = find_fresh_eval_json(root)
    if fresh_path:
        with fresh_path.open("r", encoding="utf-8") as handle:
            fresh_json = json.load(handle)
        missing = [key for key in ("win", "draw", "lose", "rew") if key not in fresh_json]
        if missing:
            state.fail(f"fresh eval JSON missing keys: {', '.join(missing)}")
        else:
            state.ok(f"fresh eval JSON present: {fresh_path}")
    else:
        state.warn("fresh eval JSON not found; release report summary is still usable")



def artifact_data_root(artifact_root: Path, dataset_root: Path | None) -> Path:
    return dataset_root if dataset_root else artifact_root / "football"


def check_jsonl_sample(path: Path, required_key: str, state: CheckState) -> None:
    if not path.is_file():
        state.fail(f"missing JSONL file: {path}")
        return
    if path.stat().st_size == 0:
        state.fail(f"empty JSONL file: {path}")
        return
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
    try:
        record = json.loads(first)
    except json.JSONDecodeError as exc:
        state.fail(f"invalid JSONL sample in {path}: {exc}")
        return
    value = record.get(required_key)
    if value:
        state.ok(f"JSONL sample ok: {path}")
    else:
        state.fail(f"JSONL sample missing nonempty {required_key!r}: {path}")



def check_npz_readable(path: Path, state: CheckState, label: str) -> set[str]:
    if not path.is_file():
        state.fail(f"missing NPZ file: {path}")
        return set()
    if path.stat().st_size == 0:
        state.fail(f"empty NPZ file: {path}")
        return set()
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            names = set(archive.namelist())
    except zipfile.BadZipFile as exc:
        state.fail(f"unreadable NPZ zip container: {path}: {exc}")
        return set()
    if bad_member:
        state.fail(f"NPZ CRC check failed in {path}: {bad_member}")
        return set()
    if not names:
        state.fail(f"NPZ archive has no members: {path}")
        return set()
    state.ok(f"{label} readable: {path.name}")
    return names


def check_npz_keys(path: Path, required_keys: set[str], state: CheckState) -> None:
    names = check_npz_readable(path, state, "NPZ")
    if not names:
        return
    expected = {f"{key}.npy" for key in required_keys}
    missing = sorted(expected - names)
    if missing:
        state.fail(f"NPZ missing keys in {path}: {', '.join(missing)}")
    else:
        state.ok(f"NPZ schema ok: {path.name}")


def check_data_contract(artifact_root: Path, dataset_root: Path | None, release_root: Path | None, state: CheckState) -> None:
    data_root = artifact_data_root(artifact_root, dataset_root)
    retrieval_root = (dataset_root / "retrieval") if dataset_root else (artifact_root / "book_derived" / "retrieval")
    for name in ("policy", "reward", "transition"):
        check_jsonl_sample(retrieval_root / name / f"{name}.jsonl", "code", state)

    imaginary_root = data_root / "imaginary_dataset_0204"
    if not imaginary_root.is_dir():
        state.fail(f"missing imaginary data root: {imaginary_root}")
    else:
        samples = sorted(path for path in imaginary_root.glob("no_*.npz") if path.is_file())[:3]
        if not samples:
            state.fail(f"no raw 0204 imagined NPZ files under {imaginary_root}")
        required_imaginary = {
            "start_point",
            "current_obs",
            "gt_next_obs",
            "gt_next_actions",
            "gt_reward",
            "gt_dense_reward",
            "gt_done",
            "im_next_obs",
            "im_next_actions",
            "im_reward",
            "im_dense_reward",
            "im_done",
            "policy_code",
            "reward_code",
            "transition_code",
            "gen_times",
        }
        for sample in samples:
            check_npz_keys(sample, required_imaginary, state)

    merged_path = data_root / Path("imaginary_dataset_0204/merged_data") / PAPER_0204_MERGED_REL.name
    check_npz_keys(
        merged_path,
        {"obs", "action", "reward", "dense_rewards", "next_obs", "done", "img_obs_dict", "end_epi_list"},
        state,
    )
    if merged_path.is_file():
        digest = sha256_file(merged_path)
        if digest == EXPECTED_0204_MERGED_SHA256:
            state.ok("paper-aligned 0204 merged cache checksum ok")
        else:
            state.fail(f"paper-aligned 0204 merged cache checksum mismatch: {digest}")

    check_eval_report(release_root or artifact_root, state)


def check_imports(repo_root: Path, strict: bool, state: CheckState) -> None:
    add_pythonpath(repo_root)
    for module_name in IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            message = f"import failed: {module_name}: {exc}"
            if strict:
                state.fail(message)
            else:
                state.warn(message)
        else:
            state.ok(f"import ok: {module_name}")

    imported_optional = False
    for module_name, label in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            state.warn(f"optional import failed: {label}: {exc}")
        else:
            imported_optional = True
            state.ok(f"optional import ok: {label}")
    if strict and not imported_optional:
        state.warn("no llama-index package variant could be imported; LLM regeneration checks need the general LLM environment")


def check_repo_path(repo_root: Path, rel: str, state: CheckState) -> None:
    path = repo_root / rel
    if path.is_file():
        state.ok(f"module file present: {rel}")
        if path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                state.fail(f"syntax check failed: {rel}: {exc.msg}")
            else:
                state.ok(f"syntax check: {rel}")
    elif path.is_dir():
        state.ok(f"module directory present: {rel}")
    else:
        state.fail(f"missing module path: {rel}")


def check_data_dir(root: Path, rel: str, state: CheckState) -> None:
    path = root / rel
    if path.is_dir():
        state.ok(f"module data directory present: {path}")
    else:
        state.fail(f"missing module data directory: {path}")


def check_release_file(root: Path, rel: str, state: CheckState) -> None:
    path = root / rel
    if not path.is_file():
        state.fail(f"missing release file: {path}")
        return
    state.ok(f"release file present: {path}")
    if rel == str(FINAL_MODEL_REL):
        digest = sha256_file(path)
        if digest == EXPECTED_FINAL_SHA256:
            state.ok(f"release final model checksum ok: {readable_size(path.stat().st_size)}")
        else:
            state.fail(f"release final model checksum mismatch: {digest}")
    elif rel == str(STRICT_FIRST_STAGE_MODEL_REL):
        digest = sha256_file(path)
        if digest == EXPECTED_STRICT_FIRST_STAGE_SHA256:
            state.ok(f"release strict first-stage checksum ok: {readable_size(path.stat().st_size)}")
        else:
            state.fail(f"release strict first-stage checksum mismatch for {rel}: {digest}")


def iter_api_scan_files(repo_root: Path) -> Iterable[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        for rel in API_COMPAT_FILES:
            yield repo_root / rel
        return

    for rel in result.stdout.splitlines():
        if rel == "scripts/smoke_pipeline.py":
            continue
        path = repo_root / rel
        if path.suffix in API_COMPAT_SUFFIXES:
            yield path


def check_train_ciql_cache_override_dry_run(repo_root: Path, state: CheckState) -> None:
    sentinel = "/tmp/plfb-smoke-merged-cache.npz"
    env = os.environ.copy()
    env["PLFB_REPO_ROOT"] = str(repo_root)
    env["PLFB_MERGED_DATA_CACHE_FILE"] = sentinel
    result = subprocess.run(
        ["bash", "scripts/train_ciql.sh", "--smoke", "--dry-run"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        state.fail(f"train_ciql dry-run failed while checking merged cache override: {result.stderr.strip()}")
        return
    if f"PLFB_MERGED_DATA_CACHE_FILE={sentinel}" not in result.stdout:
        state.fail("train_ciql dry-run did not expose PLFB_MERGED_DATA_CACHE_FILE override")
    else:
        state.ok("train_ciql dry-run recognizes PLFB_MERGED_DATA_CACHE_FILE override")
    if f"--merged_data_cache_file {sentinel}" not in result.stdout:
        state.fail("train_ciql dry-run did not forward PLFB_MERGED_DATA_CACHE_FILE to --merged_data_cache_file")
    else:
        state.ok("train_ciql dry-run forwards exact merged cache to CLI")


def check_train_ciql_uncertainty_override_dry_run(repo_root: Path, state: CheckState) -> None:
    sentinel = "/tmp/plfb-smoke-uncertainty-model.d3"
    env = os.environ.copy()
    env["PLFB_REPO_ROOT"] = str(repo_root)
    env["PLFB_UNCERTAINTY_MODEL_PATH"] = sentinel
    result = subprocess.run(
        ["bash", "scripts/train_ciql.sh", "--smoke", "--dry-run"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        state.fail(f"train_ciql dry-run failed while checking uncertainty override: {result.stderr.strip()}")
        return
    if f"uncertainty_model_path={sentinel}" not in result.stdout:
        state.fail("train_ciql dry-run did not expose PLFB_UNCERTAINTY_MODEL_PATH override")
    else:
        state.ok("train_ciql dry-run recognizes PLFB_UNCERTAINTY_MODEL_PATH override")
    if f"--uncertainty_model_path {sentinel}" not in result.stdout:
        state.fail("train_ciql dry-run did not forward PLFB_UNCERTAINTY_MODEL_PATH to --uncertainty_model_path")
    else:
        state.ok("train_ciql dry-run forwards exact uncertainty model to CLI")



def check_introspect_uncertainty_overrides_dry_run(repo_root: Path, state: CheckState) -> None:
    cache_sentinel = "/tmp/plfb-smoke-stage4-cache.npz"
    model_sentinel = "/tmp/plfb-smoke-stage4-uncertainty-model.d3"
    env = os.environ.copy()
    env["PLFB_REPO_ROOT"] = str(repo_root)
    env["PLFB_MERGED_DATA_CACHE_FILE"] = cache_sentinel
    env["PLFB_UNCERTAINTY_MODEL_PATH"] = model_sentinel
    result = subprocess.run(
        ["bash", "scripts/introspect_uncertainty.sh", "--smoke", "--dry-run"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        state.fail(f"introspect_uncertainty dry-run failed while checking overrides: {result.stderr.strip()}")
        return
    if f"--merged_data_cache_file {cache_sentinel}" not in result.stdout:
        state.fail("introspect_uncertainty dry-run did not forward PLFB_MERGED_DATA_CACHE_FILE")
    else:
        state.ok("introspect_uncertainty dry-run forwards exact merged cache to CLI")
    if f"--uncertainty_model_path {model_sentinel}" not in result.stdout:
        state.fail("introspect_uncertainty dry-run did not forward PLFB_UNCERTAINTY_MODEL_PATH")
    else:
        state.ok("introspect_uncertainty dry-run forwards exact uncertainty model to CLI")


def check_generate_imagined_sampled_data_dry_run(repo_root: Path, state: CheckState) -> None:
    work_root = "/tmp/plfb-smoke-stage3-work"
    expected_sampled = f"{work_root}/sampled_data"
    env = os.environ.copy()
    env["PLFB_REPO_ROOT"] = str(repo_root)
    env["PLFB_WORK_ROOT"] = work_root
    env.pop("PLFB_SAMPLED_DATA_PATH", None)
    result = subprocess.run(
        ["bash", "scripts/generate_imagined_trajectories.sh", "--dry-run", "--number", "7"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        state.fail(f"generate_imagined_trajectories dry-run failed: {result.stderr.strip()}")
        return
    if f"PLFB_SAMPLED_DATA_PATH={expected_sampled}" not in result.stdout:
        state.fail("generate_imagined_trajectories dry-run did not default sampled data under PLFB_WORK_ROOT")
    else:
        state.ok("generate_imagined_trajectories dry-run defaults sampled data under PLFB_WORK_ROOT")
    if "football_llm/llm/sampled_data" in result.stdout:
        state.fail("generate_imagined_trajectories dry-run still exposes legacy repo-local sampled-data path")
    if "--number 7" not in result.stdout:
        state.fail("generate_imagined_trajectories dry-run did not forward extra generate_main.py args")
    else:
        state.ok("generate_imagined_trajectories dry-run forwards extra generate_main.py args")


def check_train_ciql_comment_guard_dry_run(repo_root: Path, state: CheckState) -> None:
    env = os.environ.copy()
    env["PLFB_REPO_ROOT"] = str(repo_root)
    env["PLFB_TRAIN_COMMENT"] = "final-inferred-cache-full-200k-runtime"
    result = subprocess.run(
        ["bash", "scripts/train_ciql.sh", "--smoke", "--dry-run"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        state.fail("train_ciql dry-run accepted a too-long PLFB_TRAIN_COMMENT")
        return
    if "PLFB_TRAIN_COMMENT" not in result.stderr or "d3rlpy filesystem name limits" not in result.stderr:
        state.fail("train_ciql long-comment guard returned an unexpected error")
    else:
        state.ok("train_ciql rejects too-long PLFB_TRAIN_COMMENT before d3rlpy creates logdirs")

def check_api_compat(repo_root: Path, state: CheckState) -> None:
    scanned = 0
    for path in iter_api_scan_files(repo_root):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [pattern for pattern in LEGACY_OPENAI_PATTERNS if pattern in text]
        llama_found = []
        if path.suffix == ".py" and rel.as_posix() not in LLAMA_INDEX_PATTERN_ALLOWLIST:
            llama_found = [pattern for pattern in LEGACY_LLAMA_INDEX_PATTERNS if pattern in text]
        hygiene = [rule.label for rule in PUBLIC_HYGIENE_PATTERNS if rule.pattern.search(text)]
        scanned += 1
        if found:
            state.fail(f"legacy OpenAI API/model pattern in {rel}: {', '.join(found)}")
        if llama_found:
            state.fail(f"legacy LlamaIndex API pattern in {rel}: {', '.join(llama_found)}")
        if hygiene:
            state.fail(f"public hygiene issue in {rel}: {', '.join(hygiene)}")
    state.ok(f"OpenAI/LlamaIndex API and public hygiene scan: {scanned} tracked text files")


def check_module_map(repo_root: Path, artifact_root: Path, dataset_root: Path | None, release_root: Path | None, state: CheckState) -> None:
    runbook_path = repo_root / "docs/paper_module_operations.md"
    if runbook_path.is_file():
        runbook_text = runbook_path.read_text(encoding="utf-8", errors="ignore")
    else:
        runbook_text = ""
        state.fail(f"missing paper module operations runbook: {runbook_path}")

    documented_files: set[str] = set()
    for spec in MODULE_SPECS:
        state.ok(f"module start: {spec.name}")
        for rel in spec.files:
            check_repo_path(repo_root, rel, state)
            if rel in runbook_text:
                documented_files.add(rel)
            else:
                state.fail(f"paper module runbook missing file mapping for {rel}")
        data_root = dataset_root or artifact_root
        data_dirs = spec.source_dirs if dataset_root else spec.artifact_dirs
        for rel in data_dirs:
            check_data_dir(data_root, rel, state)
        for rel in spec.release_files:
            check_release_file(release_root or artifact_root, rel, state)
    if documented_files:
        state.ok(f"paper module runbook covers {len(documented_files)} module files")
    check_api_compat(repo_root, state)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "source", "layout", "data-contract", "imports", "eval-report", "module-map", "api"], default="all")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("PLFB_ARTIFACT_ROOT", "plfb_artifacts")))
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional lower-level source-style football data root for new experiments.")
    parser.add_argument("--release-root", type=Path, default=None, help="Optional release root containing artifacts/ and reports/.")
    parser.add_argument("--strict-imports", action="store_true", help="Fail instead of warn when runtime imports are missing.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    state = CheckState()
    repo_root = args.repo_root.resolve()
    artifact_root = args.artifact_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    release_root = args.release_root.resolve() if args.release_root else None

    if args.mode in ("all", "source"):
        check_source(repo_root, state)
    if args.mode in ("all", "layout"):
        check_layout(artifact_root, dataset_root, release_root, state)
    if args.mode in ("all", "eval-report"):
        check_eval_report(release_root or artifact_root, state)
    if args.mode in ("all", "data-contract"):
        check_data_contract(artifact_root, dataset_root, release_root, state)
    if args.mode in ("all", "imports"):
        check_imports(repo_root, args.strict_imports, state)
    if args.mode in ("all", "module-map"):
        check_module_map(repo_root, artifact_root, dataset_root, release_root, state)
    if args.mode == "api":
        check_api_compat(repo_root, state)
        check_train_ciql_cache_override_dry_run(repo_root, state)
        check_train_ciql_uncertainty_override_dry_run(repo_root, state)
        check_introspect_uncertainty_overrides_dry_run(repo_root, state)
        check_generate_imagined_sampled_data_dry_run(repo_root, state)
        check_train_ciql_comment_guard_dry_run(repo_root, state)

    print(f"[summary] failures={state.failures} warnings={state.warnings}")
    return 1 if state.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
