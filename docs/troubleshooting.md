# Troubleshooting

## d3rlpy pulls an incompatible Gym version

Install the vendored d3rlpy fork with `--no-deps`:

```bash
python -m pip install --no-deps -e football_llm/d3rlpy
```

This keeps the `gym==0.21.0` version from `environment-universal.yml` or `environment-gfootball.yml` instead of letting upstream d3rlpy metadata install a newer Gym release.

## `pydantic` import errors during evaluation

Use `pydantic<2`. The evaluation path imports parser modules that expect the pydantic v1 API.

## Which environment should I use?

Use `environment-universal.yml` for public paper-pipeline reproduction, CIQL retraining/evaluation, and smoke checks. It is Python 3.10 and pip-first for Python packages, but it includes conda-forge native dependencies for GFootball/SDL/Boost. Use `environment.yml` for defaults-only lightweight book/data work when football runtime is not needed. The release-maintainer publication environment is not needed for normal reproduction. If you only evaluate or retrain the released football artifacts, LlamaIndex is not used at runtime. Kaggle football helper enums are required by the retained evaluation postprocess path and are included in the universal environment.

## Kaggle helper dependency fails to build

`kaggle-environments==1.14.1` is required for final football evaluation and CIQL retraining/evaluation because the rule-based postprocess imports Kaggle football helper enums. It builds the `vec-noise` transitive package from source, so make sure the active environment has `wheel`, `setuptools`, `cmake`, and a C++ compiler available before installing. If your platform still fails to build it, create `environment-gfootball.yml` or install the same pinned package in an isolated Python 3.8 or Python 3.10 environment and run the football evaluation there.

## Conda cache or home quota errors

On shared systems, `conda env create` may fail before solving if the default package cache or environment directory is on a small home filesystem, or if a constrained login shell cannot load conda-forge repodata for the football environments. Point conda at a writable work directory before creating the environment:

```bash
mkdir -p /path/to/work/conda-pkgs /path/to/work/conda-envs
export CONDA_PKGS_DIRS=/path/to/work/conda-pkgs
export CONDA_ENVS_PATH=/path/to/work/conda-envs
conda env create -f environment-universal.yml
```

A solver dry-run can be used to validate the conda-level dependencies without installing packages. Prefer `micromamba` or `mamba` for the football environments:

```bash
micromamba create --dry-run -y -n plfb-universal -f environment-universal.yml
micromamba create --dry-run -y -n plfb-gfootball -f environment-gfootball.yml
```

If conda classic is killed while collecting conda-forge repodata, retry with `micromamba`, `mamba`, or libmamba before changing package pins; that failure is a solver resource issue, not evidence that the PLfB code is broken. If you still use conda classic, keep caches and environments on a writable work directory:

```bash
CONDA_PKGS_DIRS=/path/to/work/conda-pkgs \
CONDA_ENVS_PATH=/path/to/work/conda-envs \
conda env create -f environment-universal.yml --dry-run
```

## PyTorch wheel selection

The universal environment defaults to the public `torch==2.0.1` pip wheel for portable smoke and reproduction runs. If you need CPU-only or CUDA-specific wheels, replace the torch pip line using the official PyTorch selector before creating the environment; do not add private package indexes.

## GFootball native extension fails to import

Rebuild Google Research Football inside the active environment:

```bash
python -m pip install -e football_llm/setup/football
```

If your platform requires OpenGL or SDL development headers, install them with your platform package manager before the editable GFootball install. `environment-universal.yml` and `environment-gfootball.yml` include conda-forge Boost/SDL packages for Linux users; use `micromamba`/`mamba` if conda classic cannot solve those packages, or use platform native packages plus the defaults-only lightweight environment for non-football stages.

## Headless rendering or BMP warnings

Headless football evaluation may emit image-loading warnings for team logos. These warnings are not necessarily fatal if episodes continue and evaluation results are produced.


## `File name too long` when CIQL starts

d3rlpy builds an experiment directory name from the training comment and the main CIQL controls. Keep `PLFB_TRAIN_COMMENT` to 24 characters or fewer; the wrapper rejects longer values before d3rlpy creates log directories. Use a compact label such as `alpha-test2`, `ciqlresume`, or `seed0`, and keep the detailed run description in your scheduler output path or external notes instead.

## Hugging Face download is slow

Enable `hf_transfer` before downloading the public artifacts:

```bash
python -m pip install "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1
bash scripts/download_artifacts.sh
```


## LlamaIndex version mismatch

The retained LLM/RAG scripts were written against the pre-0.10 LlamaIndex API, and the public universal and lightweight Python 3.8 environments pin `llama-index>=0.9,<0.10` for reproducibility. The final football CIQL evaluation and retraining paths do not require LlamaIndex when using the released artifacts.

For new LLM-stage development, use a separate modern environment and migrate old `ServiceContext` imports to the current `llama_index.core.Settings` / `llama_index.core` APIs before running generation at scale.

## OpenAI model selection

LLM-backed stages default to `PLFB_OPENAI_CHAT_MODEL=gpt-4o-mini` and `PLFB_OPENAI_EMBEDDING_MODEL=text-embedding-3-small`. Override them, and optionally `PLFB_OPENAI_ACTION_MODEL`, `PLFB_OPENAI_CODE_MODEL`, `PLFB_OPENAI_FILTER_MODEL`, or `PLFB_OPENAI_AGG_MODEL`, if your provider or experiment requires another current model. Stage 3 defaults to `PLFB_USE_OPENAI_COMPAT_CLIENT=1`, which honors `OPENAI_BASE_URL`; set it to `0` for the SDK-only path.
