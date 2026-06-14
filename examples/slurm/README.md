# Generic SLURM Examples

These files are scheduler templates, not site-specific launch files. Before running, edit:

- `#SBATCH --partition`
- `#SBATCH --account` if your scheduler or accounting setup requires one
- time limits, memory, and GPU type/count
- conda/module activation lines
- `PLFB_REPO_ROOT`, `PLFB_ARTIFACT_ROOT`, and output paths

No institution-specific paths or accounts are required by the code. Each template calls a public `scripts/*.sh` entry point so the scheduler and non-scheduler workflows stay aligned.

## Templates

- `smoke_pipeline.sbatch`: validates source, artifact layout, final checkpoint checksum, reports, imports, and module mapping.
- `book_understanding.sbatch`: runs Stage 1 LLM-backed book understanding from `PLFB_BOOK_JSONL`.
- `generate_imagined_trajectories.sbatch`: runs Stage 3 LLM-backed trajectory generation.
- `eval_ciql.sbatch`: evaluates the released final CIQL checkpoint.
- `train_ciql.sbatch`: launches the retained CIQL training configuration. Set `PLFB_TRAIN_STEPS=20`, `PLFB_STEPS_PER_EPOCH=10`, and `PLFB_EVAL_TRIALS=1` or pass `--smoke` for a short run before full training.
