# Paper Traceability

## Selected Final Model

The public release keeps the paper-aligned football CIQL checkpoint:

| Item | Value |
| --- | --- |
| Checkpoint | `artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3` |
| Algorithm | CIQL |
| Main controls | `alpha=60.0`, `target_value=0.02`, `ent_target_coef=0.8`, `coef_t=0.5`, `coef_r=0.5`, `obs_stack_num=4`, `strategy=all_replaced` |

Use `scripts/eval_ciql.sh` to evaluate this checkpoint after downloading the public Hugging Face artifacts. Reported benchmark results are documented in the paper and project page.

## Training Data and First-stage Status

The source-verified 2024-02 imagined dataset cache is public:

`football/imaginary_dataset_0204/merged_data/v3datatrace_real_num=0&extra_real_traj_num=0&obs_stack_num=4&rollout_num=0.npz`


The release includes a public first-stage checkpoint for replay experiments:

`artifacts/football/strict_repro_first_stage_ba0e02e/model_290000.d3`


This checkpoint is included so the public CIQL training wrapper can replay the released workflow from public artifacts.
