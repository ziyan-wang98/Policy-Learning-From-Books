# Final CIQL Model

The selected public model is the paper-aligned football CIQL checkpoint:

```text
artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3
```

| Field | Value |
| --- | --- |
| Algorithm | CIQL |
| `alpha` | `60.0` |
| `target_value` | `0.02` |
| `ent_target_coef` | `0.8` |
| `coef_t` / `coef_r` | `0.5` / `0.5` |
| `obs_stack_num` | `4` |
| Action replacement strategy | `all_replaced` |

Evaluate it with:

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
export PLFB_WORK_ROOT=$PWD/runs
bash scripts/eval_ciql.sh --dry-run
bash scripts/eval_ciql.sh
```

This checkpoint is the released final policy used by the public evaluation wrapper. Use `scripts/eval_ciql.sh` to run evaluation with the downloaded Hugging Face artifacts.
