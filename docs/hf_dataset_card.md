# Hugging Face Dataset Card Draft

## Dataset Summary

`ziyan98/plfb` contains derived artifacts for reproducing the public Policy Learning from Books football release, including final offline datasets, derived book knowledge artifacts, the selected CIQL checkpoint, and release reports.

## Intended Use

The dataset is intended for research reproduction, model evaluation, and inspection of the released PLfB pipeline artifacts. It is not a raw book corpus.

## Data Sources

The release includes derived outputs produced by the PLfB pipeline. Raw tutorial book text is excluded from the public dataset.

## Sensitive Content and Credentials

The release excludes API keys, provider credentials, private endpoints, personal paths, raw text files, and runtime logs that are not needed for public reproduction.

## Model Artifact

The final football checkpoint is:

```text
artifacts/football/final_uri_best/model_rew_0.5&step_48000.d3
```

See `docs/final_ciql_model.md` for hyperparameters and the released model path.
