# Book Processing Pipeline

The public repository keeps the scripts for filtering and deriving knowledge from tutorial text, but it does not include raw book text.

Typical inputs are supplied outside the repository:

```bash
export PLFB_BOOK_JSONL=/path/to/your/book_subset.jsonl
export PLFB_WORK_ROOT=$PWD/runs
export PLFB_UNDERSTANDING_DATA_DIR=$PLFB_WORK_ROOT/understanding_data
export PLFB_UNDERSTANDING_RES_ROOT=football_data
```

## Input JSONL Schema

Each line in `PLFB_BOOK_JSONL` must be a JSON object with top-level `text` and `meta`. The football reader accepts either `meta.title` or the compact metadata form with `meta.short_book_title`:

```json
{"text":"... tutorial text ...","meta":{"title":"How to Watch Soccer - Ruud Gullit"}}
```

```json
{"text":"... tutorial text ...","meta":{"short_book_title":"How to Watch Soccer - Ruud Gullit","publication_date":"","url":""}}
```

The current football reader filters to the public football title allowlist: `Duncan Adams - Football Grounds a Fans' Guide [Retail 9781782814207]`, `The Soccer Goalkeeping Handbook (3rd Edition) - Alex Welsh`, `How to Watch Soccer - Ruud Gullit`, `Mike Smith - The Road to Glory`, and `Football Intelligence`. If your source titles differ, update the allowlist in `plfb-uri/understanding/book_reader.py` or use the retained `book_derived/` artifacts; otherwise Stage 1 can run successfully but produce no football chunks.

## Main Files

- `plfb-uri/main_understanding.py`: preferred URI entry point for extracting Policy, Dynamics, and Reward knowledge.
- `plfb-uri/understanding/book_reader.py`: LLM-backed extraction loop.
- `plfb-uri/understanding/prompt_templete.py`: task prompts for extracted knowledge.
- `football_llm/book_scripts/filter_books_v2.py`: legacy corpus filtering helper.
- `football_llm/book_scripts/filter_pi_t_obj.py`: legacy policy/transition/objective filtering helper.

## Public Artifact Path

The released Hugging Face dataset contains derived artifacts needed by the public pipeline, including retrieval-ready policy, reward, transition, and URI text outputs. If you use these artifacts, you do not need raw book text for final CIQL reproduction.

```bash
export PLFB_ARTIFACT_ROOT=$PWD/plfb_artifacts
python scripts/smoke_pipeline.py --mode layout --artifact-root "$PLFB_ARTIFACT_ROOT"
```

## Regeneration Path

To regenerate from your own text subset, place the JSONL file outside Git and either run `scripts/book_understanding.sh` or pass equivalent Hydra overrides:

```bash
export PLFB_BOOK_JSONL=/path/to/your/book_subset.jsonl
export PLFB_WORK_ROOT=$PWD/runs
export OPENAI_API_KEY=...
export PLFB_UNDERSTANDING_DATA_DIR=$PLFB_WORK_ROOT/understanding_data
export PLFB_UNDERSTANDING_RES_ROOT=football_data
bash scripts/book_understanding.sh --dry-run
bash scripts/book_understanding.sh

# Equivalent lower-level command:
mkdir -p "$PLFB_UNDERSTANDING_DATA_DIR"
cp "$PLFB_BOOK_JSONL" "$PLFB_UNDERSTANDING_DATA_DIR/book_subset.jsonl"
python plfb-uri/main_understanding.py sim_info=fb path=fb \
  path.root="$PWD/plfb-uri" \
  path.data_path="$PLFB_UNDERSTANDING_DATA_DIR" \
  path.book=book_subset.jsonl \
  path.res_root="$PLFB_UNDERSTANDING_RES_ROOT"
```

Expected regenerated output root:

```text
$PLFB_WORK_ROOT/understanding_data/football_data/book_knowledge/
  Policy/
  Dynamics/
  Reward/
```

Do not commit raw text, generated provider responses, or API keys. Keep regenerated outputs under `runs/` or another ignored directory unless preparing a curated public artifact release.
