Perfect—now that you’ve got clean split + filtered outputs, here’s a tidy package for:

1. a **drop-in doc** you can paste into `dpo/docs/M1_dataset_split.md`, and
2. a set of **scripts in `dpo/scripts/`** to (a) reproduce the filtering summary, (b) generate a compact EDA, and (c) spit out a Markdown report you can commit alongside the data.

> heads-up: ensure `dpo/scripts/__init__.py` exists (can be empty) so `python -m dpo.scripts.*` works.

---

# 📄 `dpo/docs/M1_dataset_split.md` (paste this)

````markdown
# M1 — Dataset, Splits, and Pair Filtering (DAS)

## TL;DR
- **Source graphs**: `data/processed.pt` (4223 entries).
- **Splits**: `data/das_split.pt` → **train 4025**, **val 100**, **test 98**.
- **IDs**: no single `backbone_id` in `processed.pt`; instead we canonicalize strings in each entry’s `id_list`
  to **`PDBID_MODEL_CHAIN`** (drop file extensions and hyphen decorations: `3B58_1_B-C-A → 3B58_1_B`).
- **Pairs**: `data/pairs_margin125/dpo_pairs_margin125.json` store `pdb_file`, `winner_seq`, `loser_seq`, metrics.
  We canonicalize `pdb_file` the same way and map to the DAS split.

## Why `das_split.pt` (vs external split_ids)
- `das_split.pt` is the **source of truth** for this repo’s `processed.pt`.  
- External `split_ids/*.txt` are **larger/older supersets** (e.g., test_ids has 235 vs DAS test 98).
  Every DAS test ID appears in the external list, but **137** external IDs aren’t in this `processed.pt`.

## What we actually observed (your latest run)
- Pairs loaded: **37,928**
- After ID fix and “index all aliases in `id_list`”:
  - **train 35,883**, **val 1,103**, **test 928**, **unknown 14**
- Length screening:
  - On full set: **kept 21,745**, **mismatches 16,169**, **unknown 14**  
    (mismatch type is typically **`W==L!=G`**, e.g., winner & loser length 44 vs graph length 46)
  - By split (after `filter_pairs_by_split`):
    - train: **kept 20,811 / 35,883**, mismatches **15,072**
    - val: **kept 505 / 1,103**, mismatches **598**
    - test: **kept 429 / 928**, mismatches **499**

> Interpretation: a large fraction of pairs point to the correct backbone but have **sequence-length off-by-1** (or similar),
> likely due to trimming/alt chain annotations. We keep a **clean subset** where winner/loser == graph length.

## Canonicalization rule (one-liner)
We normalize any identifier or path to **`PDBID_MODEL_CHAIN`**, e.g.:
- `3B58_1_B-C-A` → `3B58_1_B`
- `7M57_1_qq-bb` → `7M57_1_qq`
- `./data/raw/6ZU1_1_AW.pdb` → `6ZU1_1_AW`

This is applied consistently to:
- split checks,
- pair resolution by `pdb_file`,
- dataset indexing.

## Commands to reproduce

### Inspect split + compare to external list
```bash
python -m dpo.debug.inspect_das_split \
  --split_pt data/das_split.pt \
  --processed_pt data/processed.pt \
  --test_ids_file data/split_ids/test_ids_das.txt
````

### Check pairs against DAS (counts + issues)

```bash
python -m dpo.debug.check_pairs_against_split \
  --pairs_path data/pairs_margin125/dpo_pairs_margin125.json \
  --processed_pt data/processed.pt \
  --split_pt data/das_split.pt
# add --allow_test to silence warnings about test-split pairs
```

### Split pairs by DAS, then filter length mismatches

```bash
python -m dpo.scripts.split_and_filter_pairs \
  --pairs_in data/pairs_margin125/dpo_pairs_margin125.json \
  --processed_pt data/processed.pt \
  --split_pt data/das_split.pt \
  --out_dir data/pairs_margin125/by_das
```

This writes:

* `by_das/split/{train,val,test}.jsonl`
* `by_das/clean/{train, val, test}.clean.jsonl` (length-matched)
* `by_das/clean/{*.mismatch.jsonl, *.unknown.jsonl}`

### Summarize the filtered data + quick EDA (next section)

```bash
python -m dpo.scripts.summarize_filtered_pairs \
  --root data/pairs_margin125/by_das \
  --report_md data/pairs_margin125/by_das/report.md

python -m dpo.scripts.eda_metrics \
  --pairs_in data/pairs_margin125/by_das/clean/train.clean.jsonl \
  --out_dir data/pairs_margin125/by_das/eda/train

python -m dpo.scripts.eda_metrics \
  --pairs_in data/pairs_margin125/by_das/clean/val.clean.jsonl \
  --out_dir data/pairs_margin125/by_das/eda/val

python -m dpo.scripts.eda_metrics \
  --pairs_in data/pairs_margin125/by_das/clean/test.clean.jsonl \
  --out_dir data/pairs_margin125/by_das/eda/test
```

