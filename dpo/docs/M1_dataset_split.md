What’s in processed.pt

processed.pt holds the canonical gRNAde graph store (4223 entries in your dump).

Each entry is a dict with keys like:
- sequence (native RNA sequence for that backbone)
- id_list (list of synonymous backbone identifiers; e.g., 3B58_1_B-C-A, 5WF0_1_B, 7M57_1_n-Y)
- per-structure features: coords_list, sec_struct_list, sasa_list, …

There is no single backbone_id field; the ID must be derived from id_list. We canonicalize every string to the form:
PDBID_MODEL_CHAIN, i.e. basename → strip extension → keep first 3 “_” parts → drop hyphen suffixes.
Examples:
- 3B58_1_B-C-A → 3B58_1_B
- 7M57_1_qq-bb → 7M57_1_qq
- ./data/raw/6ZU1_1_AW.pdb → 6ZU1_1_AW

Splits (das_split.pt)

data/das_split.pt provides indices into processed.pt:

train: 4025, val: 100, test: 98 (in the current dataset).

We consider das_split.pt the source of truth, because it was created alongside processed.pt. For each split, we index all canonical IDs in id_list to the same graph index, so any alias from id_list resolves to the correct graph.

External split lists (data/split_ids/*.txt)

You also have split_ids/test_ids_das.txt (235 IDs), which is a larger/older superset vs this processed.pt+das_split.pt.

Comparison result you observed:

External test: 235 IDs

DAS test (from das_split.pt): 98 IDs

Extra in DAS test (not in external): 0

Missing in DAS test (present in external): 137

Interpretation: every DAS test ID is in the external set, but the external set contains additional backbones not present in this processed.pt. That’s a dataset scope difference, not a bug.

Preference pairs (dpo_pairs_margin125.json)

Your pairs file has entries like:

{
  "pdb_file": "./data/raw/6ZU1_1_AW.pdb",
  "winner_seq": "...", "loser_seq": "...",
  "winner_metrics": {...}, "loser_metrics": {...}
}


We derive the backbone ID from pdb_file using the same canonicalizer (→ 6ZU1_1_AW).

We also support pairs that carry an explicit global index (index, idx, etc.). If found, we map that index into the split via das_split.pt.

What we fixed (and why it mattered)

Canonicalization: unified ID handling across processed.pt, the external lists, and pairs (pdb_file).

Index all IDs: we map every ID in id_list (after canonicalization) to the same backbone → pairs resolve correctly even if they use a different alias than the most common one.

Pairs resolution: pairs are resolved by:

direct index (if present), else

canonicalized id (e.g., from pdb_file or other id keys).

Results (from your latest run):

Pairs: 37,928 total → train 35,883, val 1,103, test 928, unknown 14.

The 928 “test” come from pairs whose backbones fall in DAS test split (as expected).

The 14 “unknown” refer to pairs whose IDs aren’t in processed.pt (different snapshot); you can drop or add those backbones.

Commands you can run
Inspect the DAS split (counts & sample IDs)
python -m dpo.debug.inspect_das_split \
  --split_pt data/das_split.pt \
  --processed_pt data/processed.pt \
  --test_ids_file data/split_ids/test_ids_das.txt

Check pairs vs split (counts & issues)
python -m dpo.debug.check_pairs_against_split \
  --pairs_path data/pairs_margin125/dpo_pairs_margin125.json \
  --processed_pt data/processed.pt \
  --split_pt data/das_split.pt --allow_test


Add --allow_test if you don’t want warnings for test-split pairs.

Split the pairs into train/val/test JSONLs (see script below)
# Split by das_split.pt (recommended)
python -m dpo.scripts.filter_pairs_by_split \
  --pairs_in data/pairs_margin125/dpo_pairs_margin125.json \
  --processed_pt data/processed.pt \
  --split_pt data/das_split.pt \
  --out_dir data/pairs_margin125/split_by_das

# Split by external ID lists (optional)
python -m dpo.scripts.filter_pairs_by_split \
  --pairs_in data/pairs_margin125/dpo_pairs_margin125.json \
  --processed_pt data/processed.pt \
  --ids_dir data/split_ids \
  --split_source ids \
  --out_dir data/pairs_margin125/split_by_ids

Which split source should we use?

Use das_split.pt by default (it matches processed.pt one-to-one).

The external split_ids/*.txt are still useful for repro or cross-checks, but they include many IDs that aren’t in this processed.pt. The splitter below supports both sources so you can pick what you need per run.