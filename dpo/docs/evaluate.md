# Evaluating Structure Quality with MolProbity Clash Score

This document outlines how to set up and use the MolProbity clash score calculation to evaluate the quality of predicted RNA 3D structures within this project. The clash score is a standard metric that measures the number of serious steric overlaps (or "clashes") per 1000 atoms, where a **lower score indicates a better, more physically realistic structure**.

We use the MolProbity tool included in the **Phenix** software suite, called via a custom wrapper script to ensure compatibility with our Conda environments.

***

## ## 1. Prerequisites

The primary dependency is the **Phenix software suite**.

- **Installation:** Phenix must be downloaded and installed. It is recommended to install it within the project's `tools/` directory.
- **Download:** Phenix can be downloaded from its official website (requires free academic registration): [https://phenix-online.org/download/](https://phenix-online.org/download/)
- **Installation Tip:** During the command-line installation, you may need to deactivate the project's Conda environment (e.g., `conda deactivate`) and run the installer from the `(base)` environment to avoid conflicts.

***

## ## 2. Setup and Configuration

To resolve deep environment conflicts between our project's Conda environment (`grnade`) and Phenix's internal Python 2.7 environment, a dedicated wrapper script is required.

### ### 2.1. The Wrapper Script (`run_phenix.sh`)

This script uses `conda run -n base` to execute any Phenix command within the clean, compatible `base` Conda environment. This provides perfect isolation and is the key to making the integration work.

- **Location:** The script must be located at `tools/run_phenix.sh`.
- **Content:**

  ```bash
  #!/bin/bash
  # This wrapper uses `conda run` to execute Phenix commands inside the clean
  # 'base' conda environment, bypassing conflicts with the active 'grnade' env.

  # We tell conda to run a bash command in the 'base' environment.
  # Inside that command, we first source the phenix environment, and then
  # execute the command and arguments ("$@") that were passed to this wrapper.
  conda run -n base bash -c "source ./tools/phenix-1.21.2-5419/phenix_env.sh && exec \"\$@\"" -- "$@"
  ```

- **Permissions:** After creating the script, it must be made executable:

  ```bash
  chmod +x tools/run_phenix.sh
  ```

***

## ## 3. Usage in Python

The clash score calculation is performed by the `get_clash_score_phenix` function in `src/evaluator.py`.

### ### 3.1. The Python Function

The function takes the path to a structure file and the path to the wrapper script as input.

- **Function Signature:**
  ```python
  get_clash_score_phenix(pdb_file, phenix_wrapper_path)
  ```

- **Example Call:**
  ```python
  import os
  from src.evaluator import get_clash_score_phenix

  # Path to the wrapper script
  wrapper_path = "tools/run_phenix.sh"
  
  # Path to the structure file to evaluate (PDB or CIF format)
  structure_file = "dpo/debug/example_data/1Y0T.cif"

  # Calculate the score
  if os.path.exists(structure_file):
      score = get_clash_score_phenix(structure_file, wrapper_path)
      print(f"The MolProbity Clash Score is: {score}")
  ```

### ### 3.2. Integration into the Pipeline

This function can be called within the main evaluation loop. For example, in `self_consistency_score_rhofold`, it can be used to score each predicted PDB.

```python
# Inside src/evaluator.py -> self_consistency_score_rhofold() loop:
# ...
      # Compute clash if requested
      if use_clash:
          phenix_wrapper = "tools/run_phenix.sh"
          _clash = get_clash_score_phenix(design_pdb_path, phenix_wrapper)
          sc_clash.append(_clash)
# ...
```

***

## ## 4. Verification

To verify that the entire setup is working correctly, you can run the built-in test in the `evaluator.py` script.

- **Command:** From the main `ribopo/` project directory, run:
  ```bash
  python -m src.evaluator
  ```

- **Expected Output:** The script will find or download a test structure (`1Y0T.cif`), calculate its clash score, and verify it against the known value. A successful run will look like this:

  ```
  --- Running test for get_clash_score_phenix with CIF file ---
  Calculating clash score for '1Y0T.cif'...

  --- TEST RESULT ---
  ✅ Success! Calculated Clash Score: 6.4
  ✅ Score is consistent with the expected value of ~6.4.
  ```

```python
# clash score testing
if __name__ == '__main__':
    print("--- Running test for get_clash_score_phenix with CIF file ---")

    # --- Configuration ---
    PHENIX_WRAPPER_PATH = "./tools/run_phenix.sh"
    
    # MODIFICATION: Using the exact directory and CIF filename you provided
    TEST_STRUCTURE_FILE = "./dpo/debug/example_data/1Y0T.cif"
    
    # --- Test Setup ---
    # Check if your CIF file exists. If not, the script will stop.
    if not os.path.exists(TEST_STRUCTURE_FILE):
        print(f"❌ ERROR: Test file not found at the specified path.")
        print(f"Please ensure '{TEST_STRUCTURE_FILE}' exists.")
        exit(1)

    # --- Run the Function ---
    try:
        print(f"Calculating clash score for '{os.path.basename(TEST_STRUCTURE_FILE)}'...")
        score = get_clash_score_phenix(TEST_STRUCTURE_FILE, PHENIX_WRAPPER_PATH)
        
        print("\n--- TEST RESULT ---")
        print(f"✅ Success! Calculated Clash Score: {score}")

        # The clash score is for the structure itself, so it should be very similar
        # regardless of whether the input is PDB or CIF format.
        expected_score = 6.40 
        assert abs(score - expected_score) < 0.1, "Score does not match expected value!"
        print(f"✅ Score is consistent with the expected value of ~{expected_score}.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n--- TEST FAILED ---")
        print(f"❌ An error occurred: {e}")
```

USalign test
```python
# from repo root
python -m dpo.debug.test_usalign \
  --model dpo/debug/example_data/model.pdb \
  --native dpo/debug/example_data/native.pdb \
  --agg avg
```


# Thermostability metrics (ViennaRNA) — what we compute, how, and how to use

This section documents the thermostability-side metrics we added to `src/evaluator.py` (gated behind `sc_score_vienna`), how they’re computed with ViennaRNA, what they mean, and how we recommend using them in RiboPO.

---

## Dependencies & setup

* **Library**: ViennaRNA Python API (`import RNA`)
  Install: `mamba install -c conda-forge viennarna`
* **Default temperature**: **37 °C** unless stated otherwise.
* **Masking**: If some residues lack 3D coords, we slice both the **sequence** and **target dot-bracket** by `mask_coords` to keep strings contiguous for Vienna.
* **Pseudoknots**: Vienna accepts only `().` — we sanitize any other characters to `.` before computing ensemble metrics.

---

## Inputs

* `seq` (string of A/C/G/U), optionally masked.
* `target_db` (dot-bracket), same length as `seq`. If you don’t have a trustworthy target:

  * Use the **sequence’s own MFE structure at 37 °C** (we do this by default in the debug suite), or
  * Use a consensus/ground-truth 2D converted to `().` only.

---

## Metrics we compute

All of these come from a single partition function (PF) call per sequence at a given temperature; we build a `fold_compound` with `md.temperature=T`, then call `fc.mfe()` and `fc.pf()`.

### 1) Minimum Free Energy (MFE)

* **What**: The free energy of the predicted minimum-energy structure (kcal/mol).
* **How**: `mfe_db, mfe = fc.mfe()`
* **Interpretation**: Lower (more negative) is “more stable,” but **single-structure only**. MFE does **not** reflect ensemble breadth.
* **In code / outputs**: `vienna_mfe` (per-datapoint mean), `mfe_db` is also kept in the ensemble dict.

### 2) Ensemble Defect (ED) and ED per nucleotide

* **What**: Expected number of nucleotides **not** in their target state under the ensemble; `ED_per_nt = ED / N`.
* **How**: `ED = fc.ensemble_defect(target_db)`
* **Range**: `0 … N`; normalized `0 … 1`.
* **Interpretation**: **Lower is better**. Measures how concentrated the ensemble is **around the target**; directly target-dependent.
* **In code / outputs**: `vienna_ED`, `vienna_ED_per_nt`.

### 3) Probability of the target structure (`pS0`)

* **What**: Boltzmann probability of the **exact** `target_db`.
* **How**: `pS0 = fc.pr_structure(target_db)`
* **Range**: `0 … 1`. Falls with higher temperature for typical stable targets.
* **Interpretation**: **Higher is better**. Strong, interpretable signal for “how likely is the thing we want.”

### 4) Positional Shannon entropy (mean)

* **What**: Per-position uncertainty derived from pairing probabilities; we report the **mean** across positions.
* **How**: `H = fc.positional_entropy()` → list → `entropy_mean = mean(H)`
* **Interpretation**: **Lower is better** (sharper ensemble). Correlates with design rigidity. Not target-dependent.
* **In code / outputs**: `vienna_entropy` (mean). We can expose the full list if needed.

### 5) Ensemble diversity (mean base-pair distance)

* **What**: Average base-pair distance (bp-distance) across the ensemble.
* **How**: `diversity = fc.mean_bp_distance()`
* **Interpretation**: **Lower is better** (less diverse ensemble). Scales with length; optionally normalize by `N` if you compare across lengths.
* **In code / outputs**: `vienna_diversity`.

### 6) Coarse melting temperature (Tm) by `p(S0)≈0.5`

* **What**: Temperature (°C) where the **probability of the target** is closest to 0.5.
* **How**: sweep `T` from `Tmin` to `Tmax` (default **10–95 °C**, step **1 °C**):
  `fc = fold_compound(seq, T)` → `fc.pf()` → `p = pr_structure(target_db)` → pick `T` minimizing `|p-0.5|`.
* **Interpretation**: Useful **proxy**; not a strict two-state Tm. Depends on accuracy of `target_db`.
* **Performance**: Costs one PF per temperature tested; keep it to **winners** or 1–2 top samples per datapoint.
* **In code / outputs**: `vienna_Tm`.

---

## What gets returned by `evaluate()` when enabled

Add `'sc_score_vienna'` to your `metrics` list. You’ll get these per-datapoint arrays in `out`:

* `out['vienna_mfe']` — mean MFE (kcal/mol) across that item’s samples
* `out['vienna_ED']` — mean ensemble defect (nt)
* `out['vienna_ED_per_nt']` — mean ED normalized by length
* `out['vienna_pS0']` — mean probability of target structure
* `out['vienna_entropy']` — mean positional Shannon entropy
* `out['vienna_diversity']` — mean ensemble diversity (bp-distance)
* `out['vienna_Tm']` — mean coarse Tm (°C) if Tm computation is left on

> You control whether Tm is computed in the per-sample loop; feel free to compute it for **winners only** to save time.

---

## Recommended use in RiboPO (rewarding & pairing)

### As standalone reward components (direction of optimization)

* **Maximize**: `pS0`
* **Minimize**: `ED_per_nt`, `entropy_mean`, `diversity`
* **Use with care**: `MFE` (scale by length or simply track; it is less predictive than the ensemble metrics)

### Simple composite (z-scored within-batch or within-dataset)

Let `z(x)` be a standard score, clipped to `±3` to reduce outlier effects.

```
R_thermo =  + w_p * z(pS0)
            - w_ed * z(ED_per_nt)
            - w_h  * z(entropy_mean)
            - w_div* z(diversity_norm)
          + [w_tm * z(Tm_norm)]      # optional
```

* Start with equal weights (`w_* = 1.0`) and tune via ablations.
* If you compare **across lengths**, consider `diversity_norm = diversity / N`.
* For Tm, normalize per target family or use a **rank-based** score (percentile) to avoid scale issues.

### Preference-pair construction (thermo-only or thermo+structural)

* **Filter/gating** (optional but effective):
  Drop candidates with `ED_per_nt > 0.2` or `pS0 < 0.05` (tune thresholds per dataset).
* **Margining**:
  On the composite `R_thermo`, mark winner/loser if the gap exceeds **0.125–0.25 σ** (mirrors your current RMSD/pLDDT policy).
  If you require agreement with structure: enforce that the thermo winner is **not** a structure loser (e.g., `RMSD` not worse by >0.125 σ).

---

## Performance notes

* PF is roughly **O(N³)**; MFE is cheaper.
* Entropy/ED/diversity all come “for free” after one PF at that temperature.
* Tm sweeps multiply the PF cost by the number of temperatures tested; keep step coarse (1–2 °C) or do an initial coarse sweep + **binary refinement** near the minimum if you need tighter Tm.

---

## Edge cases & gotchas

* **Pseudoknots** in targets: sanitize to `.` (we do) or rely on MFE db (PK-free).
* **Length mismatch**: dot-bracket must match sequence length after masking; we assert this.
* **Very diffuse ensembles**: `pS0` can be near zero even at 20–30 °C; in such cases ED/entropy/diversity are more informative than Tm.
* **Extremely short RNAs**: diversity scales strongly with `N`; prefer normalized forms for cross-length comparisons.

---

## Where to see these in action

* `dpo/debug/test_vienna_ensemble.py` — single-sequence sanity & JSON dump
* `dpo/debug/run_generate_dbn_for_examples.py` — generate `.dbn` targets for all `vienna_*.fa`
* `dpo/debug/test_vienna_examples_suite.py` — runs MFE/ED/pS0/entropy/diversity/Tm over **all** examples; writes CSV/JSON
* `dpo/debug/test_vienna_ps0_trends.py` — checks that `p(S0)` **decreases** with temperature per example

These scripts also create or reuse example inputs in `dpo/debug/example_data/` so you can reproduce our behavior quickly.

---

## Quick mapping (function → metric)

* `vienna_mfe(seq, T)` → `mfe`, `mfe_db`
* `vienna_ensemble_metrics(seq, target_db, T, return_positional_entropy)` →
  `ED`, `ED_per_nt`, `pS0`, `entropy_mean`, `diversity` (+ optional `entropy_list`)
* `vienna_Tm_by_pS0(seq, target_db, Tmin, Tmax, step, threshold)` → `Tm`





