#!/usr/bin/env bash
# Run SSTT eval on pre-generated baseline FASTAs.
#
# Usage:
#   bash scripts/run_baseline_sstt_eval.sh <jobid> <baseline_tag>
# Example:
#   bash scripts/run_baseline_sstt_eval.sh 4790774 rhodesign
#
# Reads FASTAs from runs/phase2/baselines/<tag>/designs/<gid>/sample{0..7}.fasta
# Writes SSTT eval JSON + summary to runs/phase2/baselines/<tag>/eval_summary.json.

set -eo pipefail
JOBID="${1:?usage: $0 <jobid> <tag>}"
TAG="${2:?usage: $0 <jobid> <tag>}"
ROOT=/mnt/rna01/smh/projects/ribopo
cd "$ROOT"

DESIGNS="${ROOT}/runs/phase2/baselines/${TAG}/designs"
OUT="${ROOT}/runs/phase2/baselines/${TAG}"
mkdir -p "${OUT}"

if [[ ! -d "$DESIGNS" ]]; then
  echo "ERROR: designs dir not found: $DESIGNS"
  exit 1
fi
ndesigns=$(ls "$DESIGNS" | wc -l)
echo "[baseline-eval] $TAG: $ndesigns design dirs at $DESIGNS"

# Build a temp config: same as bench_full but with no model needed (--from_fasta_dir)
TMP_CFG="${OUT}/_eval_config.yaml"
python3 - <<PY
import yaml
from pathlib import Path
with open("dpo/configs/bench_full.yaml") as f:
    cfg = yaml.safe_load(f)
# Stub the checkpoint entry — its path is unused when --from_fasta_dir is set,
# but the loop iterates over cfg.paths.checkpoints to know what to label.
cfg["paths"]["checkpoints"] = [{"name": "${TAG}", "path": "/dev/null"}]
cfg["eval"]["out_dir"] = "${OUT}"
cfg["eval"]["n_samples"] = 8
cfg["eval"]["temperature"] = 0.1
cfg["eval"]["save_designs"] = False  # inputs already on disk
cfg["eval"].setdefault("wandb", {})["enable"] = False
# Fast-eval: skip the most expensive metrics
cfg["eval"]["use_lddt"] = False
cfg["eval"]["use_relax"] = False
metrics = cfg["eval"].get("metrics", [])
metrics = [m for m in metrics if m not in ("clash_score", "mcq", "lDDT")]
cfg["eval"]["metrics"] = metrics
Path("${TMP_CFG}").write_text(yaml.safe_dump(cfg))
print("config ->", "${TMP_CFG}")
PY

# Dispatch via srun --jobid --overlap
srun --jobid="${JOBID}" --overlap bash -c "
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null
eval \"\$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)\"
mamba activate grnade
cd /mnt/rna01/smh/projects/ribopo
python -m dpo.bench.eval_full --config '${TMP_CFG}' --from_fasta_dir '${DESIGNS}' 2>&1
" | tee "${OUT}/eval.log"

# Build a compact eval_summary.json by reusing the parser logic from
# scripts/eval_phase2_checkpoint.sh
python3 - <<'PY'
import json, glob, os
from pathlib import Path
out_dir = Path("${OUT}")
candidates = list(Path("dpo/eval_results").glob(f"eval_*${TAG}*_*.json"))
if not candidates:
    print("[warn] no results JSON found; the eval may have failed")
    raise SystemExit(0)
chosen = max(candidates, key=os.path.getmtime)
print(f"parsing {chosen}")
data = json.loads(chosen.read_text())

def find(d, keys):
    if isinstance(d, dict):
        if "per_checkpoint_results" in d and isinstance(d["per_checkpoint_results"], list):
            for entry in d["per_checkpoint_results"]:
                if isinstance(entry, dict):
                    for k in keys:
                        if k in entry and isinstance(entry[k], (int, float)) and not isinstance(entry[k], bool):
                            return entry[k]
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return v
        for v in d.values():
            r = find(v, keys)
            if r is not None: return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, keys)
            if r is not None: return r
    return None

summary = {
    "tag": "${TAG}",
    "ckpt": "from_fasta_dir",
    "recovery":         find(data, ["recovery"]),
    "scMCC":            find(data, ["sc_eternafold", "scMCC"]),
    "MFE":              find(data, ["vienna_mfe", "MFE", "mfe"]),
    "RMSD":             find(data, ["sc_rmsd", "rmsd", "RMSD"]),
    "TM":               find(data, ["sc_tm", "tm", "TM"]),
    "pLDDT":            find(data, ["sc_plddt", "plddt", "pLDDT"]),
    "diversity":        find(data, ["diversity_3mer", "diversity"]),
    "inf_all":          find(data, ["inf_all"]),
    "inf_wc":           find(data, ["inf_wc"]),
    "inf_nwc":          find(data, ["inf_nwc"]),
    "rmsd_within_8A":   find(data, ["rmsd_within_8A"]),
    "plddt_above_070":  find(data, ["plddt_above_070"]),
    "vienna_pS0":       find(data, ["vienna_pS0"]),
    "vienna_Tm":        find(data, ["vienna_Tm"]),
    "vienna_ED_per_nt": find(data, ["vienna_ED_per_nt"]),
    "perplexity":       find(data, ["perplexity"]),
}
out = out_dir / "eval_summary.json"
out.write_text(json.dumps(summary, indent=2))
print("summary ->", out)
print(summary)
PY
