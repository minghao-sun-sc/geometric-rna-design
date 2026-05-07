#!/usr/bin/env bash
# Evaluate a Pareto-Stage-2 (FiLM-conditioned) checkpoint at a specific
# scalarization weight w on the simplex. Unlike scripts/eval_phase2_checkpoint.sh,
# this DOES NOT strip the FiLM head — it loads the full WeightConditionedAutoregressiveGNN
# and samples at the supplied w.
#
# Usage:
#   bash scripts/eval_pareto_front.sh <jobid> <ckpt_path> <tag> <w_csv>
# Example:
#   bash scripts/eval_pareto_front.sh 4817734 \
#     runs/phase2/pareto_stage2/pareto_stage2_b0.12/best.pt \
#     pareto_stage2_b012_w_rmsd \
#     1,0,0

set -uo pipefail
JOBID="${1:?usage: $0 <jobid> <ckpt_path> <tag> <w_csv>}"
CKPT="${2:?usage: $0 <jobid> <ckpt_path> <tag> <w_csv>}"
TAG="${3:?usage: $0 <jobid> <ckpt_path> <tag> <w_csv>}"
W_CSV="${4:?usage: $0 <jobid> <ckpt_path> <tag> <w_csv>}"

ROOT=.
OUT_DIR="$ROOT/runs/phase2/eval/${TAG}"
mkdir -p "${OUT_DIR}"

# Build temp eval config locally (need yaml in env, do it before srun)
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval "$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)" 2>/dev/null
mamba activate grnade
set -u

cd "$ROOT"

TMP_CFG="${OUT_DIR}/_eval_config.yaml"
python3 - <<PY
import yaml
from pathlib import Path
with open("dpo/configs/bench_full.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["paths"]["checkpoints"] = [{"name": "${TAG}", "path": "${CKPT}"}]
cfg["eval"]["out_dir"] = "${OUT_DIR}"
cfg["eval"]["n_samples"] = 8
cfg["eval"]["temperature"] = 0.1
cfg["eval"]["save_designs"] = True
cfg["eval"].setdefault("wandb", {})["enable"] = False
# Phase-2 fast-eval: skip the most expensive metrics (lDDT, MCQ, clash via relax).
cfg["eval"]["use_lddt"] = False
cfg["eval"]["use_relax"] = False
metrics = cfg["eval"].get("metrics", [])
metrics = [m for m in metrics if m not in ("clash_score", "mcq", "lDDT")]
cfg["eval"]["metrics"] = metrics
# Tell the model layer to expect a 3-d FiLM weight (matches Pareto-Stage-2 default).
cfg["model"]["w_dim"] = 3
Path("${TMP_CFG}").write_text(yaml.safe_dump(cfg))
print("config ->", "${TMP_CFG}")
PY

echo "[pareto-front] tag=${TAG} jobid=${JOBID} ckpt=${CKPT} w=${W_CSV}"

# Dispatch via srun --jobid --overlap so it runs on the GPU node
exec srun --jobid="${JOBID}" --overlap bash -c "
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval \"\$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)\"
mamba activate grnade
cd $ROOT
python -m dpo.bench.eval_full --config '${TMP_CFG}' --w '${W_CSV}' 2>&1 | tee '${OUT_DIR}/eval.log'
"
