#!/usr/bin/env bash
# Evaluate a phase-2 checkpoint with the canonical SSTT pipeline.
# Outputs a per-checkpoint eval_summary.json suitable for comparison plots.
#
# Usage:
#   bash scripts/eval_phase2_checkpoint.sh <ckpt_path> <tag>
# Example:
#   bash scripts/eval_phase2_checkpoint.sh runs/phase2/thermo_surplus/m25/thermo_surplus_m25_b0.12/best.pt thermo_surplus

set -eo pipefail   # NOTE: no -u — bashrc has unbound vars
CKPT="${1:?usage: $0 <ckpt_path> <tag>}"
TAG="${2:?usage: $0 <ckpt_path> <tag>}"
OUT_DIR="runs/phase2/eval/${TAG}"
mkdir -p "${OUT_DIR}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
set +u
source ~/.bashrc.bak.2026-0319-1628 2>/dev/null || true
eval "$(/mnt/dna01/library-seq/luca/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate grnade
set -u

# Pre-process: if the checkpoint contains FiLM head weights (Pareto-DPO Stage 2),
# strip them to produce a base-compatible checkpoint that the standard eval pipeline
# can load. The Stage-2-aware evaluation (centroid w via WeightConditionedAutoregressiveGNN)
# is left for follow-up; here we report the base-equivalent policy after Stage-2 training.
EVAL_CKPT="${CKPT}"
EVAL_CKPT_NOTE=""
python -c "
import torch
from pathlib import Path
ckpt = Path('${CKPT}')
sd = torch.load(ckpt, map_location='cpu')
# Some checkpoints wrap state_dict under 'model' or 'state_dict'
inner = sd
key_used = None
for k in ('state_dict', 'model'):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        inner = sd[k]; key_used = k; break
film_keys = [k for k in inner if k.startswith('base.') or k.startswith('film.')]
if film_keys:
    new_inner = {}
    for k, v in inner.items():
        if k.startswith('film.'):
            continue
        new_inner[k.replace('base.', '', 1) if k.startswith('base.') else k] = v
    if key_used:
        sd[key_used] = new_inner
    else:
        sd = new_inner
    out = ckpt.with_name(ckpt.stem + '_baseonly.pt')
    torch.save(sd, out)
    print(f'STRIPPED_FILM:{out}')
"  > /tmp/_strip_$$.out 2>&1
if grep -q 'STRIPPED_FILM:' /tmp/_strip_$$.out; then
  EVAL_CKPT=$(grep 'STRIPPED_FILM:' /tmp/_strip_$$.out | sed 's/.*STRIPPED_FILM://')
  EVAL_CKPT_NOTE="(FiLM-stripped from ${CKPT})"
  echo "[note] using FiLM-stripped checkpoint: ${EVAL_CKPT} ${EVAL_CKPT_NOTE}"
fi
rm -f /tmp/_strip_$$.out

# Build a lightweight eval config from bench_full.yaml with the right ckpt
TMP_CFG="${OUT_DIR}/_eval_config.yaml"
python -c "
import yaml, sys
from pathlib import Path
with open('dpo/configs/bench_full.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['paths']['checkpoints'] = [{'name': '${TAG}', 'path': '${EVAL_CKPT}'}]
cfg['eval']['out_dir'] = '${OUT_DIR}'
cfg['eval']['n_samples'] = 8
cfg['eval']['temperature'] = 0.1
cfg['eval'].setdefault('wandb', {})['enable'] = False
# Phase-2 fast-eval: skip the most expensive metrics (lDDT, MCQ, clash via relax).
# This drops per-structure time from ~258s to ~60-90s, cutting full-eval from ~7h to ~2h.
# We still compute recovery, scMCC, MFE, RMSD, TM, pLDDT, INF, ED, diversity --- the
# canonical phase-2 panel. lDDT/MCQ/clash for the final picks can be re-run later.
cfg['eval']['use_lddt'] = False
cfg['eval']['use_relax'] = False
# Drop the heaviest metrics from the list so the trainer doesn't try to compute them
metrics = cfg['eval'].get('metrics', [])
metrics = [m for m in metrics if m not in ('clash_score', 'mcq', 'lDDT')]
cfg['eval']['metrics'] = metrics
Path('${TMP_CFG}').write_text(yaml.safe_dump(cfg))
print('eval config -> ${TMP_CFG}')
"

python -m dpo.bench.eval_full --config "${TMP_CFG}" 2>&1 | tee "${OUT_DIR}/eval.log"

# extract a compact summary for plotting.
# eval_full.py writes the results JSON to dpo/eval_results/eval_<split>_<ckpt>_<ts>.json.
# We look for the most recent file containing our TAG and pull canonical metrics.
python -c "
import json, glob, os
from pathlib import Path
out_dir = Path('${OUT_DIR}')

# Look in both the per-run out_dir and the canonical dpo/eval_results/ tree.
candidates = []
candidates.extend(sorted(out_dir.glob('*results*.json')))
candidates.extend(sorted(Path('dpo/eval_results').glob(f'eval_*${TAG}*_*.json'), key=os.path.getmtime))
candidates = [c for c in candidates if c.is_file()]
if not candidates:
    print('[warn] no results JSON found for tag=${TAG}')
    raise SystemExit(0)
chosen = candidates[-1]
print(f'parsing {chosen}')
data = json.loads(chosen.read_text())

def find(d, keys):
    # Prefer per_checkpoint_results (flat per-ckpt scalars) over summary (metadata dict).
    if isinstance(d, dict):
        if 'per_checkpoint_results' in d and isinstance(d['per_checkpoint_results'], list):
            for entry in d['per_checkpoint_results']:
                if isinstance(entry, dict):
                    for k in keys:
                        if k in entry and isinstance(entry[k], (int, float)) and not (isinstance(entry[k], bool)):
                            return entry[k]
        # Otherwise, recursive scan
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    for kk in ('mean', 'value', 'avg'):
                        if kk in v and isinstance(v[kk], (int, float)):
                            return v[kk]
                    continue  # dict without numeric leaf — keep searching
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
    'tag': '${TAG}',
    'ckpt': '${CKPT}',
    'recovery':         find(data, ['recovery']),
    'scMCC':            find(data, ['sc_eternafold', 'scMCC']),
    'MFE':              find(data, ['vienna_mfe', 'MFE', 'mfe']),
    'RMSD':             find(data, ['sc_rmsd', 'rmsd', 'RMSD']),
    'TM':               find(data, ['sc_tm', 'tm', 'TM']),
    'pLDDT':            find(data, ['sc_plddt', 'plddt', 'pLDDT']),
    'diversity':        find(data, ['diversity_3mer', 'diversity']),
    'inf_all':          find(data, ['inf_all']),
    'inf_wc':           find(data, ['inf_wc']),
    'inf_nwc':          find(data, ['inf_nwc']),
    'rmsd_within_8A':   find(data, ['rmsd_within_8A']),
    'plddt_above_070':  find(data, ['plddt_above_070']),
    'vienna_pS0':       find(data, ['vienna_pS0']),
    'vienna_Tm':        find(data, ['vienna_Tm']),
    'vienna_ED_per_nt': find(data, ['vienna_ED_per_nt']),
    'perplexity':       find(data, ['perplexity']),
    'clashscore':       find(data, ['clashscore_pre_relax']),
}
# GC content is not in eval_full output; compute from saved designed FASTA if present.
designs_glob = list(out_dir.glob('**/*.fasta')) + list(Path('runs/phase2').glob('**/${TAG}*designs*/*.fasta'))
if designs_glob:
    seqs = []
    for fp in designs_glob:
        with fp.open() as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('>'):
                    seqs.append(line)
    if seqs:
        def gc_frac(s):
            s = s.upper().replace('T','U')
            return sum(1 for c in s if c in {'G','C'}) / max(len(s),1)
        summary['GC'] = float(sum(gc_frac(s) for s in seqs) / len(seqs))
out = out_dir / 'eval_summary.json'
out.write_text(json.dumps(summary, indent=2))
print('summary ->', out)
print(summary)
"
