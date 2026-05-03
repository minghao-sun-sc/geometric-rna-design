"""Generate NA-MPNN baseline sequences for the DAS test set.

NA-MPNN is the MPNN-style RNA inverse-folding baseline (Nucleic Acid MPNN), the
RNA analog of ProteinMPNN. Wrapper at /mnt/rna01/smh/projects/tools/NA-MPNN/run_na_mpnn.sh
already handles env activation, decode of b/d/h/u → A/C/G/U, and design loop.

For each PDB, runs the wrapper with --num_designs 8 → 8 designs in a single
FASTA (one header + sequence per design). We split that into per-sample
FASTAs at:

    runs/phase2/baselines/na_mpnn/designs/<gid>/sample{0..7}.fasta

Then `python -m dpo.bench.eval_full --config <cfg> --from_fasta_dir <DIR>` consumes
these for the SSTT panel.

NA-MPNN runs on CPU (per its DEPLOYMENT.md) — this script can run on the login
node, no GPU jobid required. Estimate: ~5–15 sec per structure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

NA_MPNN = Path("/mnt/rna01/smh/projects/tools/NA-MPNN")


def _read_fasta(fp: Path):
    """Yield (header, seq) records, joining multi-line sequences."""
    rec_h, rec_s = None, []
    with fp.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if rec_h is not None:
                    yield rec_h, "".join(rec_s)
                rec_h = line
                rec_s = []
            elif line:
                rec_s.append(line)
    if rec_h is not None:
        yield rec_h, "".join(rec_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", default="/mnt/rna01/smh/projects/ribopo/data/das_split_raw_data/das_split_raw_pdb")
    parser.add_argument("--out-dir", default="/mnt/rna01/smh/projects/ribopo/runs/phase2/baselines/na_mpnn/designs")
    parser.add_argument("--gids-from", default="/mnt/rna01/smh/projects/ribopo/runs/phase2/baselines/test_gids.txt")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    with open(args.gids_from) as f:
        gids = [g.strip() for g in f if g.strip()]
    if args.limit:
        gids = gids[: args.limit]
    print(f"[runner] processing {len(gids)} structures with NA-MPNN")

    n_done = n_skip = n_fail = 0
    for gi, gid in enumerate(gids):
        pdb_path = pdb_dir / f"{gid}.pdb"
        if not pdb_path.exists():
            print(f"[runner] missing PDB for {gid}; skip")
            n_skip += 1
            continue
        out_dir = out_root / gid
        out_dir.mkdir(parents=True, exist_ok=True)
        if all((out_dir / f"sample{s}.fasta").exists() for s in range(args.n_samples)):
            n_done += 1
            continue

        # NA-MPNN wrapper writes to <work_dir>/seqs/<stem>.rna.fa with N+1 records
        # (first = native, rest = designs). We capture the designs.
        work_dir = out_root / f"_tmp_{gid}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                "bash", str(NA_MPNN / "run_na_mpnn.sh"),
                str(pdb_path), str(work_dir),
                str(args.n_samples), str(args.temperature), str(args.seed),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                print(f"[runner] FAIL on {gid}: rc={r.returncode}\nSTDERR: {r.stderr[-300:]}")
                n_fail += 1
                continue
            # Find the decoded RNA FASTA
            rna_fa = work_dir / "seqs" / f"{gid}.rna.fa"
            if not rna_fa.exists():
                # Try with stem (without underscores etc.)
                stem = pdb_path.stem
                rna_fa = work_dir / "seqs" / f"{stem}.rna.fa"
            if not rna_fa.exists():
                print(f"[runner] expected output missing for {gid}: {rna_fa}")
                n_fail += 1
                continue
            records = list(_read_fasta(rna_fa))
            # First record is native; designs are records[1:]
            designs = records[1:]
            if len(designs) < args.n_samples:
                print(f"[runner] WARN {gid}: only {len(designs)} designs (wanted {args.n_samples}); padding by repeat")
                while len(designs) < args.n_samples:
                    designs.append(designs[-1] if designs else (">empty", "A"))
            for s in range(args.n_samples):
                hdr, seq = designs[s]
                out_fp = out_dir / f"sample{s}.fasta"
                out_fp.write_text(f">{gid}_sample{s}\n{seq}\n")
            n_done += 1
            if (gi + 1) % 10 == 0 or gi < 3:
                print(f"[runner] {gi + 1}/{len(gids)} done={n_done} fail={n_fail} skip={n_skip}")
        except subprocess.TimeoutExpired:
            print(f"[runner] timeout on {gid}")
            n_fail += 1
        except Exception as e:
            print(f"[runner] FAIL on {gid}: {type(e).__name__}: {e}")
            n_fail += 1
        finally:
            # Clean up the per-PDB work dir
            try:
                import shutil
                shutil.rmtree(work_dir)
            except Exception:
                pass

    print(f"[runner] DONE: {n_done} done, {n_skip} skipped, {n_fail} failed")
    print(f"[runner] sequences at: {out_root}")


if __name__ == "__main__":
    main()
