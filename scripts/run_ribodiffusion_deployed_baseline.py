"""Generate RiboDiffusion (DEPLOYED, large model) baseline sequences.

This is the deployed RiboDiffusion at tools/RiboDiffusion/
with the LARGE 997 MB checkpoint `ckpts/exp_inf.pth` — distinct from the
submodule version at external/RIdiffusion/ (smaller 30 MB weight). Worth
running as a separate baseline row.

The CLI is `main.py --PDB_file <PDB> --save_folder <out> --config.eval.n_samples 8`.
Output goes to `<save_folder>/fasta/<pdb_id>_<i>.fasta` (one file per sample).
We re-pack into our standard layout:

    runs/baselines/ribodiffusion/designs/<gid>/sample{0..7}.fasta

Compute: ~20-40 sec per structure × 98 = ~30-60 min on a single A40/A100.
"""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path

RIBODIFF = Path("tools/RiboDiffusion")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb-dir", default="./data/das_split_raw_data/das_split_raw_pdb")
    p.add_argument("--out-dir", default="./runs/baselines/ribodiffusion/designs")
    p.add_argument("--gids-from", default="./runs/baselines/test_gids.txt")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir)

    with open(args.gids_from) as f:
        gids = [g.strip() for g in f if g.strip()]
    if args.limit:
        gids = gids[: args.limit]
    print(f"[runner] processing {len(gids)} structures with RiboDiffusion (deployed)")

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

        work_dir = out_root / f"_tmp_{gid}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                "python", str(RIBODIFF / "main.py"),
                "--PDB_file", str(pdb_path),
                "--save_folder", str(work_dir),
                f"--config.eval.n_samples={args.n_samples}",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(RIBODIFF))
            if r.returncode != 0:
                print(f"[runner] FAIL on {gid}: rc={r.returncode}\nSTDERR: {r.stderr[-300:]}")
                n_fail += 1
                continue

            # Output is at <work_dir>/fasta/<pdb_id>_<i>.fasta
            fasta_dir = work_dir / "fasta"
            if not fasta_dir.exists():
                print(f"[runner] no fasta/ dir produced for {gid}")
                n_fail += 1
                continue
            # Map fasta files → sample{0..n-1}.fasta
            for s in range(args.n_samples):
                # Try common naming patterns
                candidates = list(fasta_dir.glob(f"*_{s}.fasta"))
                if not candidates:
                    candidates = list(fasta_dir.glob(f"*_{s}.fa"))
                if not candidates:
                    print(f"[runner] missing sample{s} for {gid} in {fasta_dir}")
                    continue
                # Read sequence (skip header)
                with candidates[0].open() as fh:
                    lines = fh.readlines()
                seq = "".join(
                    line.strip().upper().replace("T", "U")
                    for line in lines if not line.startswith(">")
                )
                (out_dir / f"sample{s}.fasta").write_text(f">{gid}_sample{s}\n{seq}\n")
            n_done += 1
            if (gi + 1) % 5 == 0 or gi < 3:
                print(f"[runner] {gi + 1}/{len(gids)} done={n_done} fail={n_fail} skip={n_skip}")
        except subprocess.TimeoutExpired:
            print(f"[runner] timeout on {gid}")
            n_fail += 1
        except Exception as e:
            print(f"[runner] FAIL on {gid}: {type(e).__name__}: {e}")
            n_fail += 1
        finally:
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

    print(f"[runner] DONE: {n_done} done, {n_skip} skipped, {n_fail} failed")
    print(f"[runner] sequences at: {out_root}")


if __name__ == "__main__":
    main()
