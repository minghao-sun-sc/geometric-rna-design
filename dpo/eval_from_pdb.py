# dpo/eval_from_pdb.py
from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import os, argparse, yaml, math, random
import numpy as np
import torch
from torch_geometric.data import Batch
from typing import List, Tuple, Optional


from src.data.featurizer import RNAGraphFeaturizer
from src.models import AutoregressiveMultiGNNv1, NonAutoregressiveMultiGNNv1

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_model(cfg):
    Model = AutoregressiveMultiGNNv1 if cfg["model"] == "ARv1" else NonAutoregressiveMultiGNNv1
    return Model(
        node_in_dim=tuple(cfg["node_in_dim"]),
        node_h_dim=tuple(cfg["node_h_dim"]),
        edge_in_dim=tuple(cfg["edge_in_dim"]),
        edge_h_dim=tuple(cfg["edge_h_dim"]),
        num_layers=int(cfg["num_layers"]),
        drop_rate=float(cfg["drop_rate"]),
        out_dim=int(cfg["out_dim"]),
    )

def load_checkpoint(model, path: str, device: torch.device):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def featurize_pdb(pdb_path: str, device: torch.device, feat_cfg: dict):
    fzr = RNAGraphFeaturizer(
        split="test",
        radius=feat_cfg["radius"],
        top_k=feat_cfg["top_k"],
        num_rbf=feat_cfg["num_rbf"],
        num_posenc=feat_cfg["num_posenc"],
        max_num_conformers=feat_cfg["max_num_conformers"],
        noise_scale=feat_cfg["noise_scale"],
        device=device,
    )
    raw = {"pdb_filepath": pdb_path, "backbone_id": os.path.basename(pdb_path)}
    data = fzr.featurize(raw).to(device)
    letter_to_num = fzr.letter_to_num
    num_to_letter = {v:k for k,v in letter_to_num.items()}
    return data, letter_to_num, num_to_letter

@torch.no_grad()
def sequence_ll(model, data, tokens: torch.Tensor) -> float:
    saved = data.seq
    try:
        data.seq = tokens
        logits = model(data)              # [L,4]
        logp = logits.log_softmax(-1)
        idx = torch.arange(tokens.size(0), device=tokens.device)
        return float(logp[idx, tokens].sum().item())
    finally:
        data.seq = saved

def perplexity_from_ll(ll: float, length: int) -> float:
    # per-position perplexity: exp(-ll / L)
    return float(math.exp(-ll / max(1, length)))

@torch.no_grad()
def sample_sequence(model, data, temperature: float, num_to_letter: dict, seed: int) -> str:
    # simple autoregressive sampling (left-to-right)
    g = torch.Generator(device=data.seq.device).manual_seed(seed)
    L = data.seq.numel()
    toks = data.seq.clone()  # start from native as a convenient length template
    for t in range(L):
        data.seq = toks
        logits = model(data)[t] / max(1e-6, temperature)  # [4]
        probs = torch.softmax(logits, -1)
        toks[t] = torch.multinomial(probs, 1, generator=g)
    # convert to letters
    return "".join(num_to_letter[int(x)] for x in toks.tolist())

def fasta_entry(header: str, seq: str, wrap: int = 60) -> str:
    lines = [f">{header}"]
    for i in range(0, len(seq), wrap): lines.append(seq[i:i+wrap])
    return "\n".join(lines)

def eternafold_mcc(seq: str, native_ss: Optional[str] = None) -> Optional[float]:
    try:
        import RNA  # ViennaRNA python (as tutorial discusses forward folding) :contentReference[oaicite:2]{index=2}
    except Exception:
        return None
    fc = RNA.fold_compound(seq)
    ss, mfe = fc.mfe()
    if native_ss is None:
        return None
    # Very simple MCC proxy vs. native_ss; replace with your MCC util if available
    # Here we just compute TP/FP/FN/TN over paired/unpaired positions
    def pairs(s):
        stack, paired = [], {}
        for i,c in enumerate(s):
            if c == "(": stack.append(i)
            elif c == ")":
                j = stack.pop(); paired[i]=j; paired[j]=i
        return paired
    p1, p2 = pairs(ss), pairs(native_ss)
    L = len(ss)
    tp=fp=fn=tn=0
    for i in range(L):
        a = 1 if i in p1 else 0
        b = 1 if i in p2 else 0
        tp += (a==1 and b==1); tn += (a==0 and b==0)
        fp += (a==1 and b==0); fn += (a==0 and b==1)
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-9)
    return float(((tp*tn - fp*fn) / denom) if denom>0 else 0.0)

def run_eval(cfg_path: str):
    with open(cfg_path, "r") as f: cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 0)))
    dev = torch.device(f"cuda:{cfg.get('gpu',0)}" if torch.cuda.is_available() else "cpu")

    # Build + load
    model = build_model(cfg)
    model = load_checkpoint(model, cfg["checkpoint_path"], dev)

    # PDB(s)
    eval_cfg = cfg["eval"]; feat_cfg = cfg["featurizer"]
    out_fa = eval_cfg["output_fasta"]; os.makedirs(os.path.dirname(out_fa), exist_ok=True)

    pdbs = []
    if eval_cfg.get("directory"):
        for fn in os.listdir(eval_cfg["directory"]):
            if fn.endswith(".pdb"): pdbs.append(os.path.join(eval_cfg["directory"], fn))
    else:
        pdbs.append(eval_cfg["pdb_filepath"])

    with open(out_fa, "w") as fout:
        for pdb in pdbs:
            data, l2n, n2l = featurize_pdb(pdb, dev, feat_cfg)
            native_tokens = data.seq.clone()
            native_seq = "".join(n2l[int(x)] for x in native_tokens.tolist())

            ll = sequence_ll(model, data, native_tokens)
            ppl = perplexity_from_ll(ll, length=native_tokens.numel())

            print(f"[{os.path.basename(pdb)}] native PPL={ppl:.3f}")
            fout.write(fasta_entry(
                f"input_sequence, checkpoint={os.path.basename(cfg['checkpoint_path'])}, seed={cfg['seed']}",
                native_seq
            )+"\n")

            # sampling
            T = float(eval_cfg["temperature"])
            n = int(eval_cfg["n_samples"]); seed0 = int(eval_cfg["seed"])
            for k in range(n):
                samp = sample_sequence(model, data, T, n2l, seed0 + k)
                # we can compute model PPL for sampled sequence, too
                toks = torch.tensor([l2n[c] for c in samp], device=dev)
                ll_s = sequence_ll(model, data, toks)
                ppl_s = perplexity_from_ll(ll_s, len(samp))
                header = f"sample={k}, seed={seed0+k}, temperature={T}, perplexity={ppl_s:.4f}"
                fout.write(fasta_entry(header, samp)+"\n")

            # optional EternaFold MCC (requires native SS; if you have it, pass it here)
            if bool(eval_cfg.get("compute_eterna_mcc", False)):
                mcc = eternafold_mcc(native_seq, native_ss=None)
                print(f"[{os.path.basename(pdb)}] EternaFold MCC proxy = {mcc}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="dpo/configs/eval.yaml")
    args = ap.parse_args()
    run_eval(args.config)
