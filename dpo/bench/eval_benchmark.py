from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os, math, csv, time, yaml
from types import SimpleNamespace as SN
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

from src.data.featurizer import RNAGraphFeaturizer
from src.data.data_utils import get_backbone_coords
from dpo.ref_manager import build_model_from_cfg
from dpo.utils import set_seed, load_processed_pt



def _to_sn(o):
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(x) for x in o]
    return o

def load_cfg(path: str) -> SN:
    with open(path, "r") as f:
        return _to_sn(yaml.safe_load(f))


def _load_split(path: str):
    tr, va, te = torch.load(path, map_location="cpu")
    return list(map(int, tr)), list(map(int, va)), list(map(int, te))

@dataclass
class GraphItem:
    gid: str
    graph: Any    # torch_geometric.data.Data
    seq: torch.Tensor  # [L] long 0..3

class GraphDataset(Dataset):
    def __init__(self, processed_pt: str, split_pt: str, split_name: str, feat_cfg: SN, device="cpu"):
        super().__init__()
        self.device = device
        # all_items = _load_processed(processed_pt)
        all_items = load_processed_pt(processed_pt)
        tr, va, te = _load_split(split_pt)
        if split_name == "train":
            idxs = tr
        elif split_name == "val":
            idxs = va
        else:
            idxs = te

        self.items: List[GraphItem] = []
        # featurizer (same choices as training/eval)
        # Force featurizer to use CPU to avoid device mismatch issues
        featurizer_device = "cpu"
        self.featurizer = RNAGraphFeaturizer(
            split=feat_cfg.split,
            radius=feat_cfg.radius, top_k=feat_cfg.top_k,
            num_rbf=feat_cfg.num_rbf, num_posenc=feat_cfg.num_posenc,
            max_num_conformers=feat_cfg.max_num_conformers,
            noise_scale=feat_cfg.noise_scale,
            distance_eps=getattr(feat_cfg, "distance_eps", 1e-3),
            device=featurizer_device
        )
        l2n = self.featurizer.letter_to_num

        for gi in idxs:
            entry = all_items[gi]
            # build coords_list (3-bead backbone) for each conformer
            coords_list = []
            for coords in entry["coords_list"]:
                if isinstance(coords, torch.Tensor):
                    coords_list.append(get_backbone_coords(coords.clone().detach(), entry["sequence"]).numpy())
                else:
                    coords_list.append(get_backbone_coords(torch.tensor(coords), entry["sequence"]).numpy())
            raw = {
                "sequence": entry["sequence"],
                "coords_list": coords_list,
                "sec_struct_list": entry.get("sec_struct_list", ["."*len(entry["sequence"]) for _ in coords_list])
            }
            graph = self.featurizer.featurize(raw).to(device)
            # Use the graph's sequence to ensure length consistency
            seq = graph.seq.to(device)
            gid = entry["id_list"][0] if entry.get("id_list") else f"idx_{gi}"
            self.items.append(GraphItem(gid=gid, graph=graph, seq=seq))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def _human_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def _load_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    # two formats: (A) state_dict directly; (B) {"model": state_dict, ...}
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return model

@torch.no_grad()
def eval_ckpt(cfg: SN, ds: GraphDataset, ckpt_name: str, ckpt_path: str, device) -> Dict[str, float]:
    model = build_model_from_cfg(cfg.model).to(device)
    _load_weights(model, ckpt_path, device)
    model.eval()

    n_graphs = 0
    total_tokens = 0
    correct_tokens = 0
    exact_ok = 0
    sum_ce = 0.0

    for it in ds:
        # teacher-forced logits on the native sequence
        g = it.graph.clone()
        g.seq = it.seq
        logits = model(g)   # [L, 4]
        # CE and accuracy
        ce = F.cross_entropy(logits, it.seq, reduction="sum")   # NLL (sum over L)
        sum_ce += float(ce.item())
        preds = logits.argmax(dim=-1)
        correct = (preds == it.seq).sum().item()
        correct_tokens += correct
        total_tokens += it.seq.numel()
        exact_ok += 1 if (correct == it.seq.numel()) else 0
        n_graphs += 1

    nll_token = sum_ce / max(1, total_tokens)
    ppl = math.exp(nll_token)
    acc = correct_tokens / max(1, total_tokens)
    exact = exact_ok / max(1, n_graphs)

    return {
        "graphs": n_graphs,
        "token_acc": acc,
        "seq_exact": exact,
        "nll_per_token": nll_token,
        "ppl": ppl
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ds = GraphDataset(cfg.paths.processed_pt, cfg.paths.split_pt, cfg.paths.split_name, cfg.featurizer, device=device)

    # W&B run
    wb = cfg.eval.wandb
    use_wandb = getattr(wb, "enable", False)
    if use_wandb:
        run_name = wb.run_name or f"bench_{cfg.paths.split_name}_{_human_ts()}"
        wandb.init(project=wb.project, entity=wb.entity, name=run_name, tags=wb.tags)

    os.makedirs(cfg.eval.out_dir, exist_ok=True)
    rows = []
    for ck in cfg.paths.checkpoints:
        name, path = ck.name, ck.path
        print(f"\n[eval] {name}  <-  {path}")
        stats = eval_ckpt(cfg, ds, name, path, device)
        row = {
            "ckpt_name": name, "ckpt_path": path, "split": cfg.paths.split_name,
            **{k: float(v) for k, v in stats.items()}
        }
        rows.append(row)
        # print short summary
        print(f"  token_acc={row['token_acc']:.4f}  seq_exact={row['seq_exact']:.4f}  nll/tok={row['nll_per_token']:.4f}  ppl={row['ppl']:.2f}  graphs={row['graphs']}")
        # wandb
        if use_wandb:
            wandb.log({f"{name}/token_acc": row["token_acc"],
                       f"{name}/seq_exact": row["seq_exact"],
                       f"{name}/nll_per_token": row["nll_per_token"],
                       f"{name}/ppl": row["ppl"]})

    # write CSV
    out_csv = cfg.eval.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if rows:
        keys = ["ckpt_name","ckpt_path","split","graphs","token_acc","seq_exact","nll_per_token","ppl"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows: w.writerow({k: r[k] for k in keys})
        print(f"\n[wrote] {out_csv}")

    if use_wandb:
        table = wandb.Table(columns=list(rows[0].keys()) if rows else [])
        for r in rows: table.add_data(*[r[k] for k in r.keys()])
        wandb.log({"benchmark/table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
