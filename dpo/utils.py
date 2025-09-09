import os, json, yaml, random, math, time
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_processed_pt(path):
    obj = torch.load(path, map_location="cpu")
    # supports both torch.save(list) and dict {"data": list}
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        # Handle case where obj is a dict with sequence keys -> convert to list of dicts
        return list(obj.values())
    else:
        # structured like gRNAde's processed.pt (list of dicts)
        return obj


def canonical_id_from_path(pdb_path: str):
    base = os.path.basename(pdb_path)
    if base.endswith(".pdb"):
        base = base[:-4]
    return base


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup = int(warmup_steps)
        self.total = int(total_steps)
        self.min_ratio = float(min_lr_ratio)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / max(1, self.warmup)
        else:
            t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            scale = self.min_ratio + 0.5*(1 - self.min_ratio)*(1 + math.cos(math.pi * (1 - t)))
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.base_lrs[i] * scale


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.n = 0
    @property
    def avg(self):
        return self.sum / max(1, self.n)
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.n += n


def pretty_cfg(cfg):
    # flatten to dict for wandb.config
    def to_plain(o):
        if hasattr(o, "__dict__"):
            return {k: to_plain(v) for k, v in o.__dict__.items()}
        if isinstance(o, dict):
            return {k: to_plain(v) for k, v in o.items()}
        return o
    return to_plain(cfg)


def now_str():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
