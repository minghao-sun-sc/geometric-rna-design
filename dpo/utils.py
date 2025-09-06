# dpo/utils.py
import os, random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_scheduler(optim, name: str, cosine_t0_steps: int = 2000, cosine_tmult: int = 2):
    name = (name or "none").lower()
    if name == "cosine_wr":
        return CosineAnnealingWarmRestarts(optim, T_0=int(cosine_t0_steps), T_mult=int(cosine_tmult))
    return None

def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
# dpo/utils.py
import os, random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_scheduler(optim, name: str, cosine_t0_steps: int = 2000, cosine_tmult: int = 2):
    name = (name or "none").lower()
    if name == "cosine_wr":
        return CosineAnnealingWarmRestarts(optim, T_0=int(cosine_t0_steps), T_mult=int(cosine_tmult))
    return None

def save_checkpoint(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
