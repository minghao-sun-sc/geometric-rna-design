# dpo/losses.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

@torch.no_grad()
def _per_graph_lengths(batch) -> torch.Tensor:
    # [G] number of nodes per graph
    return torch.bincount(batch.batch, minlength=batch.num_graphs).clamp_min(1)

def _ll_per_graph_from_logits(
    logits: torch.Tensor,           # [N, 4]
    tokens: torch.Tensor,           # [N]
    batch,                          # PyG Batch with .batch and .num_graphs
    length_norm: bool,
    node_mask: Optional[torch.Tensor] = None,  # [N] float (0/1) optional
) -> torch.Tensor:
    logp = logits.log_softmax(-1)
    idx = torch.arange(logits.size(0), device=logits.device)
    ll_node = logp[idx, tokens]                      # [N]
    if node_mask is not None:
        ll_node = ll_node * node_mask
    ll_graph = scatter_add(ll_node, batch.batch, dim=0, dim_size=batch.num_graphs)
    if length_norm:
        if node_mask is not None:
            denom = scatter_add(node_mask, batch.batch, dim=0, dim_size=batch.num_graphs).clamp_min(1.0)
        else:
            denom = _per_graph_lengths(batch).to(ll_graph.dtype)
        ll_graph = ll_graph / denom
    return ll_graph  # [G]

def _ll_per_graph(
    model, batch, tokens, length_norm: bool, no_grad: bool, node_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    saved = batch.seq
    try:
        batch.seq = tokens
        if no_grad:
            with torch.no_grad():
                logits = model(batch)
        else:
            logits = model(batch)
    finally:
        batch.seq = saved
    return _ll_per_graph_from_logits(logits, tokens, batch, length_norm, node_mask)

def dpo_losses(
    policy, reference, batch,
    y_w: torch.Tensor, y_l: torch.Tensor, weights: torch.Tensor,
    beta: float, lambda_sft: float, length_norm: bool,
    node_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns scalar means: (loss_total, loss_dpo, loss_sft, margin)
    Each computed per-graph then averaged.
    """
    ll_pw = _ll_per_graph(policy,    batch, y_w, length_norm, no_grad=False, node_mask=node_mask)
    ll_pl = _ll_per_graph(policy,    batch, y_l, length_norm, no_grad=False, node_mask=node_mask)
    ll_rw = _ll_per_graph(reference, batch, y_w, length_norm, no_grad=True,  node_mask=node_mask)
    ll_rl = _ll_per_graph(reference, batch, y_l, length_norm, no_grad=True,  node_mask=node_mask)

    margin = beta * ((ll_pw - ll_rw) - (ll_pl - ll_rl))  # [G]
    loss_dpo = -F.logsigmoid(margin) * weights
    loss_sft = -ll_pw                                    # winner-only anchor
    loss = loss_dpo.mean() + lambda_sft * loss_sft.mean()
    return loss, loss_dpo.mean(), loss_sft.mean(), margin.mean()
