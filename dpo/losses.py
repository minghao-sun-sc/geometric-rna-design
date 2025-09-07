# dpo/losses.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

# ---------- helpers ----------

def _as_logits(model_out) -> torch.Tensor:
    """
    Make a best-effort to extract token logits from various model outputs.
    Expected final shape: [N_total_nodes, C] or [N_total_nodes, 1, C].
    """
    if isinstance(model_out, torch.Tensor):
        logits = model_out
    elif isinstance(model_out, dict):
        if "logits_seq" in model_out:
            logits = model_out["logits_seq"]
        elif "logits" in model_out:
            logits = model_out["logits"]
        else:
            raise KeyError("Model output dict missing 'logits_seq' or 'logits'.")
    elif isinstance(model_out, (tuple, list)):
        logits = model_out[0]
    else:
        raise TypeError(f"Unrecognized model output type: {type(model_out)}")

    # squeeze any singleton dims after the node dimension
    while logits.dim() >= 3 and logits.size(-3) == 1:
        logits = logits.squeeze(-3)
    # finally ensure [N, C]
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    return logits


def _get_batch_index(batch) -> torch.Tensor:
    """
    Try to obtain a per-node -> graph index mapping.
    Assumes PyG-style `batch.graph.batch` if available.
    """
    g = None
    if isinstance(batch, dict) and "graph" in batch:
        g = batch["graph"]
    elif hasattr(batch, "graph"):
        g = batch.graph
    else:
        # last resort: sometimes the batch itself is a PyG Batch
        g = batch

    if hasattr(g, "batch") and isinstance(g.batch, torch.Tensor):
        return g.batch
    if isinstance(batch, dict) and "node_batch" in batch:
        return batch["node_batch"]
    raise RuntimeError(
        "Could not infer per-node graph indices ('batch'). "
        "Expected PyG Batch with `.batch` vector."
    )


def _compute_per_graph_logp(
    model: torch.nn.Module,
    data_batch,
    targets: torch.Tensor,
    node_mask: torch.Tensor,
    length_norm: bool,
    no_grad: bool,
) -> torch.Tensor:
    """
    Returns per-graph total (or length-normalized) log-likelihood for the given targets.
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        # Set the target sequence for the model to use
        data_batch.seq = targets
        logits = _as_logits(model(data_batch))  # [N, C] or [N, 1, C]
        if logits.dim() == 3:
            # [N, 1, C] -> [N, C]
            logits = logits.squeeze(-2)

        # shapes
        # targets: [N]
        # node_mask: [N] (bool or 0/1)
        # batch_index: [N] in [0..num_graphs-1]
        batch_index = _get_batch_index(data_batch).to(logits.device)

        # log_probs per node for the true class
        log_probs = F.log_softmax(logits, dim=-1)
        per_node = torch.gather(log_probs, dim=-1, index=targets.to(logits.device).unsqueeze(-1)).squeeze(-1)

        # mask invalid nodes
        mask = node_mask.to(logits.device).to(per_node.dtype)
        
        # Validate tensor sizes before multiplication
        if per_node.shape[0] != mask.shape[0]:
            raise RuntimeError(
                f"Size mismatch in DPO loss: per_node has {per_node.shape[0]} elements "
                f"but mask has {mask.shape[0]} elements. "
                f"targets shape: {targets.shape}, logits shape: {logits.shape}, "
                f"node_mask shape: {node_mask.shape}"
            )
        
        per_node = per_node * mask

        # sum tokens per graph
        n_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        per_graph_sum = torch.zeros(n_graphs, device=per_node.device)
        per_graph_sum.scatter_add_(0, batch_index, per_node)

        if length_norm:
            # number of valid tokens per graph
            ones = mask
            per_graph_len = torch.zeros(n_graphs, device=per_node.device)
            per_graph_len.scatter_add_(0, batch_index, ones)
            # avoid div-by-zero
            per_graph_len = torch.clamp(per_graph_len, min=1.0)
            per_graph_sum = per_graph_sum / per_graph_len

        return per_graph_sum  # [G]


def _compute_sft_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    node_mask: torch.Tensor,
    length_norm: bool,
) -> torch.Tensor:
    """
    Token-level cross-entropy reduced to mean over graphs.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-2)  # [N, 1, C] -> [N, C]

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    mask = node_mask.to(nll.dtype)
    nll = nll * mask

    # aggregate by graph
    # we assume logits & batch_index on same device
    # get batch index through logits' device via a tiny trick: use targets device
    # (all data tensors should already be moved together by Lightning)
    # but call helper to be safe:
    # Note: we need access to the batch to get per-graph splits; to keep API clean,
    # SFT loss is computed outside where we already have `batch`. Hence this function
    # expects node-wise tensors only and will reduce across all nodes.
    # So instead of per-graph reduction here, we return mean over *masked nodes*.
    denom = torch.clamp(mask.sum(), min=1.0)
    if length_norm:
        # normalizing by number of valid tokens is effectively the same in this scope
        return nll.sum() / denom
    else:
        # without length-norm, we still average to keep scale consistent across batches
        return nll.sum() / denom


# ---------- main step ----------

def dpo_sft_step(
    model: torch.nn.Module,
    data_batch,
    *,
    beta: float,
    lambda_sft: float,
    length_norm: bool,
    ref_model: torch.nn.Module | None = None,
    train: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    One combined DPO+SFT step on a batch of preference pairs.

    Expected keys in `data_batch`:
      - 'y_w', 'y_l' : Long tensors of size [N_total_nodes] (winner/loser tokens)
      - 'node_mask'  : Bool/0-1 mask [N_total_nodes]
      - 'graph' PyG Batch (or anything w/ .batch) to aggregate per-graph
    """
    device = next(model.parameters()).device

    # pull labels/masks
    if isinstance(data_batch, dict):
        y_w = data_batch["y_w"].to(device).long()
        y_l = data_batch["y_l"].to(device).long()
        node_mask = data_batch["node_mask"].to(device)
    else:
        # or attribute-style
        y_w = getattr(data_batch, "y_w").to(device).long()
        y_l = getattr(data_batch, "y_l").to(device).long()
        node_mask = getattr(data_batch, "node_mask").to(device)

    # ---- policy log-likelihoods per graph ----
    logp_pol_w = _compute_per_graph_logp(model, data_batch, y_w, node_mask, length_norm, no_grad=not train)
    logp_pol_l = _compute_per_graph_logp(model, data_batch, y_l, node_mask, length_norm, no_grad=not train)

    # ---- reference (always frozen, no_grad) ----
    if ref_model is None:
        raise ValueError("ref_model must be provided for DPO.")
    logp_ref_w = _compute_per_graph_logp(ref_model, data_batch, y_w, node_mask, length_norm, no_grad=True)
    logp_ref_l = _compute_per_graph_logp(ref_model, data_batch, y_l, node_mask, length_norm, no_grad=True)

    # ---- DPO objective (policy-only gradients) ----
    # diff per graph
    advantage = (logp_pol_w - logp_pol_l) - (logp_ref_w - logp_ref_l)  # [G]
    loss_dpo = -F.logsigmoid(beta * advantage).mean()

    # ---- optional SFT auxiliary (policy-only) ----
    # compute token CE on winners; you can also add losers if desired
    with torch.no_grad():
        pol_logits = _as_logits(model(data_batch))
        if pol_logits.dim() == 3:
            pol_logits = pol_logits.squeeze(-2)
    # re-enable grad for SFT on policy
    pol_logits.requires_grad_(True)
    loss_sft = _compute_sft_loss(pol_logits, y_w, node_mask, length_norm)

    loss = loss_dpo + lambda_sft * loss_sft

    # simple accuracy (token-level over masked nodes) for logging
    with torch.no_grad():
        pred = pol_logits.argmax(dim=-1)
        acc_w = ((pred == y_w) & node_mask.bool()).sum().float() / torch.clamp(node_mask.sum(), min=1)

    return {
        "loss": loss,
        "loss_dpo": loss_dpo.detach(),
        "loss_sft": loss_sft.detach(),
        "acc_w": acc_w,
        "adv_mean": advantage.mean().detach(),
    }
