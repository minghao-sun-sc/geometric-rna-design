import torch
import torch.nn.functional as F
from typing import Optional


def seq_logprob(model, graph, seq, max_len: Optional[int] = None):
    """
    Teacher-forced log-probability of a full sequence under the AR model.
    Returns: scalar logprob (sum over positions).
    """
    # set sequence into graph (teacher forcing)
    graph = graph.clone()
    if max_len is not None:
        seq = seq[:max_len]
        graph.seq = graph.seq[:max_len]
        # we also need to mask graph nodes/edges down to max_len -> build subgraph
        # Simple & safe: recompute edge mask for nodes < max_len
        node_mask = torch.arange(graph.node_s.size(0), device=seq.device) < len(seq)
        ei = graph.edge_index
        ekeep = node_mask[ei[0]] & node_mask[ei[1]]
        graph.edge_index = ei[:, ekeep]
        graph.node_s = graph.node_s[node_mask]
        graph.node_v = graph.node_v[node_mask]
        graph.mask_confs = graph.mask_confs[node_mask]
        graph.mask_coords = graph.mask_coords[node_mask]
        graph.edge_s = graph.edge_s[ekeep]
        graph.edge_v = graph.edge_v[ekeep]
    graph.seq = seq

    logits = model.forward(graph)  # [L, 4]
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, seq.view(-1,1)).sum()
    return lp


def ce_on_sequence(model, graph, seq, max_len: Optional[int] = None):
    if max_len is not None and len(seq) > max_len:
        # replicate max_len trimming from seq_logprob
        graph = graph.clone()
        seq = seq[:max_len]
        node_mask = torch.arange(graph.node_s.size(0), device=seq.device) < len(seq)
        ei = graph.edge_index
        ekeep = node_mask[ei[0]] & node_mask[ei[1]]
        graph.edge_index = ei[:, ekeep]
        graph.node_s = graph.node_s[node_mask]
        graph.node_v = graph.node_v[node_mask]
        graph.mask_confs = graph.mask_confs[node_mask]
        graph.mask_coords = graph.mask_coords[node_mask]
        graph.edge_s = graph.edge_s[ekeep]
        graph.edge_v = graph.edge_v[ekeep]
    else:
        graph = graph

    graph = graph.clone()
    graph.seq = seq
    logits = model.forward(graph)
    loss = F.cross_entropy(logits, seq, reduction="mean")
    return loss


def dpo_step_losses(model, ref_model, batch, beta: float = 0.1, label_smoothing: float = 0.0, max_len: Optional[int] = None):
    """
    Computes DPO loss and preference accuracy for a single pair.
    """
    # policy log-probs
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    # reference log-probs (no grad)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    # DPO objective per pair
    pi_diff  = (lpw - lpl)
    ref_diff = (lrw - lrl)
    # minus log-sigmoid for minimization
    dpo_loss = -F.logsigmoid(beta * (pi_diff - ref_diff))

    # metrics
    pref_acc = (pi_diff > 0).float().mean()
    margin = pi_diff.detach().item()

    return {
        "loss_dpo": dpo_loss,
        "pref_acc": pref_acc,
        "margin":   pi_diff.detach(),
    }
