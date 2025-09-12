import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.data import Batch as GeometricBatch


def seq_logprob(model, graph, seq, max_len: Optional[int] = None):
    """
    Teacher-forced log-probability of a full sequence under the AR model.
    Handles both single graphs and batched graphs.
    Returns: scalar logprob (sum over positions) or tensor of logprobs for batch.
    """
    # Check if we have a batch
    is_batched = isinstance(graph, GeometricBatch) and seq.dim() == 2
    
    if not is_batched:
        # Original single-graph logic
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
    else:
        # Batched graph logic - process each sample in the batch
        batch_size = seq.size(0)
        log_probs = []
        
        # Split the batched graph back into individual graphs
        graphs = graph.to_data_list()
        
        for i in range(batch_size):
            curr_seq = seq[i]
            # Remove padding (-1 values)
            valid_mask = curr_seq != -1
            curr_seq = curr_seq[valid_mask]
            
            # Process individual graph
            lp = seq_logprob(model, graphs[i], curr_seq, max_len=max_len)
            log_probs.append(lp)
        
        # Stack log probs
        return torch.stack(log_probs)


def ce_on_sequence(model, graph, seq, max_len: Optional[int] = None):
    # Check if we have a batch
    is_batched = isinstance(graph, GeometricBatch) and seq.dim() == 2
    
    if not is_batched:
        # Original single-graph logic
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
    else:
        # Batched graph logic
        batch_size = seq.size(0)
        losses = []
        graphs = graph.to_data_list()
        
        for i in range(batch_size):
            curr_seq = seq[i]
            # Remove padding (-1 values)
            valid_mask = curr_seq != -1
            curr_seq = curr_seq[valid_mask]
            
            # Process individual graph
            loss = ce_on_sequence(model, graphs[i], curr_seq, max_len=max_len)
            losses.append(loss)
        
        # Average losses across batch
        return torch.stack(losses).mean()


def dpo_step_losses(model, ref_model, batch, beta: float = 0.1, label_smoothing: float = 0.0, max_len: Optional[int] = None):
    """
    Computes DPO loss and preference accuracy for single or batched pairs.
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

    # For batched data, average the loss
    if dpo_loss.dim() > 0:
        dpo_loss = dpo_loss.mean()

    # metrics
    pref_acc = (pi_diff > 0).float().mean()
    
    # For margin, handle both single and batched cases
    if pi_diff.dim() == 0:
        margin = pi_diff.detach().item()
    else:
        margin = pi_diff.detach().mean().item()

    return {
        "loss_dpo": dpo_loss,
        "pref_acc": pref_acc,
        "margin":   pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach(),
    }

class SimPOLoss(nn.Module):
    """
    SimPO loss:
      L = -log σ( β*(avg_logp_w - avg_logp_l) - γ )
        = softplus( - (β*(avg_logp_w - avg_logp_l) - γ) )
    """
    def __init__(self, beta: float = 2.0, gamma: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

    @staticmethod
    def _avg_logp(logits, labels, ignore_index=-1):
        # logits: [B, T, V]; labels: [B, T]
        # get per-token negative log-likelihood, then negate to log-prob
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(labels.size())
        logp = -nll

        mask = (labels != ignore_index).float()
        # avoid division by zero if a sequence is fully masked
        lengths = mask.sum(dim=1).clamp_min(1.0)
        avg_logp = (logp * mask).sum(dim=1) / lengths
        return avg_logp  # [B]

    def forward(self, win_logits, win_labels, lose_logits, lose_labels):
        avg_w = self._avg_logp(win_logits, win_labels, self.ignore_index)
        avg_l = self._avg_logp(lose_logits, lose_labels, self.ignore_index)
        z = self.beta * (avg_w - avg_l) - self.gamma
        loss = F.softplus(-z)  # == -logsigmoid(z)
        # metrics for logging
        with torch.no_grad():
            reward_margin = (avg_w - avg_l)
            reward_accuracy = (self.beta * reward_margin > self.gamma).float().mean()
        return loss.mean(), {
            "simpo/avg_logp_w": avg_w.mean().item(),
            "simpo/avg_logp_l": avg_l.mean().item(),
            "simpo/z_margin": z.mean().item(),
            "simpo/reward_acc": reward_accuracy.item(),
        }
