# dpo/trainer.py
import os, math, numpy as np, torch, wandb
from torch.optim import AdamW
from dpo.losses import dpo_losses
from dpo.ref_manager import clone_as_reference
from dpo.utils import get_scheduler

def build_model_from_config(config):
    from src.models import AutoregressiveMultiGNNv1, NonAutoregressiveMultiGNNv1
    model_cls = AutoregressiveMultiGNNv1 if config["model"] == "ARv1" else NonAutoregressiveMultiGNNv1
    return model_cls(
        node_in_dim=tuple(config["node_in_dim"]),
        node_h_dim=tuple(config["node_h_dim"]),
        edge_in_dim=tuple(config["edge_in_dim"]),
        edge_h_dim=tuple(config["edge_h_dim"]),
        num_layers=int(config["num_layers"]),
        drop_rate=float(config["drop_rate"]),
        out_dim=int(config["out_dim"]),
    )

def _to_device_batch(batch_tuple, device):
    batch, y_w, y_l, w, node_mask, gids = batch_tuple
    return (
        batch.to(device),
        y_w.to(device),
        y_l.to(device),
        w.to(device),
        (node_mask.to(device) if node_mask is not None else None),
        gids
    )

def train_dpo(config, train_loader, val_loader, device):
    # === Build & load policy ===
    policy = build_model_from_config(config).to(device)
    model_path = config.get("model_path", "")
    if model_path:
        policy.load_state_dict(torch.load(model_path, map_location="cpu"))

    # === Build frozen reference ===
    reference = clone_as_reference(lambda: build_model_from_config(config), policy.state_dict(), device)

    # === Optim & sched ===
    opt_conf = config["optim"]
    optim = AdamW(policy.parameters(),
                  lr=float(opt_conf["lr"]),
                  betas=tuple(opt_conf["betas"]),
                  weight_decay=float(opt_conf["weight_decay"]))
    sched = get_scheduler(
        optim,
        name=opt_conf.get("scheduler", "none"),
        cosine_t0_steps=int(opt_conf.get("cosine_t0_steps", 2000)),
        cosine_tmult=int(opt_conf.get("cosine_tmult", 2)),
    )

    # === Train params ===
    tr_conf = config["train"]
    grad_accum = int(tr_conf["grad_accum_steps"])
    max_rounds = int(tr_conf["rounds"])
    epochs_per_round = int(tr_conf["epochs_per_round"])
    val_every = int(tr_conf["val_every_steps"])
    ckpt_every = int(tr_conf["ckpt_every_steps"])
    grad_clip = float(tr_conf["grad_clip"])
    save_dir = tr_conf["save_dir"]

    loss_conf = config["loss"]
    beta       = float(loss_conf["beta"])
    lambda_sft = float(loss_conf["lambda_sft"])
    length_norm= bool(loss_conf["length_norm"])

    global_step = 0      # optimizer steps (not micro-steps)
    micro_step = 0

    for rnd in range(max_rounds):
        wandb.log({"round": rnd})
        for epoch in range(epochs_per_round):
            policy.train()
            running = {"loss": [], "loss_dpo": [], "loss_sft": [], "margin": []}

            optim.zero_grad(set_to_none=True)
            for batch_tuple in train_loader:


                
                batch, y_w, y_l, w, node_mask, _ = _to_device_batch(batch_tuple, device)

                # Forward & loss (per-graph reduced)
                loss, loss_dpo, loss_sft, margin = dpo_losses(
                    policy, reference, batch, y_w, y_l, w,
                    beta=beta, lambda_sft=lambda_sft, length_norm=length_norm, node_mask=node_mask
                )
                # Scale by grad_accum for proper accumulation
                (loss / grad_accum).backward()
                micro_step += 1

                running["loss"].append(loss.item())
                running["loss_dpo"].append(loss_dpo.item())
                running["loss_sft"].append(loss_sft.item())
                running["margin"].append(margin.item())

                if micro_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    if sched is not None:
                        sched.step(global_step)
                    global_step += 1

                    # Logging (optimizer step)
                    if wandb.run is not None:
                        wandb.log({
                            "step": global_step,
                            "train/loss":   float(np.mean(running["loss"][-grad_accum:])),
                            "train/loss_dpo": float(np.mean(running["loss_dpo"][-grad_accum:])),
                            "train/loss_sft": float(np.mean(running["loss_sft"][-grad_accum:])),
                            "train/margin": float(np.mean(running["margin"][-grad_accum:])),
                            "lr": optim.param_groups[0]["lr"],
                            "epoch": epoch + 1 + rnd*epochs_per_round,
                        })

                    # Validation trigger on optimizer steps
                    if (global_step % val_every == 0) and (val_loader is not None):
                        policy.eval()
                        v_losses, v_margins = [], []
                        with torch.no_grad():
                            for vtuple in val_loader:
                                vbatch, vy_w, vy_l, vw, vnode_mask, _ = _to_device_batch(vtuple, device)
                                vloss, vloss_dpo, vloss_sft, vmargin = dpo_losses(
                                    policy, reference, vbatch, vy_w, vy_l, vw,
                                    beta=beta, lambda_sft=lambda_sft, length_norm=length_norm, node_mask=vnode_mask
                                )
                                v_losses.append(vloss.item()); v_margins.append(vmargin.item())
                        if wandb.run is not None:
                            wandb.log({
                                "val/loss": float(np.mean(v_losses)),
                                "val/margin": float(np.mean(v_margins)),
                                "val/step": global_step
                            })
                        policy.train()

                    # Checkpointing
                    if global_step % ckpt_every == 0:
                        os.makedirs(save_dir, exist_ok=True)
                        ckpt_path = os.path.join(save_dir, f"ckpt_step{global_step}.pt")
                        torch.save(policy.state_dict(), ckpt_path)
                        if wandb.run is not None:
                            wandb.log({"ckpt_path": ckpt_path})

        # === End of round: reference <- policy (frozen) ===
        reference = clone_as_reference(lambda: build_model_from_config(config), policy.state_dict(), device)

    # Final save
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "final_policy.pt")
    torch.save(policy.state_dict(), final_path)
    if wandb.run is not None:
        wandb.run.summary["final_policy"] = final_path
