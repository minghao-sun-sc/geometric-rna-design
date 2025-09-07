import argparse, yaml, torch
from dpo.model_factory import build_model
from dpo.lora import apply_lora, mark_only_lora_as_trainable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = build_model(cfg)

    lcfg = cfg.get("lora", {})
    wrapped = apply_lora(model,
                         enabled=lcfg.get("enabled", False),
                         r=int(lcfg.get("r", 8)),
                         alpha=int(lcfg.get("alpha", 16)),
                         dropout=float(lcfg.get("dropout", 0.0)),
                         target_modules=tuple(lcfg.get("target_modules", [""])),
                         train_bias=lcfg.get("train_bias", "none"))
    print(f"[LoRA] wrapped modules: {wrapped}")
    if wrapped > 0:
        mark_only_lora_as_trainable(model, train_bias=lcfg.get("train_bias","none"))
    n_train = sum(p.requires_grad for p in model.parameters())
    n_total  = sum(1 for _ in model.parameters())
    print(f"[LoRA] trainable params: {n_train} / {n_total}")

if __name__ == "__main__":
    main()
