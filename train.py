"""Train one model on SHU-MI (subjects 1-15 / 16-20 / 21-25), select the epoch by val AUROC, test once.

python train.py --model cbramod --seed 0
python train.py --model cbramod --seed 0 --name e3_cbramod_scratch --set pretrained=false

Writes TensorBoard logs to runs/{name}/ and results to results/{name}/seed{seed}.json (+ test predictions).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.cbramod import CBraModClassifier
from src.data import make_loaders, train_subjects
from src.metrics import binary_metrics, per_subject_metrics
from src.simpleconv import EEGSimpleConv
from src.utils import get_device, load_config, set_seed

N_CHANNELS = 32
N_PATCHES = 4  # 4 s at 200 Hz = 4 patches of 1 s
SFREQ = 200


def build_model(cfg):
    if cfg["model"] == "cbramod":
        weights = cfg["weights"] if cfg["pretrained"] else None
        model = CBraModClassifier(N_CHANNELS, N_PATCHES, cfg["head"], cfg["dropout"], weights)
        model.backbone.requires_grad_(not cfg["frozen"])  # official --frozen: only the head is trained (E6)
        return model
    if cfg["model"] == "simpleconv":
        return EEGSimpleConv(N_CHANNELS, SFREQ, cfg["fm"], cfg["n_convs"], cfg["resampling"], cfg["kernel_size"])
    raise ValueError(f"unknown model: {cfg['model']}")


def make_optimizer(model, cfg):
    if cfg["head_lr"] is None:
        groups = [{"name": "all", "params": list(model.parameters()), "lr": cfg["lr"]}]
    else:  # official multi_lr: separate lr for the pretrained backbone and the new head
        named = list(model.named_parameters())
        groups = [
            {"name": "backbone", "params": [p for n, p in named if n.startswith("backbone.")], "lr": cfg["lr"]},
            {"name": "head", "params": [p for n, p in named if not n.startswith("backbone.")], "lr": cfg["head_lr"]},
        ]
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits, ys, subjects = [], [], []
    for x, y, subject in loader:
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
        subjects.append(subject)
    logits, y = torch.cat(logits), torch.cat(ys)
    prob = torch.sigmoid(logits).numpy()
    metrics = binary_metrics(y.numpy(), prob)
    metrics["loss"] = F.binary_cross_entropy_with_logits(logits, y).item()
    return metrics, prob, y.numpy(), torch.cat(subjects).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cbramod", "simpleconv"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="experiment name for runs/ and results/ (default: model)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = parser.parse_args()
    cfg = load_config(args.model, args.set)
    name = args.name or args.model

    set_seed(args.seed)
    torch.set_num_threads(4)  # default (1 per core) oversubscribes the CPU when runs share a machine; data prep stalls the GPU
    device = get_device()
    subjects = train_subjects(cfg["n_train_subjects"], args.seed)
    loaders = make_loaders(cfg["batch_size"], subjects, pin_memory=device.type == "cuda", ea=cfg["ea"])
    model = build_model(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] * len(loaders["train"]), eta_min=cfg["min_lr"]
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} seed {args.seed} on {device}: {n_params / 1e6:.2f}M params, train subjects {subjects}")

    writer = SummaryWriter(f"runs/{name}/seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    writer.add_text("config", f"```\n{yaml.safe_dump(cfg)}```")
    best_auroc, step = -1.0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        model.train()
        for x, y, _ in loaders["train"]:
            loss = F.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()
            scheduler.step()
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            for group in optimizer.param_groups:
                writer.add_scalar(f"lr/{group['name']}", group["lr"], step)
            step += 1

        val, *_ = evaluate(model, loaders["val"], device)
        for key, value in val.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        if val["auroc"] > best_auroc:
            best_auroc, best_epoch, best_val = val["auroc"], epoch, val
            best_state = copy.deepcopy(model.state_dict())
        print(f"epoch {epoch:3d}  val auroc {val['auroc']:.4f}  bacc {val['bacc']:.4f}  {time.time() - start:.0f}s")

    # Test once, with the best-val checkpoint
    model.load_state_dict(best_state)
    test, prob, y, subject = evaluate(model, loaders["test"], device)
    test_by_subject = per_subject_metrics(y, prob, subject)
    for key, value in test.items():
        writer.add_scalar(f"test/{key}", value, best_epoch)
    for s, metrics in test_by_subject.items():
        writer.add_scalar(f"test_by_subject/auroc_sub{s}", metrics["auroc"], best_epoch)
    writer.close()
    print(f"best epoch {best_epoch}  test " + "  ".join(f"{k} {v:.4f}" for k, v in test.items()))

    os.makedirs(f"results/{name}", exist_ok=True)
    result = {
        "name": name, "seed": args.seed, "config": cfg, "train_subjects": subjects, "n_params": n_params,
        "best_epoch": best_epoch, "val": best_val, "test": test, "test_by_subject": test_by_subject,
    }
    with open(f"results/{name}/seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez(f"results/{name}/seed{args.seed}_test_preds.npz", prob=prob, y=y, subject=subject)


if __name__ == "__main__":
    main()
