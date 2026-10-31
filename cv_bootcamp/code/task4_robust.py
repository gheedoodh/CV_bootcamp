import argparse
import json
import os

import torch.nn as nn

from common import (
    ELRLoss, GCELoss, WarmupThenLoss,
    add_common_args, env_info, get_device, load_data, load_resnet18,
    make_dirs, make_transforms, run_training, write_predictions,
)


# ---------------------------------------------------------------------------
# Method registry: (name, loss_factory)
# Each factory takes (num_train_examples, args) and returns an nn.Module.
# Adding a method here automatically ships it as `predictions/t4_<name>.csv`.
# ---------------------------------------------------------------------------

def _gce_factory(n, args):
    return WarmupThenLoss(args.warmup_epochs, GCELoss(q=args.gce_q))


def _ls_factory(n, args):
    return nn.CrossEntropyLoss(label_smoothing=args.ls_smoothing)


def _elr_factory(n, args):
    return ELRLoss(n, num_classes=10, lam=args.elr_lam, beta=args.elr_beta)


METHODS = [
    ("gce", _gce_factory),
    ("ls",  _ls_factory),
    ("elr", _elr_factory),
]


def main():
    parser = argparse.ArgumentParser(description="Task 4 - Robust Training")
    add_common_args(parser)

    # Method hyperparameters, defaulted to what the notebook used.
    parser.add_argument("--gce_q", type=float, default=0.7)
    parser.add_argument("--ls_smoothing", type=float, default=0.27)
    parser.add_argument("--elr_lam", type=float, default=3.0)
    parser.add_argument("--elr_beta", type=float, default=0.7)
    parser.set_defaults(warmup_epochs=13)   # used by GCE only

    # Allow restricting to a subset of methods if you want a quick rerun.
    parser.add_argument("--only", nargs="+", default=None,
                        help="subset of methods to train, e.g. --only elr")

    args = parser.parse_args()

    make_dirs(args.out_dir)
    logs = os.path.join(args.out_dir, "logs")
    preds_dir = os.path.join(args.out_dir, "predictions")

    data = load_data(args.data_dir)
    train_tf, eval_tf = make_transforms(data.mean, data.std)
    resnet = load_resnet18(args.data_dir)
    device = get_device()

    selected = METHODS if args.only is None else [
        (n, f) for n, f in METHODS if n in set(args.only)
    ]
    if not selected:
        raise SystemExit(f"--only={args.only} matched no method; "
                         f"valid: {[n for n, _ in METHODS]}")

    print(f"Training {len(selected)} method(s): {[n for n, _ in selected]}")
    print(f"Recipe: epochs={args.epochs}  lr={args.lr}  wd={args.weight_decay}  "
          f"momentum={args.momentum}  batch={args.batch_size}  seed={args.seed}")

    # ------------------------------------------------------------------
    # Train each method
    # ------------------------------------------------------------------
    per_method = {}
    for name, factory in selected:
        print(f"\n== t4_{name} ==")
        res = run_training(
            args=args, tag=f"t4_{name}",
            resnet_cls=resnet, device=device,
            train_images=data.train_images, train_labels=data.train_noisy,
            val_images=data.val_images, val_labels=data.val_noisy,
            test_images=data.test_images,
            train_tf=train_tf, eval_tf=eval_tf,
            loss_factory=lambda n, f=factory: f(n, args),
            select_from_epoch=0,
            track_train=False,           # not needed for Task 4
        )
        write_predictions(res.test_preds[res.best_epoch],
                          os.path.join(preds_dir, f"t4_{name}.csv"))
        per_method[name] = {
            "best_epoch": int(res.best_epoch),
            "best_val_noisy_acc": float(res.best_val),
            "final_val_noisy_acc": float(res.final_val),
            "n_epochs": int(len(res.history)),
            "test_file": f"predictions/t4_{name}.csv",
        }
        print(f"[{name}] best epoch={res.best_epoch}  "
              f"best noisy-val={res.best_val:.4f}  "
              f"final noisy-val={res.final_val:.4f}")

    # ------------------------------------------------------------------
    # Pick the best method (highest best noisy-val) and write summary
    # ------------------------------------------------------------------
    best_name = max(per_method, key=lambda n: per_method[n]["best_val_noisy_acc"])
    best_val = per_method[best_name]["best_val_noisy_acc"]

    # Method config snapshot for Task 5.
    config = {
        "gce":  {"q": args.gce_q, "warmup_epochs": args.warmup_epochs},
        "ls":   {"smoothing": args.ls_smoothing},
        "elr":  {"lam": args.elr_lam, "beta": args.elr_beta},
    }

    summary = {
        "methods_trained": list(per_method.keys()),
        "budget": {
            "training_runs_used": len(per_method),
            "training_runs_limit": 3,
            "networks_per_method": 1,
            "epochs_per_network": args.epochs,
            "epochs_limit": 60,
        },
        "recipe": {
            "optimizer": "SGD",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": f"CosineAnnealingLR(T_max={args.scheduler_tmax or args.epochs})",
            "batch_size": args.batch_size,
            "augmentation": "RandomHorizontalFlip + RandomCrop(32, padding=4)",
            "seed": args.seed,
        },
        "selection_signal": "noisy validation accuracy only",
        "per_method": per_method,
        "best_method": best_name,
        "best_method_val_noisy_acc": float(best_val),
        "best_method_config": config[best_name],
        "env": env_info(),
        "note": (
            "All three methods share the Task 1 recipe; only the loss changed. "
            "Best is the method with the highest noisy validation accuracy, per "
            "the problem statement. Task 5 reads `best_method` and "
            "`best_method_config` from this file."
        ),
    }
    with open(os.path.join(logs, "t4_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Console leaderboard
    # ------------------------------------------------------------------
    print("\n== Task 4 leaderboard (best noisy-val) ==")
    for name in sorted(per_method, key=lambda n: -per_method[n]["best_val_noisy_acc"]):
        row = per_method[name]
        marker = "  <-- best" if name == best_name else ""
        print(f"  {name:4s}  best_epoch={row['best_epoch']:2d}  "
              f"best_noisy_val={row['best_val_noisy_acc']:.4f}  "
              f"final_noisy_val={row['final_val_noisy_acc']:.4f}{marker}")
    print(f"\nBest method written to logs/t4_summary.json: {best_name}")

    print("\nWrote:")
    for name in per_method:
        print(f"  predictions/t4_{name}.csv")
    print("  logs/t4_summary.json")
    print("  logs/t4_<method>_history.csv  +  *_all_epochs.npy")


if __name__ == "__main__":
    main()