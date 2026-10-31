import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import (
    LabeledImages, add_common_args, build_model, get_device, load_data,
    load_resnet18, make_dirs, make_eval_loader, make_folds, make_transforms,
    predict, set_seed,
)


def train_one_fold(args, resnet, device, fold_idx,
                   tr_images, tr_labels, ho_images, ho_labels,
                   val_images, val_noisy, train_tf, eval_tf):
    """Train one fold and return (per-sample CE on held-out, history, val_acc)."""
    fold_seed = args.seed + fold_idx
    set_seed(fold_seed, args.deterministic)

    model = build_model(resnet, device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum,
    )
    tmax = args.scheduler_tmax if args.scheduler_tmax else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)

    tr_ds = LabeledImages(tr_images, tr_labels, train_tf)
    g = torch.Generator()
    g.manual_seed(fold_seed)
    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True,
        generator=g, num_workers=args.num_workers,
    )

    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.fold_epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels, _ in tr_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            with torch.no_grad():
                correct += (logits.argmax(1) == labels).sum().item()
        scheduler.step()
        history.append({
            "fold": fold_idx, "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_acc": correct / seen,
            "lr": optimizer.param_groups[0]["lr"],
        })
        if (epoch + 1) % 5 == 0 or epoch == args.fold_epochs - 1:
            print(f"  [fold {fold_idx} ep {epoch:2d}] "
                  f"loss={history[-1]['train_loss']:.4f} "
                  f"acc={history[-1]['train_acc']:.4f} "
                  f"lr={history[-1]['lr']:.4f}")

    # Score the held-out fold: per-sample CE.
    model.eval()
    ho_ds = LabeledImages(ho_images, ho_labels, eval_tf)
    ho_loader = DataLoader(ho_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    ho_losses = []
    with torch.no_grad():
        for images, labels, _ in tqdm(ho_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            ho_losses.append(
                F.cross_entropy(logits, labels, reduction="none").cpu().numpy())
    ho_scores = np.concatenate(ho_losses)
    assert len(ho_scores) == len(ho_images)

    # Fold model's accuracy on the global 5k noisy val (for the report).
    val_loader = make_eval_loader(val_images, eval_tf)
    val_preds = predict(model, val_loader, device)
    val_acc = float((val_preds == np.asarray(val_noisy)).mean())

    return ho_scores, pd.DataFrame(history), val_acc


def main():
    parser = argparse.ArgumentParser(description="Task 3 - Find the Mislabeled Images")
    add_common_args(parser)
    parser.add_argument("--folds", type=int, default=5,
                        help="number of CV folds (one training run each; budget is 5)")
    parser.add_argument("--fold_epochs", type=int, default=35,
                        help="epochs per fold; the notebook used 35 of the 60-step cosine")
    args = parser.parse_args()

    if args.folds > 5:
        sys.exit(f"--folds={args.folds} exceeds the Task 3 budget of 5 training runs.")
    if args.fold_epochs > args.epochs:
        sys.exit(f"--fold_epochs={args.fold_epochs} > --epochs={args.epochs}; "
                 f"the fold schedule cannot exceed the recipe schedule.")

    make_dirs(args.out_dir)
    logs = os.path.join(args.out_dir, "logs")
    preds_dir = os.path.join(args.out_dir, "predictions")

    data = load_data(args.data_dir)
    train_tf, eval_tf = make_transforms(data.mean, data.std)
    resnet = load_resnet18(args.data_dir)
    device = get_device()

    # Fold splits are stratified by noisy labels only.
    folds = make_folds(data.train_noisy, seed=args.seed, n_splits=args.folds)
    assert len(folds) == args.folds

    # Measurement-only mask; never used for training or selection.
    mislabeled_mask = data.train_noisy != data.train_clean   # (45000,) bool

    n_train = len(data.train_images)
    all_scores_raw = np.full(n_train, np.nan, dtype=np.float64)
    all_scores_z = np.full(n_train, np.nan, dtype=np.float64)

    fold_metrics = []
    for fold_idx, (tr_pos, ho_pos) in enumerate(folds):
        print(f"\n== fold {fold_idx} / {args.folds}  "
              f"({len(tr_pos)} train, {len(ho_pos)} held-out) ==")
        ho_scores, history, val_acc = train_one_fold(
            args=args, resnet=resnet, device=device, fold_idx=fold_idx,
            tr_images=data.train_images[tr_pos], tr_labels=data.train_noisy[tr_pos],
            ho_images=data.train_images[ho_pos], ho_labels=data.train_noisy[ho_pos],
            val_images=data.val_images, val_noisy=data.val_noisy,
            train_tf=train_tf, eval_tf=eval_tf,
        )

        mu, sd = float(ho_scores.mean()), float(ho_scores.std())
        ho_z = (ho_scores - mu) / max(sd, 1e-8)

        all_scores_raw[ho_pos] = ho_scores
        all_scores_z[ho_pos] = ho_z

        history.to_csv(os.path.join(logs, f"t3_fold_{fold_idx}_history.csv"),
                       index=False)

        # Per-fold diagnostics (measurement only).
        ho_mis = mislabeled_mask[ho_pos].astype(int)
        if 0 < ho_mis.sum() < len(ho_mis):
            fold_auroc = float(roc_auc_score(ho_mis, ho_scores))
            fold_ap = float(average_precision_score(ho_mis, ho_scores))
        else:
            fold_auroc = fold_ap = float("nan")
        fold_metrics.append({
            "fold": fold_idx,
            "n_train": int(len(tr_pos)),
            "n_heldout": int(len(ho_pos)),
            "val_noisy_acc": val_acc,
            "heldout_auroc": fold_auroc,
            "heldout_ap": fold_ap,
            "heldout_mislabel_rate": float(ho_mis.mean()),
        })
        print(f"  held-out AUROC={fold_auroc:.4f} AP={fold_ap:.4f} "
              f"val_noisy_acc={val_acc:.4f}")

    if np.isnan(all_scores_z).any():
        missing = int(np.isnan(all_scores_z).sum())
        sys.exit(f"{missing} training images were never held out — fold coverage bug.")

    # ---- Pooled diagnostics (measurement only) ---------------------------
    y_true = mislabeled_mask.astype(int)
    k = int(y_true.sum())
    pooled_auroc = float(roc_auc_score(y_true, all_scores_z))
    pooled_ap = float(average_precision_score(y_true, all_scores_z))
    order = np.argsort(-all_scores_z)
    precision_at_k = float(y_true[order[:k]].mean())
    baseline_ap = float(y_true.mean())

    print("\n== Task 3 pooled diagnostics ==")
    print(f"  n_mislabeled        : {k}")
    print(f"  baseline AP (= rate): {baseline_ap:.4f}")
    print(f"  pooled AUROC        : {pooled_auroc:.4f}")
    print(f"  pooled AP           : {pooled_ap:.4f}")
    print(f"  precision@{k:<7d}  : {precision_at_k:.4f}")

    # ---- Submission file --------------------------------------------------
    out = pd.DataFrame({
        "index": data.train_idx,
        "score": all_scores_z.astype(np.float32),
    })
    assert len(out) == 45000
    assert out["index"].is_unique
    assert set(out["index"]) == set(data.train_idx)
    assert out["score"].notna().all()
    out.to_csv(os.path.join(preds_dir, "t3_noise_scores.csv"), index=False)

    # ---- Detail file for inspection (not submitted) ----------------------
    pd.DataFrame({
        "index": data.train_idx,
        "raw_ce": all_scores_raw,
        "score_z": all_scores_z,
        "is_mislabeled": mislabeled_mask.astype(int),   # measurement only
    }).to_csv(os.path.join(logs, "t3_scores_details.csv"), index=False)

    # ---- Summary JSON -----------------------------------------------------
    summary = {
        "n_folds": args.folds,
        "fold_epochs": args.fold_epochs,
        "scheduler_tmax": args.scheduler_tmax if args.scheduler_tmax else args.epochs,
        "seed_base": args.seed,
        "fold_seed_rule": "seed + fold_idx",
        "signal": "per-sample cross-entropy on held-out fold, z-scored per fold, then pooled",
        "pooled": {
            "auroc": pooled_auroc,
            "ap": pooled_ap,
            "precision_at_k": precision_at_k,
            "k": k,
            "baseline_ap": baseline_ap,
            "n_scored": int(len(out)),
        },
        "per_fold": fold_metrics,
        "budget": {
            "training_runs_used": args.folds,
            "training_runs_limit": 5,
            "epochs_per_fold": args.fold_epochs,
            "epochs_limit": args.epochs,
        },
        "note": (
            "Training and fold stratification used noisy labels only. Clean labels "
            "are used only to compute AUROC / AP / precision@k and the diagnostic "
            "column in logs/t3_scores_details.csv."
        ),
    }
    with open(os.path.join(logs, "t3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print("  predictions/t3_noise_scores.csv   (45,000 rows)")
    print("  logs/t3_scores_details.csv")
    print("  logs/t3_summary.json")
    print("  logs/t3_fold_*_history.csv")


if __name__ == "__main__":
    main()