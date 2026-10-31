import argparse
import json
import os

import numpy as np
import pandas as pd
import torch.nn as nn

from common import (
    add_common_args, env_info, get_device, load_data, load_resnet18,
    make_dirs, make_transforms, run_training, write_predictions,
)


# ---------------------------------------------------------------------------
# Data statistics
# ---------------------------------------------------------------------------

def confusion_10x10(df, clean_col="clean_label", noisy_col="noisy_label"):
    mat = np.zeros((10, 10), dtype=np.int64)
    for c, n in zip(df[clean_col].to_numpy(), df[noisy_col].to_numpy()):
        mat[int(c), int(n)] += 1
    return mat


def per_class_error(mat):
    row_sums = mat.sum(axis=1)
    return (1.0 - np.diag(mat) / np.maximum(row_sums, 1)).tolist()


def top_confusions(mat, k=12):
    off = mat.copy()
    np.fill_diagonal(off, 0)
    flat = off.ravel()
    order = np.argsort(-flat)[:k]
    return [(int(i // 10), int(i % 10), int(flat[i])) for i in order if flat[i] > 0]


def load_class_names(data_dir):
    path = os.path.join(data_dir, "classes.txt")
    if not os.path.exists(path):
        return [str(i) for i in range(10)]
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    return names if len(names) == 10 else [str(i) for i in range(10)]


def plot_confusion(mat, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
                    xticklabels=classes, yticklabels=classes)
        ax.set_xlabel("noisy label")
        ax.set_ylabel("clean (true) label")
        ax.set_title("Clean vs noisy label confusion (training set)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # matplotlib/seaborn optional
        print(f"(skipping confusion heatmap: {exc})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 1 - Data & Baselines")
    add_common_args(parser)
    args = parser.parse_args()

    make_dirs(args.out_dir)
    logs = os.path.join(args.out_dir, "logs")
    preds_dir = os.path.join(args.out_dir, "predictions")

    data = load_data(args.data_dir)
    train_tf, eval_tf = make_transforms(data.mean, data.std)
    resnet = load_resnet18(args.data_dir)
    classes = load_class_names(args.data_dir)
    device = get_device()

    # -- 1. Noise rate + confusion matrix -----------------------------------
    train_mat = confusion_10x10(data.train_df)
    val_mat = confusion_10x10(data.val_df)
    train_rate = float((data.train_noisy != data.train_clean).mean())
    val_rate = float((data.val_noisy != data.val_clean).mean())
    n_mislabeled = int((data.train_noisy != data.train_clean).sum())

    pd.DataFrame(train_mat, index=classes, columns=classes).to_csv(
        os.path.join(logs, "t1_confusion_train.csv"))
    pd.DataFrame(val_mat, index=classes, columns=classes).to_csv(
        os.path.join(logs, "t1_confusion_val.csv"))
    plot_confusion(train_mat, classes, os.path.join(logs, "t1_confusion_train.png"))

    summary = {
        "train_noise_rate": train_rate,
        "val_noise_rate": val_rate,
        "n_train": int(len(data.train_idx)),
        "n_val": int(len(data.val_idx)),
        "n_mislabeled_train": n_mislabeled,     # = true k for Task 3 precision@k
        "per_class_error_train": per_class_error(train_mat),
        "per_class_error_val": per_class_error(val_mat),
        "top_confusions_train": top_confusions(train_mat, k=12),
        "classes": classes,
        "env": env_info(),
        "seed": args.seed,
        "epochs": args.epochs,
    }
    with open(os.path.join(logs, "t1_noise_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"train noise rate = {train_rate:.4f}   (n_mislabeled = {n_mislabeled})")
    print(f"val   noise rate = {val_rate:.4f}")
    print("top confusions (clean -> noisy, count):")
    for c, n, cnt in summary["top_confusions_train"][:8]:
        print(f"  {classes[c]:>10s} -> {classes[n]:<10s} {cnt}")

    # -- 2. Clean-reference model (allowed: this IS the Task 1 reference) ----
    # Model selection still on noisy val only, per rules.
    print("\n== t1_clean_ref: CE on CLEAN labels ==")
    clean_res = run_training(
        args=args, tag="t1_clean_ref",
        resnet_cls=resnet, device=device,
        train_images=data.train_images, train_labels=data.train_clean,
        val_images=data.val_images, val_labels=data.val_noisy,
        test_images=data.test_images,
        train_tf=train_tf, eval_tf=eval_tf,
        loss_factory=lambda n: nn.CrossEntropyLoss(),
        select_from_epoch=0,
        track_train=False,
    )

    # -- 3. Noisy baseline --------------------------------------------------
    # track_train=True so Task 2 can reconstruct the per-epoch trajectory
    # from these saved arrays WITHOUT retraining (same run, same budget).
    print("\n== t1_noisy_ce: CE on NOISY labels ==")
    noisy_res = run_training(
        args=args, tag="t1_noisy_ce",
        resnet_cls=resnet, device=device,
        train_images=data.train_images, train_labels=data.train_noisy,
        val_images=data.val_images, val_labels=data.val_noisy,
        test_images=data.test_images,
        train_tf=train_tf, eval_tf=eval_tf,
        loss_factory=lambda n: nn.CrossEntropyLoss(),
        select_from_epoch=0,
        track_train=True,
    )

    # -- 4. Task 1 requires FINAL-epoch test predictions --------------------
    write_predictions(clean_res.test_preds[-1],
                      os.path.join(preds_dir, "t1_clean_ref.csv"))
    write_predictions(noisy_res.test_preds[-1],
                      os.path.join(preds_dir, "t1_noisy_ce.csv"))

    # -- 5. Handy summary for the report -----------------------------------
    final_summary = {
        "clean_ref": {
            "final_val_noisy_acc": float(clean_res.final_val),
            "final_val_clean_acc": float(
                (clean_res.val_preds[-1] == data.val_clean).mean()),
            "best_val_noisy_acc": float(clean_res.best_val),
            "best_epoch": int(clean_res.best_epoch),
            "test_file": "predictions/t1_clean_ref.csv",
        },
        "noisy_ce": {
            "final_val_noisy_acc": float(noisy_res.final_val),
            "final_val_clean_acc": float(
                (noisy_res.val_preds[-1] == data.val_clean).mean()),
            "best_val_noisy_acc": float(noisy_res.best_val),
            "best_epoch": int(noisy_res.best_epoch),
            "test_file": "predictions/t1_noisy_ce.csv",
        },
    }
    with open(os.path.join(logs, "t1_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\nWrote:")
    print("  predictions/t1_clean_ref.csv   (final epoch)")
    print("  predictions/t1_noisy_ce.csv    (final epoch)")
    print("Task 2 will read logs/t1_noisy_ce_*_all_epochs.npy — no retraining.")


if __name__ == "__main__":
    main()