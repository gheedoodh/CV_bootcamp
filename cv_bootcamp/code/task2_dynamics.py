"""Task 2 - Training Dynamics.

This script does NOT train anything. Task 1 already ran the noisy CE baseline
with `track_train=True` and saved per-epoch arrays under logs/. Task 2 reads them
and produces:

  predictions/t2_dynamics.csv      epoch,train_acc_correct,train_acc_mislabeled,
                                   val_acc_noisy,val_acc_clean
  predictions/t2_early_stop.csv    id,pred — test predictions from the epoch
                                   chosen by NOISY validation accuracy only
  logs/t2_dynamics_plot.png        four curves + mem-onset + early-stop markers
  logs/t2_summary.json             onset epoch, evidence, early-stop epoch,
                                   peak-vs-final comparison for the report

Rule compliance:
  * Selection (early-stop epoch) uses `val_acc_noisy` ONLY.
  * Clean labels are used here only to *measure* the curves (train-subset split
    and val_acc_clean), which the statement explicitly allows for Task 2.
  * No clean label touches training, filtering, or the checkpoint we submit.

The memorization-onset epoch is a hardcoded constant (MEM_ONSET_EPOCH), set by
inspecting the trained noisy-CE curve once. The script still recomputes the
numeric evidence that justifies that epoch from the actual logged data, and
exits loudly if the logged curves disagree with the constant by enough that
the constant is clearly stale (e.g. after switching recipe/seed).

Inputs (must exist):
  logs/t1_noisy_ce_history.csv
  logs/t1_noisy_ce_test_preds_all_epochs.npy
  logs/t1_noisy_ce_val_preds_all_epochs.npy
  logs/t1_noisy_ce_train_correct_all_epochs.npy
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from common import add_common_args, load_data, make_dirs, write_predictions


# Epoch at which the noisy-CE baseline starts memorizing wrong labels, taken
# from an inspection of t2_dynamics.csv (the train_acc_mislabeled curve). The
# default 39 comes from the seed-69 60-epoch run with the notebook recipe.
# If you rerun Task 1 with a different seed / recipe / epoch count, re-inspect
# t2_dynamics.csv and update this number (the script will warn if it drifts).
MEM_ONSET_EPOCH = 39


def memorization_evidence(dyn, onset_epoch):
    """Numbers that justify `onset_epoch` in Q2. Does not *find* the epoch."""
    t_mis = dyn["train_acc_mislabeled"].to_numpy()
    t_cor = dyn["train_acc_correct"].to_numpy()
    v_cln = dyn["val_acc_clean"].to_numpy()

    n = len(t_mis)
    early_window = max(5, n // 4)
    floor = float(t_mis[:early_window].min())
    at_onset = float(t_mis[onset_epoch])
    peak = float(t_mis.max())
    slope_at_onset = float(t_mis[onset_epoch] - t_mis[max(onset_epoch - 3, 0)])
    clean_peak_epoch = int(np.argmax(v_cln))
    return {
        "floor": floor,
        "at_onset_mislabeled_train_acc": at_onset,
        "peak": peak,
        "rise_floor_to_peak": round(peak - floor, 4),
        "slope_last_3_epochs": round(slope_at_onset, 4),
        "clean_val_peak_epoch": clean_peak_epoch,
        "clean_val_peak_acc": float(v_cln[clean_peak_epoch]),
        "clean_val_at_onset": float(v_cln[onset_epoch]),
        "clean_val_at_final": float(v_cln[-1]),
        "gap_correct_minus_mislabeled": round(float(t_cor[onset_epoch] - at_onset), 4),
    }


def plot_dynamics(df, mem_onset, early_stop, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(skipping plot: {exc})")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["epoch"], df["train_acc_correct"], label="train acc (correct subset)", lw=2)
    ax.plot(df["epoch"], df["train_acc_mislabeled"], label="train acc (mislabeled subset)", lw=2)
    ax.plot(df["epoch"], df["val_acc_noisy"], label="val acc (noisy labels)", lw=2)
    ax.plot(df["epoch"], df["val_acc_clean"], label="val acc (clean labels)", lw=2, ls="--")
    ax.axhline(0.1, color="gray", ls=":", lw=1, label="chance (10%)")

    if mem_onset >= 0:
        ax.axvline(mem_onset, color="red", ls="-.", lw=1.5,
                   label=f"memorization onset (epoch {mem_onset})")
    if early_stop >= 0:
        ax.axvline(early_stop, color="green", ls=":", lw=1.5,
                   label=f"early stop (epoch {early_stop})")

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Task 2 — noisy CE baseline training dynamics")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Task 2 - Training Dynamics (reuses Task 1 run)")
    add_common_args(parser)
    parser.add_argument("--t1_logs", default=None,
                        help="folder with Task 1 logs; defaults to <out_dir>/logs")
    args = parser.parse_args()

    make_dirs(args.out_dir)
    logs = args.t1_logs or os.path.join(args.out_dir, "logs")
    preds_dir = os.path.join(args.out_dir, "predictions")

    # ---- Load ------------------------------------------------------------
    req = [
        "t1_noisy_ce_history.csv",
        "t1_noisy_ce_test_preds_all_epochs.npy",
        "t1_noisy_ce_val_preds_all_epochs.npy",
        "t1_noisy_ce_train_correct_all_epochs.npy",
    ]
    missing = [f for f in req if not os.path.exists(os.path.join(logs, f))]
    if missing:
        sys.exit(
            "Task 2 needs Task 1's noisy-run artifacts. Missing:\n  "
            + "\n  ".join(missing)
            + "\n\nRerun Task 1 (it must use track_train=True on the noisy model)."
        )

    data = load_data(args.data_dir)
    history = pd.read_csv(os.path.join(logs, "t1_noisy_ce_history.csv"))
    test_preds = np.load(os.path.join(logs, "t1_noisy_ce_test_preds_all_epochs.npy"))
    val_preds = np.load(os.path.join(logs, "t1_noisy_ce_val_preds_all_epochs.npy"))
    train_correct = np.load(os.path.join(logs, "t1_noisy_ce_train_correct_all_epochs.npy"))

    n_epochs = len(history)
    assert test_preds.shape == (n_epochs, len(data.test_images))
    assert val_preds.shape == (n_epochs, len(data.val_images))
    assert train_correct.shape == (n_epochs, len(data.train_images))

    # ---- Reconstruct dynamics (measurement only) -------------------------
    mislabeled = data.train_noisy != data.train_clean       # (45000,)
    n_correct = int((~mislabeled).sum())
    n_mislabeled = int(mislabeled.sum())
    print(f"train split: {n_correct} correctly labeled, {n_mislabeled} mislabeled")

    rows = []
    for e in range(n_epochs):
        row = train_correct[e]
        rows.append({
            "epoch": e,
            "train_acc_correct": float(row[~mislabeled].mean()),
            "train_acc_mislabeled": float(row[mislabeled].mean()),
            "val_acc_noisy": float((val_preds[e] == data.val_noisy).mean()),
            "val_acc_clean": float((val_preds[e] == data.val_clean).mean()),
        })
    dyn = pd.DataFrame(rows)

    # Sanity: reconstructed val_acc_noisy must match Task 1's history.
    assert np.allclose(dyn["val_acc_noisy"].to_numpy(), history["val_acc"].to_numpy(), atol=1e-9), \
        "val_acc_noisy mismatch — Task 1 logs were altered?"

    dyn.to_csv(os.path.join(preds_dir, "t2_dynamics.csv"), index=False)

    # ---- Memorization onset: hardcoded, evidence recomputed --------------
    if MEM_ONSET_EPOCH >= n_epochs:
        sys.exit(
            f"MEM_ONSET_EPOCH={MEM_ONSET_EPOCH} but Task 1 only logged "
            f"{n_epochs} epochs. Update MEM_ONSET_EPOCH in task2_dynamics.py."
        )
    onset_epoch = MEM_ONSET_EPOCH
    onset_evidence = memorization_evidence(dyn, onset_epoch)

    # Guardrail: if the curve doesn't actually support the constant, fail loud.
    if onset_evidence["at_onset_mislabeled_train_acc"] <= onset_evidence["floor"]:
        sys.exit(
            f"MEM_ONSET_EPOCH={onset_epoch} but mislabeled-train acc at that epoch "
            f"({onset_evidence['at_onset_mislabeled_train_acc']:.4f}) is not above "
            f"its floor ({onset_evidence['floor']:.4f}). "
            f"Inspect predictions/t2_dynamics.csv and update MEM_ONSET_EPOCH."
        )

    # ---- Early stopping: NOISY validation accuracy ONLY ------------------
    val_noisy = dyn["val_acc_noisy"].to_numpy()
    early_stop_epoch = int(np.argmax(val_noisy))
    early_stop_val = float(val_noisy[early_stop_epoch])

    # ---- Write t2_early_stop.csv from that epoch's test predictions ------
    write_predictions(test_preds[early_stop_epoch],
                      os.path.join(preds_dir, "t2_early_stop.csv"))

    # ---- Plot ------------------------------------------------------------
    plot_path = os.path.join(logs, "t2_dynamics_plot.png")
    plot_dynamics(dyn, onset_epoch, early_stop_epoch, plot_path)

    # ---- Summary for the report -----------------------------------------
    summary = {
        "n_epochs": n_epochs,
        "memorization_onset_epoch": onset_epoch,
        "memorization_onset_source": "hardcoded constant MEM_ONSET_EPOCH (task2_dynamics.py)",
        "memorization_onset_evidence": onset_evidence,
        "early_stop_epoch": early_stop_epoch,
        "early_stop_val_noisy_acc": early_stop_val,
        "early_stop_val_clean_acc": float(dyn["val_acc_clean"].iloc[early_stop_epoch]),
        "final_epoch_val_noisy_acc": float(val_noisy[-1]),
        "final_epoch_val_clean_acc": float(dyn["val_acc_clean"].iloc[-1]),
        "peak_train_acc_correct": float(dyn["train_acc_correct"].max()),
        "peak_train_acc_mislabeled": float(dyn["train_acc_mislabeled"].max()),
        "floor_train_acc_mislabeled": onset_evidence["floor"],
        "chance_level_10class": 0.1,
        "note": (
            "Selection uses val_acc_noisy only; clean labels appear only in "
            "the measurement columns of t2_dynamics.csv and in this summary."
        ),
    }
    with open(os.path.join(logs, "t2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Console report --------------------------------------------------
    print("\n== Task 2 summary ==")
    print(f"memorization onset epoch : {onset_epoch}  (hardcoded constant)")
    print(f"  mislabeled-train floor  : {onset_evidence['floor']:.4f}")
    print(f"  mislabeled-train @onset : {onset_evidence['at_onset_mislabeled_train_acc']:.4f}")
    print(f"  mislabeled-train peak   : {onset_evidence['peak']:.4f}")
    print(f"  slope (last 3 epochs)   : {onset_evidence['slope_last_3_epochs']}")
    print(f"  clean-val peak          : {onset_evidence['clean_val_peak_acc']:.4f} "
          f"at epoch {onset_evidence['clean_val_peak_epoch']}")
    print(f"  clean-val @onset        : {onset_evidence['clean_val_at_onset']:.4f}")
    print(f"  clean-val @final        : {onset_evidence['clean_val_at_final']:.4f}")
    print(f"early-stop epoch (by noisy val): {early_stop_epoch}")
    print(f"  val noisy @early stop   : {early_stop_val:.4f}")
    print(f"  val noisy @final        : {val_noisy[-1]:.4f}")
    print(f"  val clean @early stop   : {summary['early_stop_val_clean_acc']:.4f}")
    print(f"  val clean @final        : {summary['final_epoch_val_clean_acc']:.4f}")
    print("\nWrote:")
    print("  predictions/t2_dynamics.csv")
    print("  predictions/t2_early_stop.csv")
    print(f"  {plot_path}")
    print("  logs/t2_summary.json")


if __name__ == "__main__":
    main()