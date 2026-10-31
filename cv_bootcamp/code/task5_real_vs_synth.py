import argparse
import json
import os
import sys

import numpy as np

from common import (
    add_common_args, env_info, get_device, load_data, load_resnet18,
    make_dirs, make_transforms, run_training, write_predictions,
)
# Reuse Task 4's loss factories so hyperparameters can't silently drift.
# task4_robust has no import side effects (its main() is under a __main__ guard).
from task4_robust import METHODS as T4_METHODS

LOSS_FACTORIES = dict(T4_METHODS)   # {"gce": factory, "ls": factory, "elr": factory}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_method_config(args, method, cfg):
    """Restore the winning method's hyperparameters from Task 4's summary."""
    if method == "gce":
        args.gce_q = float(cfg.get("q", args.gce_q))
        args.warmup_epochs = int(cfg.get("warmup_epochs", args.warmup_epochs))
    elif method == "ls":
        args.ls_smoothing = float(cfg.get("smoothing", args.ls_smoothing))
    elif method == "elr":
        args.elr_lam = float(cfg.get("lam", args.elr_lam))
        args.elr_beta = float(cfg.get("beta", args.elr_beta))
    else:
        sys.exit(f"Unknown method '{method}'; expected one of {list(LOSS_FACTORIES)}")


def check_recipe_matches(args, t4_summary):
    """Task 5 must retrain with the same recipe as Tasks 1/4 for a fair
    comparison. Warn (don't crash) on drift so the user can decide."""
    r = t4_summary.get("recipe", {})
    expected = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "epochs": args.epochs,
    }
    mismatches = []
    for k, v in expected.items():
        if k in r and r[k] != v:
            mismatches.append(f"{k}: t4={r[k]} vs t5={v}")
    if mismatches:
        print("!! recipe drift vs Task 4:")
        for m in mismatches:
            print("   " + m)
        print("   Task 5 comparison may not be apples-to-apples.")


def load_task4_summary(logs):
    path = os.path.join(logs, "t4_summary.json")
    if not os.path.exists(path):
        sys.exit(
            "Task 5 needs logs/t4_summary.json to know which Task 4 method won. "
            "Run task4_robust.py first."
        )
    with open(path) as f:
        return json.load(f)


def clean_val_acc_at(val_preds_all_epochs, epoch, val_clean):
    """Post-hoc measurement only — clean labels are not used for selection."""
    return float((val_preds_all_epochs[epoch] == np.asarray(val_clean)).mean())


def real_noise_reference(logs, data):
    """Pull real-noise numbers for the Q5 comparison. Measurement only."""
    t1_path = os.path.join(logs, "t1_summary.json")
    t4_path = os.path.join(logs, "t4_summary.json")
    if not (os.path.exists(t1_path) and os.path.exists(t4_path)):
        return {"note": "Task 1 or Task 4 summary missing; real-noise reference unavailable"}

    with open(t1_path) as f:
        t1 = json.load(f)
    with open(t4_path) as f:
        t4 = json.load(f)

    ce = t1["noisy_ce"]
    ce_clean_at_best = None
    vp = os.path.join(logs, "t1_noisy_ce_val_preds_all_epochs.npy")
    if os.path.exists(vp):
        ce_clean_at_best = clean_val_acc_at(
            np.load(vp), ce["best_epoch"], data.val_clean)

    best_name = t4["best_method"]
    pm = t4["per_method"][best_name]
    bm_clean_at_best = None
    vp = os.path.join(logs, f"t4_{best_name}_val_preds_all_epochs.npy")
    if os.path.exists(vp):
        bm_clean_at_best = clean_val_acc_at(
            np.load(vp), pm["best_epoch"], data.val_clean)

    return {
        "ce_baseline": {
            "best_epoch": int(ce["best_epoch"]),
            "best_val_noisy": float(ce["best_val_noisy_acc"]),
            "final_val_noisy": float(ce["final_val_noisy_acc"]),
            "clean_val_at_best": ce_clean_at_best,
            "final_clean_val": float(ce["final_val_clean_acc"]),
        },
        "best_method": {
            "method": best_name,
            "best_epoch": int(pm["best_epoch"]),
            "best_val_noisy": float(pm["best_val_noisy_acc"]),
            "final_val_noisy": float(pm["final_val_noisy_acc"]),
            "clean_val_at_best": bm_clean_at_best,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task 5 - Real vs Synthetic Noise")
    add_common_args(parser)

    # Method hyperparameters, same names/defaults as task4_robust.py so
    # apply_method_config can override them from the summary.
    parser.add_argument("--gce_q", type=float, default=0.7)
    parser.add_argument("--ls_smoothing", type=float, default=0.27)
    parser.add_argument("--elr_lam", type=float, default=3.0)
    parser.add_argument("--elr_beta", type=float, default=0.7)
    parser.set_defaults(warmup_epochs=13)

    parser.add_argument("--force_best_method", default=None,
                        choices=sorted(LOSS_FACTORIES.keys()),
                        help="override the Task 4 winner (for debugging only)")
    args = parser.parse_args()

    make_dirs(args.out_dir)
    logs = os.path.join(args.out_dir, "logs")
    preds_dir = os.path.join(args.out_dir, "predictions")

    data = load_data(args.data_dir)
    train_tf, eval_tf = make_transforms(data.mean, data.std)
    resnet = load_resnet18(args.data_dir)
    device = get_device()

    # ---- 1. Determine which method is "best" -----------------------------
    t4 = load_task4_summary(logs)
    best_method = args.force_best_method or t4["best_method"]
    if best_method not in LOSS_FACTORIES:
        sys.exit(f"Task 4 summary says best_method='{best_method}', "
                 f"but valid methods are {list(LOSS_FACTORIES)}.")
    best_cfg = t4.get("best_method_config", {}) if not args.force_best_method else {}
    apply_method_config(args, best_method, best_cfg)

    print(f"Task 4 winner: {best_method}  "
          f"(best noisy val = {t4.get('best_method_val_noisy_acc'):.4f})")
    print(f"Method config restored from t4_summary: {best_cfg}")
    check_recipe_matches(args, t4)

    # Sanity: synthetic noise rate should match the human noise rate.
    synth_rate_train = float((data.train_synth != data.train_clean).mean())
    synth_rate_val = float((data.val_synth != data.val_clean).mean())
    human_rate_train = float((data.train_noisy != data.train_clean).mean())
    print(f"noise rate  train: human={human_rate_train:.4f}  "
          f"synthetic={synth_rate_train:.4f}")
    print(f"noise rate  val:   synthetic={synth_rate_val:.4f}")

    # ---- 2. Train CE baseline on synthetic labels ------------------------
    print(f"\n== t5_baseline_synth: CE on SYNTHETIC labels ==")
    ce_res = run_training(
        args=args, tag="t5_baseline_synth",
        resnet_cls=resnet, device=device,
        train_images=data.train_images, train_labels=data.train_synth,
        val_images=data.val_images, val_labels=data.val_synth,   # <-- Task 5 rule
        test_images=data.test_images,
        train_tf=train_tf, eval_tf=eval_tf,
        loss_factory=lambda n: __import__("torch").nn.CrossEntropyLoss(),
        select_from_epoch=0,
        track_train=False,
    )
    write_predictions(ce_res.test_preds[ce_res.best_epoch],
                      os.path.join(preds_dir, "t5_baseline_synth.csv"))
    

    # ---- 3. Train best Task 4 method on synthetic labels -----------------
    print(f"\n== t5_best_synth: {best_method} on SYNTHETIC labels ==")
    bm_factory = LOSS_FACTORIES[best_method]
    bm_res = run_training(
        args=args, tag="t5_best_synth",
        resnet_cls=resnet, device=device,
        train_images=data.train_images, train_labels=data.train_synth,
        val_images=data.val_images, val_labels=data.val_synth,   # <-- Task 5 rule
        test_images=data.test_images,
        train_tf=train_tf, eval_tf=eval_tf,
        loss_factory=lambda n: bm_factory(n, args),
        select_from_epoch=0,
        track_train=False,
    )
    write_predictions(bm_res.test_preds[bm_res.best_epoch],
                      os.path.join(preds_dir, "t5_best_synth.csv"))
    

    # ---- 4. Post-hoc measurement: clean-val accuracies -------------------
    ce_clean_best = clean_val_acc_at(ce_res.val_preds, ce_res.best_epoch, data.val_clean)
    bm_clean_best = clean_val_acc_at(bm_res.val_preds, bm_res.best_epoch, data.val_clean)
    ce_clean_final = clean_val_acc_at(ce_res.val_preds, -1, data.val_clean)
    bm_clean_final = clean_val_acc_at(bm_res.val_preds, -1, data.val_clean)

    synthetic_reference = {
        "ce_baseline": {
            "best_epoch": int(ce_res.best_epoch),
            "best_val_synth": float(ce_res.best_val),
            "final_val_synth": float(ce_res.final_val),
            "clean_val_at_best": ce_clean_best,
            "final_clean_val": ce_clean_final,
        },
        "best_method": {
            "method": best_method,
            "best_epoch": int(bm_res.best_epoch),
            "best_val_synth": float(bm_res.best_val),
            "final_val_synth": float(bm_res.final_val),
            "clean_val_at_best": bm_clean_best,
            "final_clean_val": bm_clean_final,
        },
    }

    # ---- 5. Real-noise reference + comparison ----------------------------
    real_ref = real_noise_reference(logs, data)

    comparison = {}
    if "ce_baseline" in real_ref and "best_method" in real_ref:
        ce_real = real_ref["ce_baseline"]
        bm_real = real_ref["best_method"]
        ce_syn = synthetic_reference["ce_baseline"]
        bm_syn = synthetic_reference["best_method"]

        comparison = {
            "ce_baseline": {
                "best_val_real": ce_real["best_val_noisy"],
                "best_val_synth": ce_syn["best_val_synth"],
                "delta_best_val_real_minus_synth": round(
                    ce_real["best_val_noisy"] - ce_syn["best_val_synth"], 4),
                "clean_val_at_best_real": ce_real["clean_val_at_best"],
                "clean_val_at_best_synth": ce_syn["clean_val_at_best"],
                "delta_clean_at_best_real_minus_synth": (
                    None if ce_real["clean_val_at_best"] is None else
                    round(ce_real["clean_val_at_best"] - ce_syn["clean_val_at_best"], 4)),
            },
            best_method: {
                "best_val_real": bm_real["best_val_noisy"],
                "best_val_synth": bm_syn["best_val_synth"],
                "delta_best_val_real_minus_synth": round(
                    bm_real["best_val_noisy"] - bm_syn["best_val_synth"], 4),
                "clean_val_at_best_real": bm_real["clean_val_at_best"],
                "clean_val_at_best_synth": bm_syn["clean_val_at_best"],
                "delta_clean_at_best_real_minus_synth": (
                    None if bm_real["clean_val_at_best"] is None else
                    round(bm_real["clean_val_at_best"] - bm_syn["clean_val_at_best"], 4)),
            },
            "method_gain_over_ce": {
                "real": round(bm_real["best_val_noisy"] - ce_real["best_val_noisy"], 4),
                "synthetic": round(
                    bm_syn["best_val_synth"] - ce_syn["best_val_synth"], 4),
            },
        }

    # ---- 6. Write summary ------------------------------------------------
    summary = {
        "task4_best_method": best_method,
        "task4_best_method_config": best_cfg,
        "task4_best_method_val_noisy_acc": float(
            t4.get("best_method_val_noisy_acc", float("nan"))),
        "noise_rates": {
            "human_train": human_rate_train,
            "synthetic_train": synth_rate_train,
            "synthetic_val": synth_rate_val,
        },
        "selection_signal": "synthetic validation accuracy only",
        "recipe": {
            "optimizer": "SGD",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": f"CosineAnnealingLR(T_max={args.scheduler_tmax or args.epochs})",
            "batch_size": args.batch_size,
            "augmentation": "RandomHorizontalFlip + RandomCrop(32, padding=4)",
            "seed": args.seed,
            "epochs": args.epochs,
        },
        "real_noise": real_ref,
        "synthetic_noise": synthetic_reference,
        "comparison": comparison,
        "env": env_info(),
        "note": (
            "Selection uses val_synth only. Clean val accuracies are computed "
            "post-hoc from the saved per-epoch val predictions and are used only "
            "for the Q5 comparison, never for training or selection."
        ),
    }
    with open(os.path.join(logs, "t5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- 7. Console report ----------------------------------------------
    print("\n== Task 5 real-vs-synthetic comparison ==")
    print(f"  best method (from Task 4)   : {best_method}")
    print(f"  human noise rate (train)    : {human_rate_train:.4f}")
    print(f"  synthetic noise rate (train): {synth_rate_train:.4f}")
    print(f"  synthetic noise rate (val)  : {synth_rate_val:.4f}")

    if comparison:
        ce = comparison["ce_baseline"]
        bm = comparison[best_method]
        print("\n  CE baseline     real_best_val={:.4f}  synth_best_val={:.4f}  "
              "delta={:.4f}".format(ce["best_val_real"], ce["best_val_synth"],
                                    ce["delta_best_val_real_minus_synth"]))
        print("  {}              real_best_val={:.4f}  synth_best_val={:.4f}  "
              "delta={:.4f}".format(best_method, bm["best_val_real"],
                                    bm["best_val_synth"],
                                    bm["delta_best_val_real_minus_synth"]))
        print("\n  method gain over CE   real: {:.4f}   synthetic: {:.4f}".format(
            comparison["method_gain_over_ce"]["real"],
            comparison["method_gain_over_ce"]["synthetic"]))
        if ce.get("clean_val_at_best_real") is not None:
            print("\n  clean val at best (real)    CE={:.4f}  {}={:.4f}".format(
                ce["clean_val_at_best_real"], best_method,
                bm["clean_val_at_best_real"]))
        print("  clean val at best (synthetic)  CE={:.4f}  {}={:.4f}".format(
            ce["clean_val_at_best_synth"], best_method,
            bm["clean_val_at_best_synth"]))

    print("\nWrote:")
    print("  predictions/t5_baseline_synth.csv   (best synthetic-val epoch)")
    print("  predictions/t5_best_synth.csv       (best synthetic-val epoch)")
    print("  predictions/t5_*_final.csv          (final-epoch fallbacks)")
    print("  logs/t5_summary.json")


if __name__ == "__main__":
    main()