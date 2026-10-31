"""Schema-only check that every task produced well-formed outputs.

Does NOT look at metric quality. Verifies file existence, column names,
row counts, value ranges, and no-NaN. Run after smoke_test.

Usage:
  python verify_outputs.py --out_dir /tmp/smoke_out
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def check_pred_csv(path, n_rows=10000):
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "pred"], f"{path}: columns {df.columns.tolist()}"
    assert len(df) == n_rows, f"{path}: {len(df)} rows, expected {n_rows}"
    assert (df["id"].values == np.arange(n_rows)).all(), \
        f"{path}: id is not exactly 0..{n_rows - 1}"
    assert df["pred"].between(0, 9).all(), f"{path}: pred outside [0, 9]"
    assert df["pred"].notna().all(), f"{path}: NaN in pred"
    return (f"OK   {os.path.basename(path):32s} "
            f"{n_rows} rows, pred in [{df['pred'].min()},{df['pred'].max()}]")


def check_dynamics(path):
    df = pd.read_csv(path)
    expected = ["epoch", "train_acc_correct", "train_acc_mislabeled",
                "val_acc_noisy", "val_acc_clean"]
    assert list(df.columns) == expected, f"{path}: columns {df.columns.tolist()}"
    assert len(df) >= 1, f"{path}: no rows"
    for c in expected[1:]:
        assert df[c].between(0, 1).all(), f"{path}: {c} outside [0, 1]"
        assert df[c].notna().all(), f"{path}: NaN in {c}"
    return f"OK   {os.path.basename(path):32s} {len(df)} epochs"


def check_noise_scores(path):
    df = pd.read_csv(path)
    assert list(df.columns) == ["index", "score"], f"{path}: columns {df.columns.tolist()}"
    assert len(df) == 45000, f"{path}: {len(df)} rows, expected 45000"
    assert df["index"].is_unique, f"{path}: duplicate indices"
    assert df["score"].notna().all(), f"{path}: NaN in score"
    assert np.isfinite(df["score"].to_numpy()).all(), f"{path}: non-finite score"
    return (f"OK   {os.path.basename(path):32s} 45000 rows, "
            f"score in [{df['score'].min():.2f},{df['score'].max():.2f}]")


def check_summary(path, required_keys):
    with open(path) as f:
        d = json.load(f)
    missing = [k for k in required_keys if k not in d]
    assert not missing, f"{path}: missing keys {missing}"
    return f"OK   {os.path.basename(path):32s} keys present: {required_keys}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    preds = os.path.join(args.out_dir, "predictions")
    logs = os.path.join(args.out_dir, "logs")
    fails = []

    def run(label, fn, *a, **kw):
        try:
            print(fn(*a, **kw))
        except Exception as e:
            fails.append(f"FAIL {label}: {e}")
            print(f"FAIL {label}: {e}")

    # ---- required prediction files ----
    run("t1_clean_ref",       check_pred_csv, os.path.join(preds, "t1_clean_ref.csv"))
    run("t1_noisy_ce",        check_pred_csv, os.path.join(preds, "t1_noisy_ce.csv"))
    run("t2_dynamics",        check_dynamics, os.path.join(preds, "t2_dynamics.csv"))
    run("t2_early_stop",      check_pred_csv, os.path.join(preds, "t2_early_stop.csv"))
    run("t3_noise_scores",    check_noise_scores, os.path.join(preds, "t3_noise_scores.csv"))

    # Task 4: 1..3 t4_*.csv files, each a valid pred file
    if not os.path.isdir(preds):
        fails.append(f"FAIL {preds} does not exist")
    else:
        t4 = sorted(f for f in os.listdir(preds)
                    if f.startswith("t4_") and f.endswith(".csv"))
        if not (1 <= len(t4) <= 3):
            fails.append(f"FAIL Task 4: {len(t4)} t4_*.csv files, expected 1-3: {t4}")
            print(f"FAIL Task 4: {len(t4)} files, expected 1-3: {t4}")
        else:
            print(f"OK   Task 4                              "
                  f"{len(t4)} method file(s): {t4}")
            for f in t4:
                run(f, check_pred_csv, os.path.join(preds, f))

    run("t5_baseline_synth",  check_pred_csv, os.path.join(preds, "t5_baseline_synth.csv"))
    run("t5_best_synth",      check_pred_csv, os.path.join(preds, "t5_best_synth.csv"))

    # ---- summary JSONs consumed downstream ----
    run("t1_summary", check_summary, os.path.join(logs, "t1_summary.json"),
        ["clean_ref", "noisy_ce"])
    run("t2_summary", check_summary, os.path.join(logs, "t2_summary.json"),
        ["early_stop_epoch", "memorization_onset_epoch"])
    run("t3_summary", check_summary, os.path.join(logs, "t3_summary.json"),
        ["pooled", "per_fold"])
    run("t4_summary", check_summary, os.path.join(logs, "t4_summary.json"),
        ["best_method", "per_method"])
    run("t5_summary", check_summary, os.path.join(logs, "t5_summary.json"),
        ["synthetic_noise", "real_noise"])

    print()
    if fails:
        print(f"{len(fails)} failure(s):")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    print("ALL OUTPUT CHECKS PASSED")


if __name__ == "__main__":
    main()