"""Assemble <rollno>_cv_bootcamp/ from a completed --out_dir and zip it.

Run this AFTER tasks 1-5 and verify_outputs.py have all succeeded.

Expected project layout:
    <project_root>/
    ├── model.py
    ├── code/
    │   ├── build_submission.py   <- this file
    │   ├── common.py
    │   ├── task*.py
    │   ├── verify_outputs.py
    │   └── README.md
    └── data/                     <- provided inputs + results_template.json

Usage:
    python code/build_submission.py --rollno 22CS10001
    # or override defaults:
    python code/build_submission.py --data_dir DATA --out_dir OUT --rollno 22CS10001

Writes:
    <project_root>/<rollno>_cv_bootcamp/          (folder)
    <project_root>/<rollno>_cv_bootcamp.zip       (zip)
"""
import argparse
import json
import os
import shutil
import sys
import zipfile


CODE_FILES = [
    "common.py",
    "task1_baselines.py",
    "task2_dynamics.py",
    "task3_noise_scores.py",
    "task4_robust.py",
    "task5_real_vs_synth.py",
    "verify_outputs.py",
    "README.md",
]

# Required prediction files. Task 4's `t4_*.csv` files are discovered dynamically
# and appended so the whitelist stays accurate for any subset of methods.
REQUIRED_PRED_FILES = [
    "t1_clean_ref.csv",
    "t1_noisy_ce.csv",
    "t2_dynamics.csv",
    "t2_early_stop.csv",
    "t3_noise_scores.csv",
    "t5_baseline_synth.csv",
    "t5_best_synth.csv",
]


def find_file(name, search_dirs):
    """Return the first existing path for `name` in `search_dirs`."""
    for d in search_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    # Defaults computed from project_root so the script works from any CWD.
    default_data = os.path.join(project_root, "data")
    default_out = os.path.join(project_root, "out")

    ap = argparse.ArgumentParser(description="Assemble and zip the submission.")
    ap.add_argument("--data_dir", default=default_data,
                    help=f"folder with results_template.json and data (default: {default_data})")
    ap.add_argument("--out_dir", default=default_out,
                    help=f"folder written by task*.py (default: {default_out})")
    ap.add_argument("--rollno", required=True,
                    help="used as the zip / folder name: <rollno>_cv_bootcamp")
    ap.add_argument("--report_md", default=None,
                    help="path to report.md (default: <project_root>/report.md)")
    args = ap.parse_args()

    if args.report_md is None:
        args.report_md = os.path.join(project_root, "report.md")

    src_preds = os.path.join(args.out_dir, "predictions")
    src_logs = os.path.join(args.out_dir, "logs")

    if not os.path.isdir(src_preds):
        sys.exit(f"predictions/ not found under --out_dir: {src_preds}")
    if not os.path.isdir(src_logs):
        sys.exit(f"logs/ not found under --out_dir: {src_logs}")

    # ---- Prepare destination ----------------------------------------------
    root = os.path.join(project_root, f"{args.rollno}_cv_bootcamp")
    zip_path = os.path.join(project_root, f"{args.rollno}_cv_bootcamp.zip")

    if os.path.exists(root):
        shutil.rmtree(root)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    os.makedirs(os.path.join(root, "code"), exist_ok=True)
    os.makedirs(os.path.join(root, "predictions"), exist_ok=True)

    # ---- Copy code ---------------------------------------------------------
    print("== Code ==")
    search_code = [here, project_root, os.getcwd()]
    for fname in CODE_FILES:
        src = find_file(fname, search_code)
        if src is None:
            sys.exit(f"Missing code file: {fname} (searched {search_code})")
        shutil.copy(src, os.path.join(root, "code", fname))
        print(f"  {fname:<28s} <- {src}")

    # ---- Copy predictions --------------------------------------------------
    print("\n== Predictions ==")
    t4_files = sorted(f for f in os.listdir(src_preds)
                      if f.startswith("t4_") and f.endswith(".csv"))
    if not (1 <= len(t4_files) <= 3):
        sys.exit(f"Expected 1-3 t4_*.csv files under {src_preds}, found {len(t4_files)}: {t4_files}")

    pred_files = list(REQUIRED_PRED_FILES) + t4_files
    for fname in pred_files:
        src = os.path.join(src_preds, fname)
        if not os.path.exists(src):
            sys.exit(f"Missing prediction file: {src}")
        shutil.copy(src, os.path.join(root, "predictions", fname))
        print(f"  {fname:<28s} <- {src}")

    # ---- Merge summaries into results.json --------------------------------
    print("\n== results.json ==")
    template_path = os.path.join(args.data_dir, "results_template.json")
    if not os.path.exists(template_path):
        sys.exit(f"results_template.json not found in --data_dir: {template_path}")

    with open(template_path) as f:
        results = json.load(f)

    def load_summary(name):
        p = os.path.join(src_logs, name)
        if not os.path.exists(p):
            sys.exit(f"Missing summary: {p}")
        with open(p) as fh:
            return json.load(fh)

    t1 = load_summary("t1_summary.json")
    t2 = load_summary("t2_summary.json")
    t3 = load_summary("t3_summary.json")
    t4 = load_summary("t4_summary.json")
    t5 = load_summary("t5_summary.json")

    # CAUTION: the key names below are best-guess guesses at what the
    # provided results_template.json expects. The `fill()` function only
    # overwrites keys that already exist in `results`, so this cannot inject
    # unknown fields — but keys with different names in the template will
    # stay at their default values. Update the mapping to match the actual
    # template after inspecting it once.
    merged = {
        "seed": 69,
        "task1": {
            "noise_rate_train": t1.get("noise_rate", t1.get("train_noise_rate")),
            "noise_rate_val": t1.get("val_noise_rate"),
            "n_mislabeled_train": t1.get("n_mislabeled_train"),
            "clean_ref": t1.get("clean_ref"),
            "noisy_ce": t1.get("noisy_ce"),
        },
        "task2": {
            "memorization_onset_epoch": t2.get("memorization_onset_epoch"),
            "early_stop_epoch": t2.get("early_stop_epoch"),
            "early_stop_val_noisy_acc": t2.get("early_stop_val_noisy_acc"),
            "final_epoch_val_noisy_acc": t2.get("final_epoch_val_noisy_acc"),
        },
        "task3": {
            "auroc": t3.get("pooled", {}).get("auroc"),
            "ap": t3.get("pooled", {}).get("ap"),
            "precision_at_k": t3.get("pooled", {}).get("precision_at_k"),
            "k": t3.get("pooled", {}).get("k"),
        },
        "task4": {
            "methods_trained": t4.get("methods_trained"),
            "best_method": t4.get("best_method"),
            "best_val_noisy_acc": t4.get("best_method_val_noisy_acc"),
            "per_method": t4.get("per_method"),
        },
        "task5": {
            "best_method": t5.get("task4_best_method"),
            "noise_rates": t5.get("noise_rates"),
            "comparison": t5.get("comparison"),
        },
    }

    def fill(dst, src):
        """Recursively overwrite dst[k] with src[k] only when k already exists
        in dst (and isn't None). Preserves the template schema exactly."""
        for k, v in src.items():
            if k not in dst:
                continue
            if isinstance(dst[k], dict) and isinstance(v, dict):
                fill(dst[k], v)
            elif v is not None:
                dst[k] = v

    fill(results, merged)

    results_path = os.path.join(root, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  {results_path}")
    print(f"  (template: {template_path})")

    # ---- Copy report.md ----------------------------------------------------
    print("\n== report.md ==")
    if not os.path.exists(args.report_md):
        sys.exit(f"Missing report.md: {args.report_md}")
    shutil.copy(args.report_md, os.path.join(root, "report.md"))
    print(f"  report.md                <- {args.report_md}")

    # ---- Zip ---------------------------------------------------------------
    print("\n== Zip ==")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for foldername, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                fp = os.path.join(foldername, fn)
                arc = os.path.relpath(fp, project_root)
                z.write(fp, arc)
                print(f"  {arc}")

    print(f"\nWrote:")
    print(f"  {root}")
    print(f"  {zip_path}")

    # ---- Run the provided checker, if present -----------------------------
    checker = find_file("check_format.py", [args.data_dir, project_root, here, os.getcwd()])
    if checker:
        print(f"\nNext step: verify with the provided checker:")
        print(f'  python "{checker}" "{root}"')
        print(f"  python \"{checker}\" \"{zip_path}\"")
    else:
        print("\nNote: check_format.py not found in data/, project_root, code/, or CWD.")
        print("If the bootcamp provides one, run it against the folder before submitting:")
        print(f'  python check_format.py "{root}"')


if __name__ == "__main__":
    main()