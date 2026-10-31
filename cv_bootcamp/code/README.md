# CIFAR-10N Bootcamp — Submission

Learning from messy human labels on CIFAR-10N. All training uses the ResNet-18
provided in `model.py` (imported as-is, not modified), trained from scratch.
No pretrained weights, no external data.

## 1. Environment

**Reported results were produced on a Kaggle Notebook with a GPU accelerator.**
The pipeline is not Kaggle-specific — any machine with Python ≥ 3.10 and
`torch ≥ 2.0` runs it identically at the algorithm level. Minor numerical
variation across GPU models and torch builds is expected; the recipe is fixed
(SGD + cosine + ResNet-18 from scratch + seed 69), not the exact binaries.

| Component | Reported run (Kaggle) | Notes |
|---|---|---|
| OS | Linux (Kaggle) | Development and smoke tests were also validated on Windows 11 |
| Python | 3.13.15 | |
| torch | 2.11.0+cu128 | |
| torchvision | 0.26.0+cu128 | |
| CUDA | cuda version 12.8 | |
| GPU | Tesla T4 | |
| Seed | 69 | Fixed across all tasks |

To reproduce on another machine, install: `numpy pandas scikit-learn
matplotlib seaborn tqdm` and a CUDA-compatible `torch torchvision`


## 2. Project layout

```
<project_root>/
├── model.py            # provided ResNet-18 definition (must be used)
├── code/               # common.py, task1_*.py ... task5_*.py,
│                       # verify_outputs.py, build_submission.py, README.md
└── data/               # provided data files (see below)
```

The `data/` folder must contain:

```
labels.csv             # columns: index, noisy_label, synthetic_label, clean_label
train_images.npy       # (50000, 32, 32, 3) uint8
test_images.npy        # (10000, 32, 32, 3) uint8, no labels
train_idx.txt          # 45,000 indices for training
val_idx.txt            # 5,000 indices for validation
classes.txt            # 10 class names, one per line
results_template.json  # copy template for results.json
```

`model.py` sits at the project root, one level above `data/`. The scripts
locate it automatically; no path argument for it is needed.

Defaults in `common.py` are computed from the script's own location, so the
commands below work from any CWD in this layout. Output defaults to
`<project_root>/out/`.

## 3. Reproduce

From `<project_root>`, with the layout above in place. All runs use `--seed 69`
(default in `common.py`). Total wall-clock is ~40–60 min on a modern GPU;
~3–5 h on CPU only.

```bash
# 1. Data stats + baselines (60 epochs × 2 models). Writes per-epoch logs for Task 2.
python code/task1_baselines.py

# 2. Training dynamics + early-stop. Reads Task 1 logs — does NOT retrain.
python code/task2_dynamics.py

# 3. Per-image noise score via 5-fold cross-validation (5 training runs).
python code/task3_noise_scores.py

# 4. Three robust-training methods under the identical Task 1 recipe.
python code/task4_robust.py

# 5. Retrain CE baseline and best Task 4 method on synthetic labels.
python code/task5_real_vs_synth.py

# 6. Structural verifier (schema only, no metric thresholds).
python code/verify_outputs.py

# 7. Assemble submission folder and zip.
python code/build_submission.py --rollno <your-rollno>
```

No `--data_dir` or `--out_dir` needed — they resolve to `<project_root>/data`
and `<project_root>/out`. Pass them explicitly to override.

Step 6 must print `ALL OUTPUT CHECKS PASSED`. If it prints `FAIL ...`, the
named file is malformed and the submission would score 0 on that task.

Step 7 writes `<project_root>/<your-rollno>_cv_bootcamp/` and
`<project_root>/<your-rollno>_cv_bootcamp.zip`, containing `code/`,
`predictions/`, `results.json`, and `report.md`.

## 4. Outputs

Each task writes predictions into `<out_dir>/predictions/` and diagnostics
into `<out_dir>/logs/`. `build_submission.py` copies only the prediction
files listed below into the submission folder.

| File | Written by | Content |
|---|---|---|
| `t1_clean_ref.csv` | Task 1 | CE trained on clean labels, final-epoch test predictions |
| `t1_noisy_ce.csv` | Task 1 | CE trained on noisy labels, final-epoch test predictions |
| `t2_dynamics.csv` | Task 2 | Per-epoch: `train_acc_correct, train_acc_mislabeled, val_acc_noisy, val_acc_clean` |
| `t2_early_stop.csv` | Task 2 | Test predictions at the epoch with best noisy val acc |
| `t3_noise_scores.csv` | Task 3 | 45,000 rows: `index, score` (higher = more likely mislabeled) |
| `t4_gce.csv` | Task 4 | GCE (q=0.7, 13 warmup epochs) at best noisy-val epoch |
| `t4_ls.csv` | Task 4 | Label smoothing (0.27) at best noisy-val epoch |
| `t4_elr.csv` | Task 4 | ELR (λ=3.0, β=0.7) at best noisy-val epoch |
| `t5_baseline_synth.csv` | Task 5 | CE retrained on `synthetic_label` |
| `t5_best_synth.csv` | Task 5 | Best Task 4 method retrained on `synthetic_label` |

Every prediction CSV has exactly 10,000 rows with columns `id,pred`, where
`id` runs `0..9999` and `pred ∈ {0..9}`.

Diagnostic JSONs (`logs/t1_summary.json`, `logs/t3_summary.json`, etc.) feed
`results.json` via `build_submission.py` and are not part of the prediction
deliverable.

## 5. Budget & rule compliance

**Training budget used:**

| Task | Networks trained | Epochs each | Total epochs |
|---|---|---|---|
| 1 (baselines) | 2 (clean + noisy) | 60 | 120 |
| 2 (dynamics) | 0 (reuses Task 1) | — | 0 |
| 3 (noise scores) | 5 (one per CV fold) | 35 | 175 |
| 4 (robust) | 3 (one per method) | 60 | 180 |
| 5 (real vs synth) | 2 (CE + best method) | 60 | 120 |
| **Total** | **12 networks** | — | — |

All within the stated limits: ≤60 epochs per network, ≤2 networks per method,
≤5 training runs in Task 3.

**Model selection signals:**

- Tasks 1–4: **noisy validation labels only**
- Task 5: **synthetic validation labels only**

**Clean-label policy:** `clean_label` is used only for (a) training the
Task 1 reference model, and (b) *measuring* — the Task 1 confusion matrix,
the Task 2 dynamics curves, and the Task 3 AUROC/precision diagnostics.
It is never passed to any loss, filter, relabel, or model-selection step in
Tasks 2–5. Verified by construction: `common.run_training()` only sees the
`train_labels` and `val_labels` arrays passed by the caller.

## 6. Reproducibility notes

- **Seed:** 69, set as the default in `common.add_common_args()`. Changing
  `--seed` on the command line reproduces the pipeline with a different
  initialization but should give near-identical metrics.
- **Deterministic mode:** pass `--deterministic` for bit-exact reruns. This
  requires `--num_workers 0` (default). With `--num_workers > 0` the
  augmentation RNG in workers breaks bit-exactness.
- **Task 2 does not retrain.** It reads Task 1's saved per-epoch arrays
  (`t1_noisy_ce_train_correct_all_epochs.npy`, `..._val_preds_all_epochs.npy`,
  `..._test_preds_all_epochs.npy`) which Task 1 writes specifically so that
  Task 2 stays within the "one noisy baseline" budget.
- **Identical recipe across tasks.** All `run_training()` calls use the same
  optimizer (SGD, lr=0.1, wd=5e-4, momentum=0.9), schedule
  (CosineAnnealingLR, T_max=60), augmentation (RandomHorizontalFlip +
  RandomCrop(32, padding=4)), and batch size (128). Only the loss function
  changes between methods, as the rules require.
- **Task 3 folds.** 5-fold stratified split on noisy labels, random state 69.
  Per-fold models use `seed + fold_idx` (69, 70, 71, 72, 73) so a bad init
  on one fold can't corrupt the whole score vector. Fold scores are z-scored
  per fold before pooling to handle per-fold CE-scale differences.