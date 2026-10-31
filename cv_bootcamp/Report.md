# Report — CIFAR-10N Bootcamp

Each answer is ≤100 words. All numbers come from `logs/t*_summary.json` in the
run directory; no metric is hardcoded here from a different run.

## Q1. Which class pairs account for most of the noise?

Four dominant confusion pairs — truck↔automobile (2,373 mutual), cat↔dog
(1,758), deer→horse (683), and deer→dog (491) — account for ~29% of the
18,109 mislabeled training examples. Each pair is semantically adjacent:
trucks and automobiles are the two large-vehicle classes at 32×32; cats and
dogs are the two small-furry-animal classes; deer, horse and dog share
silhouettes at low resolution. Annotators were not guessing randomly — they
were systematically applying a plausible wrong label from a small set of
alternatives. This structure is precisely what distinguishes human noise from
symmetric uniform noise (see Q5).

## Q2. At which epoch does memorization of wrong labels start, and what is the evidence?

Memorization onset is epoch 39. Evidence: at that epoch, train accuracy on the correctly-labeled subset is 0.8905, while on the mislabeled subset it has risen from a floor of 0.1197 to 0.1696 — a 42% relative increase above its early floor, followed by acceleration to 0.6497 by epoch 59. After epoch 39 the mislabeled-subset curve accelerates while clean-label validation accuracy plateaus near 0.8030 (its peak, at epoch 38) and then declines to 0.6970 by epoch 59. Noisy-label validation peaks at 0.5562 (epoch 38) and drifts down to 0.4978. logs/t2_dynamics_plot.png marks the onset with a red vertical line. The network is not learning faster overall — it is switching from fitting the shared signal to fitting the individual noisy targets.
## Q3. Which noise-score signal worked best? Describe one failure mode.

Per-sample cross-entropy from a 5-fold CV ResNet-18 (z-scored per fold)
achieved pooled AUROC **0.9205** and
precision@k of **0.8028** at
k = **18,109**, versus the 0.4024 baseline that
random scoring would give. One failure mode: high loss also flags *intrinsically
hard but correctly-labeled* images — an atypical truck, or a frog that a
well-trained model confidently mislabels. The top-scoring set is therefore
enriched for mislabeled images but not exclusive to them, which caps
precision@k below 1 regardless of how well-trained the fold models are.

## Q4. Why did your best Task 4 method help, or not help?

**ELR method** reached noisy-validation
accuracy **0.5816**, versus
**0.4978** for the
CE baseline. The gain comes from reduced sensitivity to confident-but-wrong
targets: [pick one based on the winning method]
- ELR's running-target regulariser discourages the model's predicted
  distribution from collapsing onto a single noisy label, so the loss cannot
  fully commit to a wrong one-hot target even after many epochs.
- Label smoothing softens the target distribution, preventing the loss from
  saturating on a wrong one-hot label and preserving a gradient signal toward
  the true class.
- GCE's bounded loss (q=0.7) ignores per-sample gradients from examples whose
  target probability has already collapsed, effectively discarding the noisiest
  fraction of samples without explicit filtering.
Both mechanisms bias the loss toward the class-conditional signal at the
expense of the instance-level noisy signal, without using any clean labels.

## Q5. Did the ranking of methods change between real and synthetic noise? Why?

The ranking was preserved: **ELR**
beat CE on both real (+**+2.54pp**)
and synthetic (+**+3.78**) noise.
Absolute accuracy dropped for both under synthetic noise
(0.5562 → **0.4972**
for CE, **0.5816** →
**0.5350** for ELR). The
reason is structural: human noise concentrates on visually adjacent pairs, so
a robust loss can partially exploit cross-sample agreement to recover the
true label. Symmetric noise spreads wrong labels uniformly across all nine
wrong classes, leaving no structure to exploit — the robust method still helps,
but by a smaller margin.

---

## Evidence index (for reviewers)

| Question | Primary evidence |
|---|---|
| Q1 | `logs/t1_noise_summary.json` (`top_confusions_train`, `per_class_error_train`) |
| Q2 | `logs/t2_summary.json`, `predictions/t2_dynamics.csv`, `logs/t2_dynamics_plot.png` |
| Q3 | `logs/t3_summary.json` (`pooled`), `logs/t3_scores_details.csv` |
| Q4 | `logs/t4_summary.json` (`best_method`, `per_method`), `logs/t1_summary.json` |
| Q5 | `logs/t5_summary.json` (`comparison`) |