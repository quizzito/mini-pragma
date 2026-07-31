"""
M6 final step: bootstrap confidence intervals to properly compare the
foundation model (embedding probe) against XGBoost baselines, rather than
trusting single point-estimate metrics on small test sets.

Bootstrapping: resample the test set WITH REPLACEMENT many times, recompute
the metric each time, and look at the spread of results. This tells us how
much a metric could plausibly vary just from which particular users ended
up in our test set -- not a real difference in model quality.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

N_BOOTSTRAP = 1000


def bootstrap_metric(y_test, y_pred, metric_fn, n_bootstrap=N_BOOTSTRAP):
    """Resample (y_test, y_pred) pairs together, with replacement, n_bootstrap
    times, computing the metric each time. Returns an array of scores."""
    n = len(y_test)
    scores = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)  # sample n indices, with replacement
        y_test_sample = y_test[indices]
        y_pred_sample = y_pred[indices]

        # Skip degenerate resamples with only one class present (metric undefined)
        if len(np.unique(y_test_sample)) < 2:
            continue

        score = metric_fn(y_test_sample, y_pred_sample)
        scores.append(score)

    return np.array(scores)


def compare_task(task_name):
    baseline_y_test = np.load(f"results/baseline_{task_name}_y_test.npy")
    baseline_y_pred = np.load(f"results/baseline_{task_name}_y_pred.npy")
    probe_y_test = np.load(f"results/probe_{task_name}_y_test.npy")
    probe_y_pred = np.load(f"results/probe_{task_name}_y_pred.npy")

    print(f"\n{'=' * 60}")
    print(f"{task_name}")
    print(f"{'=' * 60}")

    for metric_name, metric_fn in [("ROC-AUC", roc_auc_score), ("PR-AUC", average_precision_score)]:
        baseline_scores = bootstrap_metric(baseline_y_test, baseline_y_pred, metric_fn)
        probe_scores = bootstrap_metric(probe_y_test, probe_y_pred, metric_fn)

        baseline_ci = np.percentile(baseline_scores, [2.5, 97.5])
        probe_ci = np.percentile(probe_scores, [2.5, 97.5])

        print(f"\n{metric_name}:")
        print(f"  Baseline (XGBoost): {baseline_scores.mean():.4f}  "
              f"[95% CI: {baseline_ci[0]:.4f}, {baseline_ci[1]:.4f}]")
        print(f"  Probe (mini-PRAGMA): {probe_scores.mean():.4f}  "
              f"[95% CI: {probe_ci[0]:.4f}, {probe_ci[1]:.4f}]")

        overlap = not (probe_ci[1] < baseline_ci[0] or baseline_ci[1] < probe_ci[0])
        if overlap:
            print(f"  -> CIs OVERLAP: difference is NOT clearly statistically meaningful")
        elif probe_ci[0] > baseline_ci[1]:
            print(f"  -> Probe CLEARLY BEATS baseline (no CI overlap)")
        else:
            print(f"  -> Baseline CLEARLY BEATS probe (no CI overlap)")


if __name__ == "__main__":
    for task in ["credit_default", "fraud", "engagement"]:
        compare_task(task)