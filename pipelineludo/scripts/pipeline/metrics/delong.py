import numpy as np
import pandas as pd
from scipy import stats

def compute_ground_truth_statistics(ground_truth, sample_weight):
    ground_truth = np.array(ground_truth)
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov



def auc_roc_ci(y_true, y_pred, alpha, sample_weight=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred,
        sample_weight
    )

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    return auc, ci

def compute_midrank(x):
  J = np.argsort(x)
  Z = x[J]
  N = len(x)
  T = np.zeros(N, dtype=float)
  i = 0
  while i < N:
      j = i
      while j < N and Z[j] == Z[i]:
          j += 1
      T[i:j] = 0.5*(i + j - 1)
      i = j
  T2 = np.empty(N, dtype=float)
  T2[J] = T + 1
  return T2


def compute_midrank_weight(x, sample_weight):
  J = np.argsort(x)
  Z = x[J]
  cumulative_weight = np.cumsum(sample_weight[J])
  N = len(x)
  T = np.zeros(N, dtype=float)
  i = 0
  while i < N:
      j = i
      while j < N and Z[j] == Z[i]:
          j += 1
      T[i:j] = cumulative_weight[i:j].mean()
      i = j
  T2 = np.empty(N, dtype=float)
  T2[J] = T
  return T2

def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)

def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_sample_weight


def save_cv_preds_and_compute_auc_ci(model, X, y, cv_folds, folds, alpha=0.95):
    all_y_true = []
    all_y_pred = []
    all_weights = []

    for site, (train_idx, test_idx) in enumerate(cv_folds.split(X, y, groups=folds)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        if len(np.unique(y_te)) < 2:
            continue

        sample_weight = compute_sample_weight(class_weight='balanced', y=y_tr)

        # train & predict
        model.fit(X_tr, y_tr, sample_weight=sample_weight)
        preds = model.predict_proba(X_te)[:, 1]

        # collect for overall
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(preds.tolist())
        sw_val = compute_sample_weight('balanced', y_te)
        all_weights.append(pd.Series(sw_val, index=y_te.index))

    # overall AUC + CI on all saved predictions
    y_all = np.array(all_y_true)
    pred_all = np.array(all_y_pred)
    weights_all = pd.concat(all_weights)
    auc, ci = auc_roc_ci(y_all, pred_all, alpha=alpha, sample_weight=weights_all.to_numpy())
    print(f'AUC: {auc}, 95% CI: {ci[0]} - {ci[1]}')

    return auc, ci


auc, ci = save_cv_preds_and_compute_auc_ci(model, X_train, y_train, custom_folds, folds)


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import compute_sample_weight

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import compute_sample_weight

def compute_experiment_auc_ci(
    base_dir: str,
    experiment: str,
    seed: str = "seed_0",
    fold_names=None,
    alpha: float = 0.95,
):
    """
    Per ogni fold in fold_names:
      - legge predictions.parquet con colonne y_true, y_pred0, y_pred1 (logits)
      - calcola probabilità di classe 1 con softmax
      - calcola pesi bilanciati sul test set
    Alla fine concatena tutto e invoca auc_roc_ci.
    """
    if fold_names is None:
        fold_names = ["GHD", "INT", "MH", "SZMC", "VHIO"]

    y_all, pred_all, w_all = [], [], []

    for f in fold_names:
        eval_dir = (
            Path(base_dir)
            / experiment
            / seed
            / f"fold_{f}"
            / "eval"
            / "00000-mb_attention_mil"
        )
        df = pd.read_parquet(eval_dir / "predictions.parquet")

        # ground truth
        y = df["y_true"].to_numpy()

        # logits → probabilità classe 1
        logit0 = df["y_pred0"].to_numpy()
        logit1 = df["y_pred1"].to_numpy()
        exp0 = np.exp(logit0)
        exp1 = np.exp(logit1)
        p1 = exp1 / (exp0 + exp1)

        # pesi bilanciati sul test set
        w = compute_sample_weight("balanced", y)

        y_all.append(y)
        pred_all.append(p1)
        w_all.append(w)

    # concatena tutti i fold
    y_all = np.concatenate(y_all)
    pred_all = np.concatenate(pred_all)
    w_all = np.concatenate(w_all)

    # chiama DeLong + CI
    auc, ci = auc_roc_ci(y_all, pred_all, alpha=alpha)

    print(f"{experiment:8s} → AUC={auc:.3f}, {int(alpha*100)}% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    return auc, ci


if __name__ == "__main__":
    base = (
        "/share/project10/home/smithadam/matteMIL/cross_validation/"
        "mil/os_months_6/classification/cross_validation/"
        "hypothesis_driven/pyrad-noimp"
    )
    experiments = [
        "rwd",
        "rwd_radfm",
        "rwd_radpy",
        "rwd_dp",
        "rwd_radfm_dp",
        "rwd_radpy_dp",
        "rwd_radpy_dp_genomics",
        "rwd_radfm_dp_genomics",
    ]

    # lancia la routine per ciascun esperimento
    for exp in experiments:
        compute_experiment_auc_ci(base, exp)
