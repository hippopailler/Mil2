import numpy as np
import pandas as pd
from scipy import stats
import os

def compute_ground_truth_statistics(ground_truth):
    ground_truth = np.array(ground_truth)
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())

    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    order, label_1_count = compute_ground_truth_statistics(
        ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def auc_roc_ci(y_true, y_pred, alpha):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred
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

def fastDeLong(predictions_sorted_transposed, label_1_count):
    return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)


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

# ------------------------------------------------------------------------------
training_type = 'cross_validation'
sub1 = ''
sub2 = ''
path_pre = f'{training_type}/mil' # new_path
path_suf = f'classification/{training_type}/hypothesis_driven/pyrad-noimp'
outcomes = ['os_months_24'] # 'DCR', 'ORR', 

for outcome in outcomes:
    base_path = os.path.join(sub1, sub2, path_pre, outcome, path_suf)

    # Find all subdirectories ending with 'seed_0'
    seed_directories = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == 'seed_0':
                seed_directories.append(os.path.join(root, dir_name))

    # Process each seed directory
    for path in seed_directories:
        print(f"Processing: {path}")
        
        # if path does not contain folder called eval
        if not os.path.isdir(os.path.join(path, 'eval')):
            # list of all folders in path
            folders = [f.name for f in os.scandir(path) if f.is_dir()]
            # empty dataframe
            predictions = pd.DataFrame(columns=['slide', 'y_true', 'y_pred0', 'y_pred1'])
            # add to each folder path /eval/00000-mb_attention_mil/predictions.parquet
            for folder in folders:
                os.makedirs(os.path.join(path, folder, 'eval', '00000-mb_attention_mil'), exist_ok=True)
                # read predictions.parquet
                predictions_folder = pd.read_parquet(os.path.join(path, folder, 'predictions.parquet'))
                # concatenate predictions to dataframe
                predictions = pd.concat([predictions, predictions_folder])
        elif os.path.isdir(os.path.join(path, 'eval')):
            # read predictions.parquet
            predictions = pd.read_parquet(os.path.join(path, 'eval/00000-mb_attention_mil', 'predictions.parquet'))

        # add column called pred which is the softmax of pred0 and pred1
        predictions['pred'] = np.exp(predictions['y_pred1']) / (np.exp(predictions['y_pred0']) + np.exp(predictions['y_pred1']))
        # run auc_roc_ci
        auc, ci = auc_roc_ci(predictions['y_true'], predictions['pred'], 0.95)
        # save into path + eval.csv
        with open(os.path.join(path, 'eval_auc_ci.csv'), 'w') as f:
            f.write('auc, ci_lower, ci_upper\n')
            f.write(f'{auc}, {ci[0]}, {ci[1]}\n')
        
        print(f"Completed processing for {path}: AUC = {auc:.4f}, CI = [{ci[0]:.4f}, {ci[1]:.4f}]")







