"""Métriques pour l'évaluation des modèles MIL."""
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from os.path import join
import warnings
from sklearn import metrics
from typing import Dict, List, Optional, Union, Tuple
from util import log
from MIL import errors
from pandas.core.frame import DataFrame
from plot import scatter

def _generate_tile_roc(yt_and_yp: Tuple[np.ndarray, np.ndarray]) -> 'ClassifierMetrics':
    """Génère ROC au niveau des tuiles."""
    y_true, y_pred = yt_and_yp
    class_metrics = ClassifierMetrics(y_true, y_pred)
    return class_metrics

class ClassifierMetrics:
    """Classe pour calculer les métriques de classification."""
    
    def __init__(self, y_true, y_pred, autofit=True):
        self.y_true = y_true
        self.y_pred = y_pred
        
        self.fpr = None
        self.tpr = None
        self.threshold = None
        self.auroc = None
        self.precision = None
        self.recall = None
        self.ap = None

        if autofit:
            self.roc_fit()
            self.prc_fit()

    def roc_fit(self):
        """Calcule ROC curve."""
        self.fpr, self.tpr, self.threshold = metrics.roc_curve(self.y_true, self.y_pred)
        self.auroc = metrics.auc(self.fpr, self.tpr)

    def prc_fit(self):
        """Calcule Precision-Recall curve."""
        self.precision, self.recall, _ = metrics.precision_recall_curve(self.y_true, self.y_pred)
        self.ap = metrics.average_precision_score(self.y_true, self.y_pred)

def classification_metrics(
    df: pd.DataFrame,
    label: str = '',
    level: str = 'tile',
    data_dir: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """Generates categorical metrics (AUC/AP) from a set of predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred()

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving plots.
            If None, plots are not saved. Defaults to the current directory.

    Returns:
        Dict containing metrics, with the keys 'auc' and 'ap'.
    """
    label_start = "" if label == '' else f"{label}_"
    
    # Detect the number of outcomes and confirm that the number of outcomes
    # match the provided outcome names
    outcome_names = [c[:-8] for c in df.columns if c.endswith('-y_pred0')]
    if not len(outcome_names):
        raise errors.StatsError("No outcomes detected from dataframe.")

    all_auc = {outcome: [] for outcome in outcome_names}
    all_ap = {outcome: [] for outcome in outcome_names}
    def y_true_onehot(_df, i):
        return (_df.y_true == i).astype(int)

    def y_pred_onehot(_df, i):
        return (_df.y_pred_cat == i).astype(int)
    # Perform analysis separately for each outcome column
    for outcome in outcome_names:
        outcome_cols = [c for c in df.columns if c.startswith(f'{outcome}-')]
        
        # Remove the outcome name from the dataframe temporarily
        outcome_df = df[outcome_cols].rename(columns={
            orig_col: orig_col.replace(f'{outcome}-', '', 1)
            for orig_col in outcome_cols
        })
        log.info(f"Validation metrics for outcome [green]{outcome}[/]:")
        y_pred_cols = [f'y_pred{i}' for i in range(len([c for c in outcome_df.columns if c.startswith('y_pred')]))]
        num_cat = len(y_pred_cols)
        if not num_cat:
            raise errors.StatsError(
                f"Could not find predictions column for outcome {outcome}"
            )

        # Sort the prediction columns so that argmax will work as expected
        y_pred_cols = [f'y_pred{i}' for i in range(num_cat)]
        if len(y_pred_cols) != num_cat:
            raise errors.StatsError(
                "Malformed dataframe, unable to find all prediction columns"
            )
        if not all(col in outcome_df.columns for col in y_pred_cols):
            raise errors.StatsError("Malformed dataframe, invalid column names")

        # Convert to one-hot encoding
        outcome_df['y_pred_cat'] = outcome_df[y_pred_cols].values.argmax(1)

        log.debug(f"Calculating metrics with a thread pool")
        p = mp.dummy.Pool(8)
        yt_and_yp = [
            ((outcome_df.y_true == i).astype(int), outcome_df[f'y_pred{i}'])
            for i in range(num_cat)
        ]
        try:
            for i, fit in enumerate(p.imap(_generate_tile_roc, yt_and_yp)):
                if data_dir is not None:
                    fit.save_roc(data_dir, f"{label_start}{outcome}_{level}_ROC{i}")
                    fit.save_prc(data_dir, f"{label_start}{outcome}_{level}_PRC{i}")
                all_auc[outcome] += [fit.auroc]
                all_ap[outcome] += [fit.ap]
                auroc_str = 'NA' if not fit.auroc else f'{fit.auroc:.3f}'
                ap_str = 'NA' if not fit.ap else f'{fit.ap:.3f}'
                thresh = 'NA' if not fit.opt_thresh else f'{fit.opt_thresh:.3f}'
                log.info(
                    f"{level}-level AUC (cat #{i:>2}): {auroc_str} "
                    f"AP: {ap_str} (opt. threshold: {thresh})"
                )
        except ValueError as e:
            # Occurs when predictions contain NaN
            log.error(f'Error encountered when generating AUC: {e}')
            all_auc[outcome] = -1
            all_ap[outcome] = -1
        p.close()

        # Calculate tile-level accuracy.
        # Category-level accuracy is determined by comparing
        # one-hot predictions to one-hot y_true.
        #         
        # Calcul des métriques pour chaque catégorie
        for i in range(num_cat):
            try:
                yt_in_cat =  y_true_onehot(outcome_df, i)
                n_in_cat = yt_in_cat.sum()
                correct = y_pred_onehot(outcome_df.loc[yt_in_cat == 1], i).sum()
                category_accuracy = correct / n_in_cat
                perc = category_accuracy * 100
                log.info(f"Category {i} acc: {perc:.1f}% ({correct}/{n_in_cat})")
            except IndexError:
                log.warning(f"Error with category accuracy for cat # {i}")

    return {
        'auc': all_auc,
        'ap': all_ap
    }

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule l'indice de concordance."""
    E = y_pred[:, -1]  # Events
    y_pred = y_pred[:, :-1]  # Predictions
    return metrics.concordance_index_censored(
        event_indicator=E.astype(bool),
        event_time=y_true.flatten(),
        estimate=y_pred.flatten()
    )[0]

def survival_metrics(
    df: DataFrame,
    level: str = 'tile',
    label: str = '',
    data_dir: str = '',
) -> Dict[str, float]:
    """Generates survival metrics (concordance index) from a set of predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred().
            The dataframe columns should be appropriately named using
            sf.stats.name_columns().

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving plots.
            Defaults to None.

    Returns:
        Dict containing metrics, with the key 'c_index'.
    """
    survival_cols = ('time-y_true', 'time-y_pred', 'event-y_true')
    if any(c not in df.columns for c in survival_cols):
        raise ValueError(
            "Improperly formatted dataframe to survival_metrics(), "
            f"must have columns {survival_cols}. Got: {list(df.columns)}"
        )

    # Calculate metrics
    try:
        c_index = concordance_index(
            df['time-y_true'].values,
            df[['time-y_pred', 'event-y_true']].values,
        )
        c_str = 'NA' if not c_index else f'{c_index:.3f}'
        log.info(f"C-index ({level}-level): {c_str}")
    except ZeroDivisionError as e:
        log.error(f"Error calculating concordance index: {e}")
        c_index = -1
    return {
        'c_index': c_index
    }
def regression_metrics(
    df: DataFrame,
    label: str = '',
    level: str = 'tile',
    data_dir: str = '',
) -> Dict[str, List[float]]:
    """Generates metrics (R^2, coefficient of determination) from predictions.

    Args:
        df (pd.DataFrame): Pandas DataFrame containing labels, predictions,
            and optionally uncertainty, as returned by sf.stats.df_from_pred()

    Keyword args:
        label (str, optional): Label prefix/suffix for ROCs.
            Defaults to an empty string.
        level (str, optional): Group-level for the predictions. Used for
            labeling plots. Defaults to 'tile'.
        data_dir (str, optional): Path to data directory for saving.
            Defaults to None.
        neptune_run (:class:`neptune.Run`, optional): Neptune run in which to
            log results. Defaults to None.

    Returns:
        Dict containing metrics, with the key 'r_squared'.
    """

    label_end = "" if label == '' else f"_{label}"

    # Detect the outcome names
    outcome_names = [c[:-7] for c in df.columns if c.endswith('-y_pred')]
    _outcomes_by_true = [c[:-7] for c in df.columns if c.endswith('-y_true')]
    if ((sorted(outcome_names) != sorted(_outcomes_by_true))
       or not len(outcome_names)):
        raise ValueError("Improperly formatted dataframe to regression_metrics(); "
                         "could not detect outcome names. Ensure that "
                         "prediction columns end in '-y_pred' and ground-truth "
                         "columns end in '-y_true'. Try setting column names "
                         "with slideflow.stats.name_columns(). "
                         f"DataFrame columns: {list(df.columns)}")

    # Calculate metrics
    y_pred_cols = [f'{o}-y_pred' for o in outcome_names]
    y_true_cols = [f'{o}-y_true' for o in outcome_names]
    r_squared = scatter(
        df[y_true_cols].values,
        df[y_pred_cols].values,
        data_dir,
        f"{label_end}_by_{level}",
    )

    # Show results
    for o, r in zip(outcome_names, r_squared):
        r_str = "NA" if not r else f'{r:.3f}'
        log.info(f"[green]{o}[/]: R-squared ({level}-level): {r_str}")

    return {
        'r_squared': r_squared,
    }
