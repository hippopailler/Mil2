"""Utility functions for MIL."""

import os
import inspect
import numpy as np
import pandas as pd
import MIL.errors as errors
import MIL.util as util

from os.path import exists, join, isdir
from typing import Optional, Tuple, Union, Dict, List, Any, TYPE_CHECKING

from MIL.model.torch_utils import get_device
from MIL.mil import mil_config
from ._params import TrainerConfig
from MIL.util import load_json, log, path_to_name, zip_allowed
from MIL.dataset import Dataset


if TYPE_CHECKING:
    import torch


# -----------------------------------------------------------------------------


def aggregate_trainval_bags_by_slide(
    bags: np.ndarray,
    labels: Dict[str, int],
    train_slides: List[str],
    val_slides: List[str],
    *,
    log_manifest: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate training/validation bags by slide.

    Args:
        bags (np.ndarray): Array of bag paths.
        labels (dict): Dictionary mapping slide names to labels.
        train_slides (list): List of training slide names.
        val_slides (list): List of validation slide names.

    Keyword Args:
        log_manifest (str): Path to manifest file to write.
            Defaults to None.

    Returns:
        tuple: (bags, targets, train_idx, val_idx)

    """
    # Prepare targets
    targets = np.array([labels[path_to_name(f)] for f in bags])

    # Prepare training/validation indices
    train_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in train_slides])
    val_idx = np.array([i for i, bag in enumerate(bags)
                        if path_to_name(bag) in val_slides])

    # Write slide/bag manifest
    if log_manifest is not None:
        util.log_manifest(
            [bag for bag in bags if path_to_name(bag) in train_slides],
            [bag for bag in bags if path_to_name(bag) in val_slides],
            labels=labels,
            filename=log_manifest
        )

    return bags, targets, train_idx, val_idx


def get_labels(
    datasets: Union[Dataset, List[Dataset]],
    outcomes: Union[str, List[str]],
    model_type: str,
    *,
    format: str = 'name',
    events: Optional[str] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Get labels for a dataset.

    Args:
        datasets (Dataset or list(Dataset)): Dataset(s) containing labels.
        outcomes (str or list(str)): Outcome(s) to extract.
        model_type (str): Type of model to use.

    Keyword Args:
        format (str): Format for categorical labels. Either 'id' or 'name'.
            Defaults to 'name'.

    """
    if isinstance(datasets, Dataset):
        datasets = [datasets]

    # Prepare labels and slides
    labels = {}
    if model_type in ['classification', 'ordinal', 'multimodal']:
        all_unique = []
        for dts in datasets:
            _labels, _unique = dts.labels(outcomes, format=format)
            labels.update(_labels)
            all_unique.append(_unique)
        unique = np.unique(all_unique)
    elif model_type in ['survival', 'multimodal_survival']:
        if events is None:
            raise ValueError("For survival models, 'events' parameter must be provided")
        for dts in datasets:
            time_labels, _ = dts.labels(outcomes, use_float=True)
            event_labels, _ = dts.labels(events, use_float=True)
            # Create tuples of (time, event) for each slide
            for slide in time_labels:
                labels[slide] = (time_labels[slide][0], event_labels[slide][0])
        unique = None
    else:
        for dts in datasets:
            _labels, _unique = dts.labels(outcomes, use_float=True)
            labels.update(_labels)
        unique = None
    return labels, unique


def rename_df_cols(df, outcomes, model_type, inplace=False):
    """Rename columns of a DataFrame based on outcomes.

    This standarization of column names enables metrics calculation
    to be consistent across different models and outcomes.

    Args:
        df (pd.DataFrame): DataFrame with columns to rename.
            For classification outcomes, there is assumed to be a single "y_true"
            column which will be renamed to "{outcome}-y_true", and multiple
            "y_pred{n}" columns which will be renamed to "{outcome}-y_pred{n}".
            For regression outcomes, there are assumed to be multiple "y_true{n}"
            and "y_pred{n}" columns which will be renamed to "{outcome}-y_true{n}"
            and "{outcome}-y_pred{n}", respectively.
        outcomes (str or list(str)): Outcome(s) to append to column names.
            If there are multiple outcome names, these are joined with a hyphen.
        categorical (bool): Whether the outcomes are categorical.

    """
    if model_type in ['classification', 'ordinal', 'multimodal']:
        return _rename_categorical_df_cols(df, outcomes, inplace=inplace)
    elif model_type in ['survival', 'multimodal_survival']:
        return _rename_survival_df_cols(df, outcomes, inplace=inplace)
    else:
        return _rename_continuous_df_cols(df, outcomes, inplace=inplace)


def _rename_categorical_df_cols(df, outcomes, inplace=False):
    outcome_name = outcomes if isinstance(outcomes, str) else '-'.join(outcomes)
    return df.rename(
        columns={c: f"{outcome_name}-{c}" for c in df.columns if c != 'slide'},
        inplace=inplace
    )


def _rename_continuous_df_cols(df, outcomes, inplace=False):
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    cols_to_rename = {f'y_pred{o}': f"{outcomes[o]}-y_pred" for o in range(len(outcomes))}
    cols_to_rename.update({f'y_true{o}': f"{outcomes[o]}-y_true" for o in range(len(outcomes))})
    return df.rename(columns=cols_to_rename, inplace=inplace)

def _rename_survival_df_cols(df, outcomes, inplace=False):
    df = df.rename(columns={
        'y_true0': 'time-y_true',
        'y_true1': 'event-y_true',
        'y_pred0': 'time-y_pred'
    }, inplace=inplace)
    return df


# -----------------------------------------------------------------------------

def _detect_device(
    model: "torch.nn.Module",
    device: Optional[str] = None,
    verbose: bool = False
) -> "torch.device":
    """Auto-detect device from the given model."""
    import torch

    if device is None:
        device = next(model.parameters()).device
        if verbose:
            log.debug(f"Auto device detection: using {device}")
    elif isinstance(device, str):
        if verbose:
            log.debug(f"Using {device}")
        device = torch.device(device)
    return device


def _export_attention(
    dest: str,
    y_att: Union[List[np.ndarray], List[List[np.ndarray]]],
    slides: List[str]
) -> None:
    """Export attention scores to a directory."""
    if not exists(dest):
        os.makedirs(dest)
    for slide, att in zip(slides, y_att):

        if isinstance(att, (list, tuple)) and not zip_allowed():
            raise RuntimeError(
                "Cannot export multimodal attention scores to a directory (NPZ) "
                "when ZIP functionality is disabled. Enable zip functionality "
                "by setting 'SF_ALLOW_ZIP=1' in your environment, or by "
                "wrapping your script in 'with sf.util.enable_zip():'.")

        elif isinstance(att, (list, tuple)):
            out_path = join(dest, f'{slide}_att.npz')
            np.savez(out_path, *att)

        elif zip_allowed():
            out_path = join(dest, f'{slide}_att.npz')
            np.savez(out_path, att)

        else:
            out_path = join(dest, f'{slide}_att.npy')
            np.save(out_path, att)

    log.info(f"Attention scores exported to [green]{out_path}[/]")

