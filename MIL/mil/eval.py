"""Tools for evaluation MIL models."""

import os
import inspect
import pandas as pd
import numpy as np
import MIL.errors as errors

from rich.progress import track
from os.path import join, exists, dirname
from typing import Union, List, Optional, Callable, Tuple, Any, TYPE_CHECKING
from MIL.util import load_json, write_json, get_new_model_dir, log, path_to_name
from MIL.dataset import Dataset
from ._params import TrainerConfig
from . import utils

if TYPE_CHECKING:
    import torch
    from .features import MILFeatures

# -----------------------------------------------------------------------------
# User-facing API for evaluation and prediction.

def eval_mil(
    weights: str,
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: Optional[TrainerConfig] = None,
    *,
    events: Optional[str] = None,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    uq: bool = False,
    aggregation_level: Optional[str] = None,
    **heatmap_kwargs
) -> pd.DataFrame:
    """Evaluate a multiple-instance learning model.

    Saves results for the evaluation in the target folder, including
    predictions (parquet format), attention (Numpy format for each slide),
    and attention heatmaps (if ``attention_heatmaps=True``).

    Logs classifier metrics (AUROC and AP) to the console.

    Args:
        weights (str): Path to model weights to load.
        dataset (sf.Dataset): Dataset to evaluation.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Not available for multi-modal MIL models. Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        aggregation_level (str, optional): Aggregation level for predictions.
            Either 'slide' or 'patient'. Defaults to None (uses the model
            configuration).
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    """
    if isinstance(bags, str):
        utils._verify_compatible_tile_size(weights, bags)

    model, config = utils.load_model_weights(weights, config)
    model.eval()
    params = {
        'model_path': weights,
        'eval_bags': bags,
        'eval_filters': dataset._filters,
        'mil_params': load_json(join(weights, 'mil_params.json'))
    }
    return config.eval(
        model,
        dataset,
        outcomes,
        bags,
        events=events,
        outdir=outdir,
        attention_heatmaps=attention_heatmaps,
        uq=uq,
        params=params,
        aggregation_level=aggregation_level,
        **heatmap_kwargs
    )


def predict_mil(
    model: Union[str, Callable],
    dataset: "Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[str, np.ndarray, List[str]],
    *,
    events: Optional[str] = None,
    config: Optional[TrainerConfig] = None,
    attention: bool = False,
    aggregation_level: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions for a dataset from a saved MIL model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        dataset (sf.Dataset): Dataset from which to generation predictions.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Keyword args:
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for the MIL model. Required if model is a loaded ``torch.nn.Module``.
            Defaults to None.
        attention (bool): Whether to calculate attention scores. Defaults to False.
        uq (bool): Whether to generate uncertainty estimates. Experimental. Defaults to False.
        aggregation_level (str): Aggregation level for predictions. Either 'slide'
            or 'patient'. Defaults to None.
        attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of predictions.

        list(np.ndarray): Attention scores (if ``attention=True``)
    """
    # Load the model
    if isinstance(model, str):
        model_path = model
        model, config = utils.load_model_weights(model_path, config)
        model.eval()

        if isinstance(bags, str):
            utils._verify_compatible_tile_size(model_path, bags)
    elif config is None:
        raise ValueError("If model is not a path, a TrainerConfig object must be provided via the 'config' argument.")

    # Validate aggregation level.
    if aggregation_level is None:
        aggregation_level = config.aggregation_level
    if aggregation_level not in ('slide', 'patient'):
        raise ValueError(
            f"Unrecognized aggregation level: '{aggregation_level}'. "
            "Must be either 'patient' or 'slide'."
        )

    # Prepare labels.
    labels, _ = utils.get_labels(dataset, outcomes, config.model_type, events=events, format='id')

    # Prepare bags and targets.
    slides = list(labels.keys())
    if isinstance(bags, str):
        bags = dataset.get_bags(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Aggregate bags by slide or patient.
    if aggregation_level == 'patient':
        if config.model_type == 'multimodal_survival':
            raise NotImplementedError(
                "Patient-level aggregation not yet supported for multimodal survival models"
            )
        # Get nested list of bags, aggregated by slide.
        slide_to_patient = dataset.patients()
        n_slide_bags = len(bags)
        bags, y_true = utils.aggregate_bags_by_patient(bags, labels, slide_to_patient)
        log.info(f"Aggregated {n_slide_bags} slide bags to {len(bags)} patient bags.")

        # Create prediction dataframe.
        patients = [slide_to_patient[path_to_name(b[0])] for b in bags]
        df_dict = dict(patient=patients, y_true=y_true)

    else:
        # Ensure slide names are sorted according to the bags.
        slides = [path_to_name(b) for b in bags]
        y_true = np.array([labels[s] for s in slides])

        # Create prediction dataframe.
        df_dict = dict(slide=slides)

        # Handle continous outcomes.
        if len(y_true.shape) > 1:
            for i in range(y_true.shape[-1]):
                df_dict[f'y_true{i}'] = y_true[:, i]
        else:
            df_dict['y_true'] = y_true

    # Inference.
    model.eval()
    pred_out = config.predict(model, bags, attention=attention, **kwargs)
    if kwargs.get('uq'):
        y_pred, y_att, y_uq = pred_out
    else:
        y_pred, y_att = pred_out

    # Update dataframe with predictions.
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    if kwargs.get('uq'):
        for i in range(y_uq.shape[-1]):
            df_dict[f'uncertainty{i}'] = y_uq[:, i]
    df = pd.DataFrame(df_dict)

    if config.model_type in ['survival', 'multimodal_survival']:
        df['y_pred0'] = -df['y_pred0']

    if attention:
        return df, y_att
    else:
        return df

def predict_multimodal_mil(
    model: Union[str, Callable],
    dataset: "Dataset",
    outcomes: Union[str, List[str]],
    bags: Union[np.ndarray, List[List[str]]],
    *,
    config: Optional[TrainerConfig] = None,
    attention: bool = False,
    aggregation_level: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[np.ndarray]]]:
    """Generate predictions for a dataset from a saved multimodal MIL model.

    Args:
        model (torch.nn.Module): Model from which to generate predictions.
        dataset (sf.Dataset): Dataset from which to generation predictions.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Keyword args:
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for the MIL model. Required if model is a loaded ``torch.nn.Module``.
            Defaults to None.
        attention (bool): Whether to calculate attention scores. Defaults to False.
        uq (bool): Whether to generate uncertainty estimates. Defaults to False.
        aggregation_level (str): Aggregation level for predictions. Either 'slide'
            or 'patient'. Defaults to None.
        attention_pooling (str): Attention pooling strategy. Either 'avg'
                or 'max'. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of predictions.

        list(np.ndarray): Attention scores (if ``attention=True``)
    """

    # Load the model
    if isinstance(model, str):
        model_path = model
        model, config = utils.load_model_weights(model_path, config)
        model.eval()

        # Verify tile size compatibility for each bag source.
        for b in bags:
            if isinstance(b, str):
                utils._verify_compatible_tile_size(model_path, b)
    elif config is None:
        raise ValueError("If model is not a path, a TrainerConfig object must be provided via the 'config' argument.")

    # Validate aggregation level.
    if aggregation_level is not None and aggregation_level != 'slide':
        raise ValueError(
            f"Unrecognized aggregation level: '{aggregation_level}'. "
            "Multimodal MIL models only support 'slide' aggregation."
        )

    # Prepare labels.
    labels, _ = utils.get_labels(dataset, outcomes, config.model_type, format='id')

    # Prepare bags and targets.
    slides = list(labels.keys())

    # Load multimodal bags.
    if isinstance(bags[0], str):
        bags, val_slides = utils._get_nested_bags(dataset, bags)

    # This is where we would aggregate bags by slide or patient.
    # This is not yet supported.

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b[0]) for b in bags]
    y_true = np.array([labels[s] for s in slides])

    # Create prediction dataframe.
    df_dict = dict(slide=slides)

    # Handle continous outcomes.
    if len(y_true.shape) > 1:
        for i in range(y_true.shape[-1]):
            df_dict[f'y_true{i}'] = y_true[:, i]
    else:
        df_dict['y_true'] = y_true

    # Inference.
    model.eval()
    y_pred, y_att = config.predict(model, bags, attention=attention, **kwargs)

    # Update dataframe with predictions.
    for i in range(y_pred.shape[-1]):
        df_dict[f'y_pred{i}'] = y_pred[:, i]
    df = pd.DataFrame(df_dict)

    if attention:
        return df, y_att
    else:
        return df


# -----------------------------------------------------------------------------
# Prediction from bags.

def predict_from_bags(
    model: "torch.nn.Module",
    bags: Union[np.ndarray, List[str]],
    *,
    attention: bool = False,
    attention_pooling: Optional[str] = None,
    use_lens: bool = False,
    device: Optional[Any] = None,
    apply_softmax: Optional[bool] = None,
    uq: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate MIL predictions for a list of bags.

    Predictions are generated for each bag in the list one at a time, and not batched.

    Args:
        model (torch.nn.Module): Loaded PyTorch MIL model.
        bags (np.ndarray, list(str)): Bags to generate predictions for. Each bag
            should contain PyTorch array of features from all tiles in a slide,
            with the shape ``(n_tiles, n_features)``.

    Keyword Args:
        attention (bool): Whether to calculate attention scores. Defaults to False.
        attention_pooling (str, optional): Pooling strategy for attention scores.
            Can be 'avg', 'max', or None. Defaults to None.
        use_lens (bool): Whether to use the length of each bag as an additional
            input to the model. Defaults to False.
        device (str, optional): Device on which to run inference. Defaults to None.
        apply_softmax (bool): Whether to apply softmax to the model output. Defaults
            to True for categorical outcomes, False for continuous outcomes.
        uq (bool): Whether to generate uncertainty estimates. Defaults to False.

    Returns:
        Tuple[np.ndarray, List[np.ndarray]]: Predictions and attention scores.

    """
    import torch

    attention, uq = utils._validate_model(model, attention, uq, allow_errors=True)
    model.eval()

    y_pred = []
    y_att  = []
    uncertainty = []
    device = utils._detect_device(model, device, verbose=True)

    for bag in bags:
        if utils._is_list_of_paths(bag):
            # If bags are passed as a list of paths, load them individually.
            loaded = torch.cat([utils._load_bag(b).to(device) for b in bag], dim=0)
        else:
            loaded = utils._load_bag(bag).to(device)
        loaded = torch.unsqueeze(loaded, dim=0)

        with torch.inference_mode():
            # Run inference.
            _y_pred, _y_att, _y_uq = run_inference(
                model,
                loaded,
                attention=attention,
                attention_pooling=attention_pooling,
                uq=uq,
                apply_softmax=apply_softmax,
                device=device,
                use_lens=use_lens
            )

            # Convert to numpy.
            if _y_pred is not None:
                _y_pred = _y_pred.cpu().numpy()
            if _y_att is not None:
                _y_att = _y_att.cpu().numpy()
            if _y_uq is not None:
                _y_uq = _y_uq.cpu().numpy()

            # Append to running lists.
            y_pred.append(_y_pred)
            if _y_att is not None:
                y_att.append(_y_att)
            if _y_uq is not None:
                uncertainty.append(_y_uq)

    yp = np.concatenate(y_pred, axis=0)
    if uq:
        uncertainty = np.concatenate(uncertainty, axis=0)
        return yp, y_att, uncertainty
    else:
        return yp, y_att


def predict_from_multimodal_bags(
    model: "torch.nn.Module",
    bags: Union[List[np.ndarray], List[List[str]]],
    *,
    attention: bool = True,
    attention_pooling: Optional[str] = None,
    use_lens: bool = True,
    device: Optional[Any] = None,
    apply_softmax: Optional[bool] = None,
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """Generate multi-mag MIL predictions for a nested list of bags.

    Args:
        model (torch.nn.Module): Loaded PyTorch MIL model.
        bags (list(list(str))): Nested list of bags to generate predictions for.
            Each bag should contain PyTorch array of features from all tiles in a slide,
            with the shape ``(n_tiles, n_features)``.

    Keyword Args:
        attention (bool): Whether to calculate attention scores. Defaults to False.
        attention_pooling (str, optional): Pooling strategy for attention scores.
            Can be 'avg', 'max', or None. Defaults to None.
        use_lens (bool): Whether to use the length of each bag as an additional
            input to the model. Defaults to False.
        device (str, optional): Device on which to run inference. Defaults to None.
        apply_softmax (bool): Whether to apply softmax to the model output. Defaults
            to True for categorical outcomes, False for continuous

    Returns:
        Tuple[np.ndarray, List[List[np.ndarray]]]: Predictions and attention scores.

    """
    import torch

    y_pred = []
    n_mag = len(bags[0])
    y_att  = [[] for _ in range(n_mag)]
    device = utils._detect_device(model, device, verbose=True)

    # Ensure the model has attention capabilities.
    if attention and not hasattr(model, 'calculate_attention'):
        log.warning(
            "Model '{}' does not have a method 'calculate_attention'. "
            "Unable to calculate or display attention heatmaps.".format(
                model.__class__.__name__
            )
        )
        attention = False

    for bag in bags:
        loaded = [torch.unsqueeze(utils._load_bag(b).to(device), dim=0)
                  for b in bag]
        with torch.inference_mode():
            if use_lens:
                model_args = [(mag_bag, torch.from_numpy(np.array([mag_bag.shape[1]])).to(device))
                              for mag_bag in loaded]
            else:
                model_args = (loaded,)
            model_out = model(*model_args)
            if attention:
                raw_att = model.calculate_attention(*model_args)
                for mag in range(n_mag):
                    att = torch.squeeze(raw_att[mag], dim=0)
                    att = utils._pool_attention(torch.squeeze(att), pooling=attention_pooling)
                    # If we have multi-channel attention, then the attenion channel (last) needs to
                    # be moved to the first dimension.
                    if len(att.shape) == 2:
                        att = torch.moveaxis(att, -1, 0)
                    y_att[mag].append(att.cpu().numpy())
            if apply_softmax:
                model_out = torch.nn.functional.softmax(model_out, dim=1)
            y_pred.append(model_out.cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att

def predict_from_mixed_bags(
    model: "torch.nn.Module",
    bags: List[str],
    *,
    attention: bool = False,
    use_lens: bool = False, # for compatibility with other predict_from_bags
    uq: bool = False, # for compatibility with other predict_from_bags
    device: Optional[Any] = None,
    apply_softmax: Optional[bool] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate predictions from mixed multimodal bags.

    Args:
        model: PyTorch model
        bags: List of paths to .pt files containing mixed multimodal features

    Keyword Args:
        attention: Whether to return attention scores
        device: Device on which to run inference
        apply_softmax: Whether to apply softmax to outputs

    Returns:
        Tuple containing:
        - Predictions array (shape: n_bags x n_classes)
        - List of attention arrays (if attention=True)
    """
    import torch
    
    device = utils._detect_device(model, device, verbose=False)
    y_pred = []
    y_att = []

    for bag in bags:
        # Load the multimodal bag dictionary
        bag_dict = torch.load(bag)
        
        # Get features and mask
        mask = bag_dict['mask'].to(device)
        features = []
        for i in range(1, len(mask) + 1):
            feat_key = f'feature{i}'
            features.append(bag_dict[feat_key].to(device))

        # Add batch dimension
        features = [f.unsqueeze(0) for f in features]
        mask = mask.unsqueeze(0)

        with torch.inference_mode():
            # Forward pass
            if attention:
                _pred, _att = model(*features, mask, attention=True, decode=False)
                y_att.append(_att.squeeze(0).cpu().numpy())
            else:
                _pred = model(*features, mask)

            if apply_softmax:
                _pred = torch.nn.functional.softmax(_pred, dim=1)
            
            y_pred.append(_pred.cpu().numpy())

    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att if attention else None

# -----------------------------------------------------------------------------
# Low-level runners for inference and evaluation.

def run_inference(
    model: "torch.nn.Module",
    loaded_bags: "torch.Tensor",
    *,
    attention: bool = False,
    attention_pooling: Optional[str] = None,
    uq: bool = False,
    forward_kwargs: Optional[dict] = None,
    apply_softmax: Optional[bool] = None,
    use_lens: Union[bool, "torch.Tensor"] = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Low-level interface for running inference on a MIL model.

    Args:
        model (torch.nn.Module): Loaded PyTorch MIL model.
        loaded_bags (torch.Tensor): Loaded bags to run inference on.

    Keyword Args:
        attention (bool): Whether to calculate attention scores. Defaults to False.
        attention_pooling (str, optional): Pooling strategy for attention scores.
            Can be 'avg', 'max', or None. Defaults to None.
        uq (bool): Whether to generate uncertainty estimates. Defaults to False.
        forward_kwargs (dict, optional): Additional keyword arguments to pass to
            the model's forward function. Defaults to None.
        apply_softmax (bool): Whether to apply softmax to the model output. Defaults
            to True for categorical outcomes, False for continuous outcomes.
        use_lens (bool, torch.Tensor): Whether to use the length of each bag as an
            additional input to the model. If a tensor is passed, this will be used
            as the lens. Defaults to False.
        device (str, optional): Device on which to run inference. Defaults to None.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]: Predictions,
        attention scores, and uncertainty estimates. For multi-dimensional attention,
        the first dimension of the attention scores will be the attention channel.

    """
    import torch

    if forward_kwargs is None:
        forward_kwargs = dict()

    y_pred, y_att, y_uncertainty = None, None, None

    # Prepare lens
    device = utils._detect_device(model, device, verbose=False)
    if isinstance(use_lens, bool) and use_lens:
        lens = torch.full((loaded_bags.shape[0],), loaded_bags.shape[1], device=device)
        model_args = (loaded_bags, lens)
    elif use_lens is not False and use_lens is not None:
        model_args = (loaded_bags, use_lens)
    else:
        model_args = (loaded_bags,)

    if uq and 'uq' in inspect.signature(model.forward).parameters:
        kw = dict(uq=True, **forward_kwargs)
    elif uq:
        raise RuntimeError("Model does not support UQ.")
    else:
        kw = forward_kwargs

    # Check if the model can return attention during inference. 
    # If so, this saves us a forward pass through the model.
    if attention and 'return_attention' in inspect.signature(model.forward).parameters:
        model_out, y_att = model(*model_args, return_attention=True, **kw)
    # Otherwise, use the model's `calculate_attention` function directly.
    elif attention:
        model_out = model(*model_args, **kw)
        y_att = model.calculate_attention(*model_args)
    else:
        model_out = model(*model_args, **kw)

    # Parse uncertainty from model output.
    if uq:
        y_pred, y_uncertainty = model_out
    else:
        y_pred = model_out

    if attention:
        y_att = utils._pool_attention(torch.squeeze(y_att), pooling=attention_pooling)
        # If we have multi-channel attention, then the attenion channel (last) needs to
        # be moved to the first dimension.
        if len(y_att.shape) == 2:
            y_att = torch.moveaxis(y_att, -1, 0)

    if apply_softmax:
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    return y_pred, y_att, y_uncertainty


def run_eval(
    model: "torch.nn.Module",
    dataset: Dataset,
    outcomes: Union[str, List[str]],
    bags: Union[str, List[str]],
    config: TrainerConfig,
    *,
    events: Optional[str] = None,
    outdir: str = 'mil',
    attention_heatmaps: bool = False,
    uq: bool = False,
    params: Optional[dict] = None,
    aggregation_level: Optional[str] = None,
    **heatmap_kwargs
) -> pd.DataFrame:
    """Evaluate a standard, single-mode multi-instance learning model.

    Args:
        model (torch.nn.Module): Loaded PyTorch MIL model.
        dataset (sf.Dataset): Dataset to evaluation.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model.

    Keyword arguments:
        outdir (str): Path at which to save results.
        attention_heatmaps (bool): Generate attention heatmaps for slides.
            Defaults to False.
        interpolation (str, optional): Interpolation strategy for smoothing
            attention heatmaps. Defaults to 'bicubic'.
        cmap (str, optional): Matplotlib colormap for heatmap. Can be any
            valid matplotlib colormap. Defaults to 'inferno'.
        norm (str, optional): Normalization strategy for assigning heatmap
            values to colors. Either 'two_slope', or any other valid value
            for the ``norm`` argument of ``matplotlib.pyplot.imshow``.
            If 'two_slope', normalizes values less than 0 and greater than 0
            separately. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of predictions.
    """
    # Generate predictions.
    predict_kwargs = dict(
        model=model,
        dataset=dataset,
        config=config,
        outcomes=outcomes,
        events=events,
        bags=bags,
        attention=True,
        aggregation_level=aggregation_level
    )
    if config.is_multimodal:
        if uq:
            log.warning("Uncertainty estimates are not supported for multi-modal models.")
        df, y_att = predict_multimodal_mil(**predict_kwargs)
    else:
        df, y_att = predict_mil(uq=uq, **predict_kwargs)

    # Save results.
    if outdir:
        if not exists(outdir):
            os.makedirs(outdir)
        model_dir = get_new_model_dir(outdir, config.model_config.model)
        if params is not None:
            write_json(params, join(model_dir, 'mil_params.json'))
        pred_out = join(model_dir, 'predictions.parquet')
        df.to_parquet(pred_out)
        log.info(f"Predictions saved to [green]{pred_out}[/]")
    else:
        model_dir = None

    # Print classification metrics, including per-category accuracy)
    metrics_df = utils.rename_df_cols(df, outcomes, model_type=config.model_type)
    config.run_metrics(metrics_df, level='slide', outdir=model_dir)

    # Export attention
    if outdir and y_att:
        if 'slide' in df.columns:
            slides_or_patients = df.slide.values
        elif 'patient' in df.columns:
            slides_or_patients = df.patient.values
        else:
            raise ValueError("Malformed dataframe; cannot find 'slide' or 'patient' column.")
        utils._export_attention(join(model_dir, 'attention'), y_att, slides_or_patients)

    # Attention heatmaps
    # Not supported for multimodal models
    if attention_heatmaps and not config.is_multimodal:
        log.warning("Cannot generate attention heatmaps for multi-modal models.")

    return df

# -----------------------------------------------------------------------------
# Tile-level predictions.

def get_mil_tile_predictions(
    weights: str,
    dataset: "Dataset",
    bags: Union[str, np.ndarray, List[str]],
    *,
    config: Optional[TrainerConfig] = None,
    outcomes: Union[str, List[str]] = None,
    dest: Optional[str] = None,
    uq: bool = False,
    device: Optional[Any] = None,
    tile_batch_size: int = 512,
    **kwargs
) -> pd.DataFrame:
    """Generate tile-level predictions for a MIL model.

    Args:
        weights (str): Path to model weights to load.
        dataset (:class:`slideflow.Dataset`): Dataset.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.

    Keyword Args:
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.
        outcomes (str, list(str)): Outcomes.
        dest (str): Path at which to save tile predictions.
        uq (bool): Whether to generate uncertainty estimates. Defaults to False.
        device (str, optional): Device on which to run inference. Defaults to None.
        tile_batch_size (int): Batch size for tile-level predictions. Defaults
            to 512.
        attention_pooling (str): Attention pooling strategy. Either 'avg'
            or 'max'. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe of tile predictions.

    """
    import torch

    if isinstance(bags, str):
        utils._verify_compatible_tile_size(weights, bags)

    # Load model and configuration.
    model, config = utils.load_model_weights(weights, config)
    device = utils._detect_device(model, device, verbose=True)
    model.eval()
    model.to(device)

    if outcomes is not None:
        labels, _ = utils.get_labels(dataset, outcomes, config.model_type, format='id')

    # Prepare bags.
    slides = dataset.slides()
    if isinstance(bags, str):
        bags = dataset.get_bags(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b) for b in bags]
    
    log.info("Generating predictions for {} slides and {} bags.".format(len(slides), len(bags)))
    
    # Set model to eval, and prepare bags.
    use_attention, uq = utils._validate_model(model, True, uq, allow_errors=True)

    # First, start with slide-level inference and attention.
    slide_pred, attention = config.predict(model, bags, attention=use_attention, **kwargs)

    df_slides = []
    df_attention = []
    df_preds = []
    df_uq = []
    df_true = []
    df_loc_x = []
    df_loc_y = []

    # Then, generate tile predictions for each slide:
    for i, (bag, slide) in track(enumerate(zip(bags, slides)),
                            description="Generating tile predictions",
                            total=len(bags)):

        # Prepare bags, and resize bag dimension to the batch dimension.
        loaded_bags = torch.unsqueeze(utils._load_bag(bag, device=device), dim=1)

        # Split loaded bags into smaller batches for inference (tile_batch_size)
        if len(loaded_bags) > tile_batch_size:
            loaded_bags = torch.split(loaded_bags, tile_batch_size, dim=0)
        else:
            loaded_bags = [loaded_bags]

        _running_pred = []
        _running_uq = []

        # Run inference on each batch.
        for batch in loaded_bags:
            with torch.inference_mode():
                pred_out = config.batched_predict(model, batch, uq=uq, device=device, attention=True, **kwargs)

            if uq or len(pred_out) == 3:
                _pred, _att, _uq = utils._output_to_numpy(*pred_out)
                if _uq is not None and len(_uq):
                    _running_uq.append(_uq)
            else:
                _pred, _att = utils._output_to_numpy(*pred_out)
            _running_pred.append(_pred)

        # Concatenate predictions and attention.
        tile_pred = np.concatenate(_running_pred, axis=0)
        if len(_running_uq):
            tile_uq = np.concatenate(_running_uq, axis=0)

        # Verify the shapes are consistent.
        if attention is not None and len(attention):
            assert len(tile_pred) == attention[i].shape[-1]
        n_bags = len(tile_pred)

        # Find the associated locations.
        bag_index = join(dirname(bag), f'{slide}.index.npz')
        if exists(bag_index):
            locations = np.load(bag_index)['arr_0']
            assert len(locations) == n_bags
            df_loc_x.append(locations[:, 0])
            df_loc_y.append(locations[:, 1])

        # Add to dataframe lists.
        df_preds.append(tile_pred)
        if uq:
            df_uq.append(tile_uq)
        if attention is not None and len(attention):
            df_attention.append(attention[i])
        df_slides += [slide for _ in range(n_bags)]
        if outcomes is not None:
            _label = labels[slide]
            df_true += [_label for _ in range(n_bags)]

    # Update dataframe with predictions.
    df_dict = dict(slide=df_slides)
    if len(df_attention):
        df_attention = np.concatenate(df_attention, axis=-1)
    df_preds = np.concatenate(df_preds, axis=0)

    # Tile location
    if df_loc_x:
        df_dict['loc_x'] = np.concatenate(df_loc_x, axis=0)
        df_dict['loc_y'] = np.concatenate(df_loc_y, axis=0)

    # Attention
    if attention is not None and len(attention):
        if len(df_attention.shape) == 1:
            df_dict['attention'] = df_attention
        else:
            for _a in range(len(df_attention)):
                df_dict[f'attention-{_a}'] = df_attention[_a]

    # Uncertainty
    if uq:
        df_uq = np.concatenate(df_uq, axis=0)
        for i in range(df_uq[0].shape[0]):
            df_dict[f'uncertainty{i}'] = df_uq[:, i]

    # Ground truth
    if outcomes is not None:
        df_dict['y_true'] = df_true

    # Predictions
    for i in range(df_preds[0].shape[0]):
        df_dict[f'y_pred{i}'] = df_preds[:, i]

    # Final processing to dataframe & disk
    df = pd.DataFrame(df_dict)
    if dest is not None:
        df.to_parquet(dest)
        log.info("{} tile predictions exported to [green]{}[/]".format(
            df_preds.shape[0],
            dest
        ))
    return df


def save_mil_tile_predictions(
    weights: str,
    dataset: "Dataset",
    bags: Union[str, np.ndarray, List[str]],
    config: Optional[TrainerConfig] = None,
    outcomes: Union[str, List[str]] = None,
    dest: str = 'mil_tile_preds.parquet',
) -> pd.DataFrame:
    return get_mil_tile_predictions(
        weights,
        dataset,
        bags,
        config=config,
        outcomes=outcomes,
        dest=dest
    )

# -----------------------------------------------------------------------------
# Feature extraction and attention heatmaps.

def generate_mil_features(
    weights: str,
    dataset: "Dataset",
    bags: Union[str, np.ndarray, List[str]],
    *,
    config: Optional[TrainerConfig] = None,
) -> "MILFeatures":
    """Generate activations weights from the last layer of an MIL model.

    Returns MILFeatures object.

    Args:
        weights (str): Path to model weights to load.
        config (:class:`slideflow.mil.TrainerConfig`):
            Configuration for building model. If ``weights`` is a path to a
            model directory, will attempt to read ``mil_params.json`` from this
            location and load saved configuration. Defaults to None.
        dataset (:class:`slideflow.Dataset`): Dataset.
        outcomes (str, list(str)): Outcomes.
        bags (str, list(str)): Path to bags, or list of bag file paths.
            Each bag should contain PyTorch array of features from all tiles in
            a slide, with the shape ``(n_tiles, n_features)``.
    """
    from .features import MILFeatures

    # Load model weights.
    model, config = utils.load_model_weights(weights, config)

    # Ensure the model is valid for generating features.
    if not hasattr(model, 'get_last_layer_activations'):
        raise errors.ModelError(
            f"Model {model.__class__.__name__} is not supported; could not "
            "find method 'get_last_layer_activations'")

    # Prepare bags and targets.
    slides = dataset.slides()
    if isinstance(bags, str):
        bags = dataset.get_bags(bags)
    else:
        bags = np.array([b for b in bags if path_to_name(b) in slides])

    # Ensure slide names are sorted according to the bags.
    slides = [path_to_name(b) for b in bags]

    # Calculate and return last-layer features.
    return MILFeatures(model, bags, slides=slides, config=config, dataset=dataset)

