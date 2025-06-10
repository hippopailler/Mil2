"""Module minimal pour le MIL, extrait de la classe Project de Slideflow."""

import os
from typing import Optional, Dict, Union, List
import pandas as pd
from .. import Dataset
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)
from util import log, exists

class Project:
    """Version minimaliste de Project pour le MIL."""

    def __init__(
        self, 
        root: str,
        create: bool = False
    ) -> None:
        """Initialise un projet minimal pour MIL.
        
        Args:
            root (str): Chemin du projet
            create (bool): Créer le projet si n'existe pas
        """
        self.root = root
        self._settings = {
            'name': 'MyProject',
            'annotations': './annotations.csv',
            'dataset_config': './datasets.json',
            'models_dir': './models',
            'eval_dir': './eval',
            'sources': ['source1']
        }
        
        # Créer les dossiers nécessaires
        if create:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.eval_dir, exist_ok=True)

    @property
    def annotations(self) -> str:
        return os.path.join(self.root, self._settings['annotations'])

    @annotations.setter 
    def annotations(self, val: str) -> None:
        self._settings['annotations'] = val

    @property
    def dataset_config(self) -> str:
        return os.path.join(self.root, self._settings['dataset_config'])

    @property
    def models_dir(self) -> str:
        return os.path.join(self.root, self._settings['models_dir'])

    @property
    def eval_dir(self) -> str:
        return os.path.join(self.root, self._settings['eval_dir'])

    def dataset(
            self,
            tile_px: Optional[int] = None,
            tile_um: Optional[Union[int, str]] = None,
            *,
            verification: Optional[str] = 'both',
            **kwargs: Any
        ) -> Dataset:
            """Return a :class:`slideflow.Dataset` object using project settings.

            Args:
                tile_px (int): Tile size in pixels
                tile_um (int or str): Tile size in microns (int) or magnification
                    (str, e.g. "20x").

            Keyword Args:
                filters (dict, optional): Dataset filters to use for
                    selecting slides. See :meth:`slideflow.Dataset.filter` for
                    more information. Defaults to None.
                filter_blank (list(str) or str, optional): Skip slides that have
                    blank values in these patient annotation columns.
                    Defaults to None.
                min_tiles (int, optional): Min tiles a slide must have.
                    Defaults to 0.
                config (str, optional): Path to dataset configuration JSON file.
                    Defaults to project default.
                sources (str, list(str), optional): Dataset sources to use from
                    configuration. Defaults to project default.
                verification (str, optional): 'tfrecords', 'slides', or 'both'.
                    If 'slides', verify all annotations are mapped to slides.
                    If 'tfrecords', check that TFRecords exist and update manifest.
                    Defaults to 'both'.

            """
            if 'config' not in kwargs:
                kwargs['config'] = self.dataset_config
            if 'sources' not in kwargs:
                kwargs['sources'] = self.sources
            try:
                if self.annotations and exists(self.annotations):
                    annotations = self.annotations
                else:
                    annotations = None
                dataset = Dataset(
                    tile_px=tile_px,
                    tile_um=tile_um,
                    annotations=annotations,
                    **kwargs
                )
            except FileNotFoundError:
                raise errors.DatasetError('No datasets configured.')
            if verification in ('both', 'slides'):
                log.debug("Verifying slide annotations...")
                dataset.verify_annotations_slides()
            if verification in ('both', 'tfrecords'):
                log.debug("Verifying tfrecords...")
                dataset.update_manifest()
            return dataset

    def train_mil(
        self,
        config: "mil.TrainerConfig",
        train_dataset: Dataset,
        val_dataset: Dataset, 
        outcomes: Union[str, List[str]],
        bags: Union[str, List[str]],
        exp_label: Optional[str] = None,
        outdir: Optional[str] = None,
        **kwargs
    ):
        """Entraîne un modèle MIL.
        
        Args:
            config: Configuration du modèle MIL
            train_dataset: Dataset d'entraînement
            val_dataset: Dataset de validation
            outcomes: Labels à prédire
            bags: Chemin vers les bags ou liste de chemins
            exp_label: Label de l'expérience
            outdir: Dossier de sortie
            **kwargs: Arguments additionnels
        """
        from slideflow.mil import train_mil

        # Configuration du dossier de sortie
        if outdir is None:
            model_name = f"mil-{exp_label}" if exp_label else "mil"
            outdir = os.path.join(self.models_dir, model_name)
            os.makedirs(outdir, exist_ok=True)

        # Entraînement
        return train_mil(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            outcomes=outcomes,
            bags=bags,
            outdir=outdir,
            **kwargs
        )