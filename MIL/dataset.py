"""Module minimal pour le Dataset MIL."""

import os
import pandas as pd
from typing import Optional, Dict, Union, List

class Dataset:
    """Version minimale de Dataset pour MIL."""

    def __init__(
        self,
        tile_px: Optional[int] = None,
        tile_um: Optional[Union[int, str]] = None,
        *,
        filters: Optional[Dict] = None,
        annotations: Optional[Union[str, pd.DataFrame]] = None,
    ) -> None:
        """Initialise un Dataset pour MIL.
        
        Args:
            tile_px: Taille des tuiles en pixels
            tile_um: Taille des tuiles en microns ou magnification (ex: "20x")
            filters: Filtres pour sélectionner les slides
            annotations: Fichier d'annotations CSV ou DataFrame
        """
        self.tile_px = tile_px
        self.tile_um = tile_um
        self._filters = filters if filters else {}
        self._annotations = None
        
        # Chargement des annotations si fournies
        if annotations is not None:
            self.load_annotations(annotations)

    def load_annotations(self, annotations: Union[str, pd.DataFrame]) -> None:
        """Charge les annotations.
        
        Args:
            annotations: Chemin vers CSV ou DataFrame
        """
        if isinstance(annotations, str):
            self._annotations = pd.read_csv(annotations)
        else:
            self._annotations = annotations.copy()

        # Vérification des colonnes requises
        if 'slide' not in self._annotations.columns:
            raise ValueError("La colonne 'slide' est manquante dans les annotations")

    @property
    def annotations(self) -> Optional[pd.DataFrame]:
        """Annotations du dataset."""
        return self._annotations

    @property
    def filtered_annotations(self) -> pd.DataFrame:
        """Annotations après application des filtres."""
        if self.annotations is not None:
            filtered_df = self.annotations.copy()
            # Application des filtres
            for filter_key, filter_vals in self.filters.items():
                if not isinstance(filter_vals, list):
                    filter_vals = [filter_vals]
                filtered_df = filtered_df[filtered_df[filter_key].isin(filter_vals)]
            return filtered_df
        return pd.DataFrame()

    @property
    def filters(self) -> Dict:
        """Filtres actifs du dataset."""
        return self._filters

    def slides(self) -> List[str]:
        """Liste des slides après filtrage."""
        if self.filtered_annotations is not None:
            return self.filtered_annotations['slide'].tolist()
        return []

