"""Module minimal pour le Dataset MIL."""

import pandas as pd
import os
import numpy as np
import MIL.errors as errors
from glob import glob
from os.path import join, exists
from typing import Optional, Dict, Union, List, Tuple
from MIL.util import log, path_to_name, Labels, as_list, EMPTY, load_json, tile_size_label

class Dataset:
    """Version minimale de Dataset pour MIL."""

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Dict[str, str]]]] = None,
        sources: Optional[Union[str, List[str]]] = None,
        tile_px: Optional[int] = None,
        tile_um: Optional[Union[int, str]] = None,
        *,
        filters: Optional[Dict] = None,
        filter_blank: Optional[Union[List[str], str]] = None,
        annotations: Optional[Union[str, pd.DataFrame]] = None,
        min_tiles: int = 0,
    ) -> None:
        """Initialise un Dataset pour MIL."""
        self.tile_px = tile_px
        self.tile_um = tile_um
        self._filters = filters if filters else {}
        self._min_tiles = min_tiles
        self._clip = {}
        self.prob_weights = None
        self._annotations = None
        self.annotations_file = None

        # Configuration par défaut comme dans Slideflow
        default_config = {
            'source1': {
                'path': 'tests/features/fake_mm_surv',
                'tfrecords': 'tests/features/fake_mm_surv/bags',
                'label': None
            }
        }

        # Utiliser la config fournie ou la config par défaut
        if isinstance(config, str):
            self.config = config
            loaded_config = load_json(config)
        elif config is None:
            loaded_config = default_config
            self._config = "<default>"
        else:
            self._config = "<dict>"
            loaded_config = config

        # Gestion des sources comme dans Slideflow
        if sources is None:
            sources = ['source1']  # Source par défaut
        sources = sources if isinstance(sources, list) else [sources]

        # Configuration des sources
        self.sources = {
            k: v for k, v in loaded_config.items() if k in sources
        }
        self.sources_names = list(self.sources.keys())

        # Ne pas afficher d'avertissement pour la source par défaut
        missing_sources = [s for s in sources if s not in self.sources and s != 'source1']
        if len(missing_sources):
            log.warn(
                "The following sources were not found in the dataset "
                f"configuration: {', '.join(missing_sources)}"
            )

        # Chargement des annotations
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
    
    @property 
    def min_tiles(self) -> int: 
        """Returns the active min_tiles filter, if any (defaults to 0).""" 
        return self._min_tiles 

    def slides(self) -> List[str]:
        """Liste des slides après filtrage."""
        if self.filtered_annotations is not None:
            return self.filtered_annotations['slide'].tolist()
        return []

    def is_float(self, header: str) -> bool: 

        """Check if labels in the given header can all be converted to float. 
        Args: 
            header (str): Annotations column header. 
        Returns: 
            bool: If all values from header can be converted to float. 

        """ 
        if self.annotations is None: 
            raise errors.DatasetError("Annotations not loaded.") 
        filtered_labels = self.filtered_annotations[header] 
        try: 
            filtered_labels = [float(o) for o in filtered_labels] 
            return True 
        except ValueError: 
            return False 
        
    def labels( 
        self, 
        headers: Union[str, List[str]], 
        use_float: Union[bool, Dict, str] = False, 
        assign: Optional[Dict[str, Dict[str, int]]] = None, 
        format: str = 'index' 

    ) -> Tuple[Labels, Union[Dict[str, Union[List[str], List[float]]], 
                             List[str], 
                             List[float]]]: 
        """Return a dict of slide names mapped to patient id and label(s). 
        Args: 
            headers (list(str)) Annotation header(s) that specifies label. 
                May be a list or string. 
            use_float (bool, optional) Either bool, dict, or 'auto'. 
                If true, convert data into float; if unable, raise TypeError. 
                If false, interpret all data as categorical. 
                If a dict(bool), look up each header to determine type. 
                If 'auto', will try to convert all data into float. For each 
                header in which this fails, will interpret as categorical. 
            assign (dict, optional):  Dictionary mapping label ids to 
                label names. If not provided, will map ids to names by sorting 
                alphabetically. 
            format (str, optional): Either 'index' or 'name.' Indicates which 
                format should be used for categorical outcomes when returning 
                the label dictionary. If 'name', uses the string label name. 
                If 'index', returns an int (index corresponding with the 
                returned list of unique outcomes as str). Defaults to 'index'. 

        Returns: 
            A tuple containing 

                **dict**: Dictionary mapping slides to outcome labels in 
                numerical format (float for continuous outcomes, int of outcome 
                label id for categorical outcomes). 

                **list**: List of unique labels. For categorical outcomes, 
                this will be a list of str; indices correspond with the outcome 
                label id. 

        """ 

        if self.annotations is None: 
            raise errors.DatasetError("Annotations not loaded.") 

        if not len(self.filtered_annotations): 
            raise errors.DatasetError( 
                "Cannot generate labels: dataset is empty after filtering." 
            ) 

        results = {}  # type: Dict 
        headers = as_list(headers) 
        unique_labels = {} 
        filtered_pts = self.filtered_annotations.patient 
        filtered_slides = self.filtered_annotations.slide 
        for header in headers: 
            if assign and (len(headers) > 1 or header in assign): 
                assigned_for_header = assign[header] 

            elif assign is not None: 
                raise errors.DatasetError( 
                    f"Unable to read outcome assignments for header {header}" 
                    f" (assign={assign})" 
                ) 
            else: 
                assigned_for_header = None 
            unique_labels_for_this_header = [] 
            try: 
                filtered_labels = self.filtered_annotations[header] 
            except KeyError: 
                raise errors.AnnotationsError(f"Missing column {header}.") 

            # Determine whether values should be converted into float 
            if isinstance(use_float, dict) and header not in use_float: 
                raise ValueError( 
                    f"use_float is dict, but header {header} is missing." 
                ) 

            elif isinstance(use_float, dict): 
                header_is_float = use_float[header] 
            elif isinstance(use_float, bool): 
                header_is_float = use_float 
            elif use_float == 'auto': 
                header_is_float = self.is_float(header) 
            else: 
                raise ValueError(f"Invalid use_float option {use_float}") 

            # Ensure labels can be converted to desired type, 
            # then assign values 
            if header_is_float and not self.is_float(header): 
                raise TypeError( 
                    f"Unable to convert all labels of {header} into 'float' " 
                    f"({','.join(filtered_labels)})." 
                ) 

            elif header_is_float: 
                log.debug(f'Interpreting column "{header}" as continuous') 
                filtered_labels = filtered_labels.astype(float) 

            else: 
                log.debug(f'Interpreting column "{header}" as categorical') 
                unique_labels_for_this_header = list(set(filtered_labels)) 
                unique_labels_for_this_header.sort() 
                for i, ul in enumerate(unique_labels_for_this_header): 
                    n_matching_filtered = sum(f == ul for f in filtered_labels) 
                    if assigned_for_header and ul not in assigned_for_header: 
                        raise KeyError( 
                            f"assign was provided, but label {ul} missing" 
                        ) 

                    elif assigned_for_header: 
                        val_msg = assigned_for_header[ul] 
                        n_s = str(n_matching_filtered) 
                        log.debug( 
                            f"{header} {ul} assigned {val_msg} [{n_s} slides]" 
                        ) 
                    else: 
                        n_s = str(n_matching_filtered) 
                        log.debug( 
                            f"{header} {ul} assigned {i} [{n_s} slides]" 
                        ) 

            def _process_cat_label(o): 
                if assigned_for_header: 
                    return assigned_for_header[o] 
                elif format == 'name': 
                    return o 
                else: 
                    return unique_labels_for_this_header.index(o) 

            # Check for multiple, different labels per patient and warn 
            pt_assign = np.array(list(set(zip(filtered_pts, filtered_labels)))) 
            unique_pt, counts = np.unique(pt_assign[:, 0], return_counts=True) 
            for pt in unique_pt[np.argwhere(counts > 1)][:, 0]: 
                dup_vals = pt_assign[pt_assign[:, 0] == pt][:, 1] 
                dups = ", ".join([str(d) for d in dup_vals]) 
                log.error( 
                    f'Multiple labels for patient "{pt}" (header {header}): ' 
                    f'{dups}' 
                ) 
            # Assemble results dictionary
            for slide, lbl in zip(filtered_slides, filtered_labels): 
                if slide in EMPTY: 
                    continue 
                if not header_is_float: 
                    lbl = _process_cat_label(lbl) 
                if slide in results: 
                    results[slide] = as_list(results[slide]) 
                    results[slide] += [lbl] 
                elif header_is_float: 
                    results[slide] = [lbl] 
                else: 
                    results[slide] = lbl 
            unique_labels[header] = unique_labels_for_this_header 
        if len(headers) == 1: 
            return results, unique_labels[headers[0]] 
        else: 
            return results, unique_labels 
        
    def get_bags(self, path, warn_missing=True): 

        """Return list of all \*.pt files with slide names in this dataset. 

        May return more than one \*.pt file for each slide. 

        Args: 
            path (str, list(str)): Directory(ies) to search for \*.pt files. 
            warn_missing (bool): Raise a warning if any slides in this dataset 
                do not have a \*.pt file. 
        """ 
        slides = self.slides() 
        if isinstance(path, str): 
            path = [path] 
        bags = [] 
        for p in path: 
            if not exists(p): 
                raise ValueError(f"Path {p} does not exist.") 
            bags_at_path = np.array([ 
                join(p, f) for f in os.listdir(p) 
                if f.endswith('.pt') and path_to_name(f) in slides 
            ]) 
            bags.append(bags_at_path) 
        bags = np.concatenate(bags) 
        unique_slides_with_bags = np.unique([path_to_name(b) for b in bags]) 
        if (len(unique_slides_with_bags) != len(slides)) and warn_missing: 
            log.warning(f"Bags missing for {len(slides) - len(unique_slides_with_bags)} slides.") 
        return bags 
    
 
    def slides(self) -> List[str]: 
        """Return a list of slide names in this dataset.""" 
        if self.annotations is None: 
            raise errors.AnnotationsError( 
                "No annotations loaded; is the annotations file empty?" 
            ) 
        if 'slide' not in self.annotations.columns: 
            raise errors.AnnotationsError( 
                f"{'slide'} not found in annotations file." 
            ) 
        ann = self.filtered_annotations 
        ann = ann.loc[~ann.slide.isin(EMPTY)] 
        slides = ann.slide.unique().tolist() 
        return slides  
    
    def tfrecords(self, source: Optional[str] = None) -> List[str]: 

        """Return a list of all tfrecords. 
        Args: 
            source (str, optional): Only return tfrecords from this dataset 
                source. Defaults to None (return all tfrecords in dataset). 
        Returns: 
            List of tfrecords paths. 
        """ 
        if source and source not in self.sources.keys(): 
            log.error(f"Dataset {source} not found.") 
            return [] 

        if source is None: 
            sources_to_search = list(self.sources.keys())  # type: List[str] 
        else: 
            sources_to_search = [source] 

        tfrecords_list = [] 
        folders_to_search = [] 

        for source in sources_to_search: 
            if not self._tfrecords_set(source): 
                log.warning(f"tfrecords path not set for source {source}") 
                continue 
            tfrecords = self.sources[source]['tfrecords'] 
            label = self.sources[source]['label'] 

            if label is None: 
                continue 
            tfrecord_path = join(tfrecords, label) 
            if not exists(tfrecord_path): 
                log.debug( 
                    f"TFRecords path not found: {tfrecord_path}" 
                ) 
                continue 
            folders_to_search += [tfrecord_path] 

        for folder in folders_to_search: 
            tfrecords_list += glob(join(folder, "*.tfrecords")) 
        tfrecords_list = list(set(tfrecords_list)) 
        # Filter the list by filters 
        if self.annotations is not None: 
            slides = self.slides() 
            filtered_tfrecords_list = [ 
                tfrecord for tfrecord in tfrecords_list 
                if path_to_name(tfrecord) in slides 
            ] 
            filtered = filtered_tfrecords_list 
        else: 
            log.warning("Error filtering TFRecords, are annotations empty?") 
            filtered = tfrecords_list 
        # Filter by min_tiles 
        manifest = self.manifest(filter=False) 
        if not all([f in manifest for f in filtered]): 
            self.update_manifest() 
            manifest = self.manifest(filter=False) 
        if self.min_tiles: 
            return [ 
                f for f in filtered 
                if f in manifest and manifest[f]['total'] >= self.min_tiles 
            ] 
        else: 
            return [f for f in filtered 
                    if f in manifest and manifest[f]['total'] > 0] 
        
    def tfrecords_folders(self) -> List[str]: 
        """Return folders containing tfrecords.""" 
        folders = [] 
        for source in self.sources: 
            if self.sources[source]['label'] is None: 
                continue 
            if not self._tfrecords_set(source): 
                log.warning(f"tfrecords path not set for source {source}") 
                continue 
            folders += [join( 
                self.sources[source]['tfrecords'], 
                self.sources[source]['label'] 
            )] 
        return folders 
    
    def update_manifest(self, force_update: bool = False) -> None: 
        """Update tfrecord manifests. 
        Args: 
            forced_update (bool, optional): Force regeneration of the 
                manifests from scratch. 

        """ 
        tfrecords_folders = self.tfrecords_folders()    

    def verify_annotations_slides(self) -> None: 

        """Verify that annotations are correctly loaded.""" 

        if self.annotations is None: 
            log.warn("Annotations not loaded.") 
            return 

        # Verify no duplicate slide names are found 
        ann = self.annotations.loc[self.annotations.slide.isin(self.slides())] 
        if not ann.slide.is_unique: 
            raise errors.AnnotationsError( 
                "Duplicate slide names detected in the annotation file." 
            ) 

        # Verify that there are no tfrecords with the same name. 
        # This is a problem because the tfrecord name is used to 
        # identify the slide. 
        tfrecords = self.tfrecords() 
        if len(tfrecords): 
            tfrecord_names = [path_to_name(tfr) for tfr in tfrecords] 
            if not len(set(tfrecord_names)) == len(tfrecord_names): 
                duplicate_tfrs = [ 
                    tfr for tfr in tfrecords 
                    if tfrecord_names.count(path_to_name(tfr)) > 1 
                ] 
                raise errors.AnnotationsError( 
                    "Multiple TFRecords with the same names detected: {}".format( 
                        ', '.join(duplicate_tfrs) 
                    ) 
                ) 
        # Verify all slides in the annotation column are valid 

        n_missing = len(self.annotations.loc[ 
            (self.annotations.slide.isin(['', ' ']) 
             | self.annotations.slide.isna()) 
        ]) 
        if n_missing == 1: 
            log.warn("1 patient does not have a slide assigned.") 
        if n_missing > 1: 
            log.warn(f"{n_missing} patients do not have a slide assigned.") 

    def manifest(self, 
        key: str = 'path', 
        filter: bool = True 
    ) -> Dict[str, Dict[str, int]]: 
        """Generate a manifest of all tfrecords. 

        Args: 
            key (str): Either 'path' (default) or 'name'. Determines key format 
                in the manifest dictionary. 
            filter (bool): Apply active filters to manifest. 

        Returns: 
            dict: Dict mapping key (path or slide name) to number of tiles. 
        """ 
        if key not in ('path', 'name'): 
            raise ValueError("'key' must be in ['path, 'name']") 

        all_manifest = {} 
        
        if filter:
            filtered_tfrecords = self.tfrecords() 
            manifest_tfrecords = list(all_manifest.keys()) 
            for tfr in manifest_tfrecords: 
                if tfr not in filtered_tfrecords: 
                    del all_manifest[tfr] 
        # Log clipped tile totals if applicable 
        for tfr in all_manifest: 
            if tfr in self._clip: 
                all_manifest[tfr]['clipped'] = min(self._clip[tfr], 
                                                   all_manifest[tfr]['total']) 

            else: 
                all_manifest[tfr]['clipped'] = all_manifest[tfr]['total'] 
        if key == 'path': 
            return all_manifest 
        else: 
            return {path_to_name(t): v for t, v in all_manifest.items()}