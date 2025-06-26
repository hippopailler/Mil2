import pandas as pd
import json
import shutil
import os
import slideflow as sf

def prepare_dataset(train_data: str, annotation_file: str, mods: dict, bag_path: str):
    """
    Prepares the dataset by:
    - Dropping unused modality columns
    - Setting up the annotation file path
    - Creating the Slideflow project and bags
    """

    # Identify which modalities are *not* active
    excluded_mods = [key for key in mods.keys() if not mods[key]]
    active_mods = [key for key in mods.keys() if mods[key]]

    print(f"\n[INFO] Preparing dataset with modalities: {active_mods}")

    if not excluded_mods:
        # If all modalities are used, copy the full dataset
        print("[INFO] Using all modalities.")
        shutil.copyfile(train_data, 'df.parquet')
    else:
        # Map each modality name to its column name in the dataframe
        mod_mapping = {
            'rwd': 'mod1', 
            'dp': 'mod3',
            'genomics': 'mod4',
        }

        # Add the correct radiomics key
        if 'radpy' in mods:
            mod_mapping['radpy'] = 'mod2'
        if 'radfm' in mods:
            mod_mapping['radfm'] = 'mod2'  # same column name, different source


        df = pd.read_parquet(train_data)
        
        # Drop columns of unused modalities (if they exist in the mapping)
        df = df.drop(columns=[mod_mapping[mod] for mod in excluded_mods if mod in mod_mapping], errors='ignore')

        # Ensure all missing values are properly handled
        df = df.applymap(lambda x: None if isinstance(x, float) and pd.isna(x) else x)
        df.dropna()  # This doesn't modify df in place

        df.to_parquet('df.parquet', index=False)

    # Update settings to point to the correct annotation file
    with open('orig_settings.json', 'r') as f:
        settings = json.load(f)

    settings['annotations'] = annotation_file

    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

    # Create project + multimodal bags
    P = sf.Project(os.getcwd(), create=True)
    sf.prepare_multimodal_mixed_bags('df.parquet', bag_path)
