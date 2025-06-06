import os
import traceback
import requests
import slideflow as sf
from slideflow.mil import mil_config
import multiprocessing
import pandas as pd
import argparse
import torch

def train_val(args):
    P = sf.Project(os.getcwd())
    P.annotations = 'annotations/ann_mm_surv.csv'

    # Vérification des annotations
    ann_df = pd.read_csv('annotations/ann_mm_surv.csv')
    print(f"Nombre de slides dans les annotations: {len(ann_df)}")
    print("Distribution des datasets:")
    print(ann_df['dataset'].value_counts())

    # Configuration des datasets
    train_slides = ann_df[ann_df['dataset'] == 'train']['slide'].tolist()
    val_slides = ann_df[ann_df['dataset'] == 'val']['slide'].tolist()
    
    print(f"Nombre de slides train: {len(train_slides)}")
    print(f"Nombre de slides val: {len(val_slides)}")
    
    bags_dir = 'fake_mm_surv/bags'
    extractor = '.'
    
    epoch = 2
    bag_size = 4
    batch_size = 4

    # Création des datasets avec mapping explicite
    train_dataset = P.dataset(
        tile_px=256, 
        tile_um=256, 
        filters={'dataset': 'train'}
    )
    print(f"Slides d'entraînement: {train_dataset.slides}")

    val_dataset = P.dataset(
        tile_px=256, 
        tile_um=256, 
        filters={'dataset': 'val'}
    )
    print(f"Slides de validation: {val_dataset.slides}")

    if True:
        model = 'mb_attention_mil'
        config = mil_config(
            'mb_attention_mil',
            # bag_size=bag_size,
            epochs=epoch,
            batch_size=batch_size,
            # aggregation_level='slide',
            loss='mm_survival_loss',
            save_monitor='c_index',
            # reconstruction_weight=0.02,
            # model_kwargs={'n_layers': 2},
            lr=0.01
        )
        config.mixed_bags = True
        # Ajout des informations de mapping au config
        # Mapping explicite des slides aux indices
        slide_to_idx = {str(i+1): i for i in range(300)}
        config.slide_mapping = slide_to_idx
        
        P.train_mil(
            config=config,
            exp_label=f'fake_mm-{model}',
            outcomes='os',
            events='death',
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            bags=bags_dir
        )

if __name__=='__main__':
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='Train MIL models with different configurations')
    parser.add_argument('--mixed', action='store_true', help='Train mixed multimodal MIL model')
    args = parser.parse_args()

    url = 'https://ntfy.sh/python_hook'
    try:
        train_val(args)
        # requests.post(url, data={'value1': 'task completed', 'value2': traceback.format_exc()})
    except: 
        # requests.post(url, data={'value1': 'exception', 'value2': traceback.format_exc()})
        raise