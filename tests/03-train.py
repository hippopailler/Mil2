import os
import traceback
import requests
import slideflow as sf
from slideflow.mil import mil_config
import multiprocessing
import pandas as pd
import argparse

def train_val(args):
    P = sf.Project(os.getcwd(), create=True)
    P.annotations = 'annotations/ann_mm_surv.csv'

    extractor = '.'
    
    epoch = 2
    bag_size = 4
    batch_size = 4

    # training and validating on the same dataset
    train_dataset = P.dataset(tile_px=256, tile_um=256, filters={'dataset': 'train'})
    val_dataset = P.dataset(tile_px=256, tile_um=256, filters={'dataset': 'val'})

    if True:
        #model = 'mb_attention_mil'
        model = 'mm_attention_mil'
        config = mil_config(
            model,
            # bag_size=bag_size,
            epochs=epoch,
            batch_size=batch_size,
            aggregation_level='slide',
            loss='mm_loss',
            lr=0.1
        )
        config.mixed_bags = True

        # Créer une liste de chemins pour chaque modalité
        bags_paths = [
            'features/fake_mm_surv/df.parquet#mod1',
            'features/fake_mm_surv/df.parquet#mod2',
            'features/fake_mm_surv/df.parquet#mod3'
        ]

        P.train_mil(
            config=config,
            exp_label=f'label',
            outcomes='adsq',
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            #bags=f'features/fake_mm/bags'
            bags=bags_paths
        )
    else:
        model = 'attention_mil' 
        config = mil_config(
            model,
            bag_size=bag_size,
            epochs=epoch,
            batch_size=batch_size,
            aggregation_level='slide',
        )
        P.train_mil(
            config=config,
            exp_label=f'fake_mm-{model}',
            outcomes='adsq',
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            bags=f'features/reinhard/uni/torch'
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