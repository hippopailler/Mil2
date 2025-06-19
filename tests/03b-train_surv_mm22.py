import os
import slideflow as sf
from slideflow.mil import mil_config
from slideflow.util import prepare_multimodal_mixed_bags

import multiprocessing
import argparse

def train_val(args):
    P = sf.Project(os.getcwd(), create=True)
    P.annotations = 'tests/annotations/ann_mm_surv.csv'

    extractor = '.'
    
    epoch = 2
    bag_size = 4
    batch_size = 4

    # Chemins des données
    bags_dir ='bags'
    df_path = 'tests/features/fake_mm_surv/df.parquet'    

    # Lecture des features depuis le parquet
    #df_features = pd.read_parquet('tests/features/fake_mm_surv/df.parquet')
    
    prepare_multimodal_mixed_bags(
    path=df_path,        # Chemin vers le fichier de features
    bags_path=bags_dir   # Dossier où sauvegarder les bags
    )
    
    # training and validating on the same dataset
    train_dataset = P.dataset(tile_px=256, tile_um=256, filters={'dataset': 'train'})
    val_dataset = P.dataset(tile_px=256, tile_um=256, filters={'dataset': 'val'})



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

        P.train_mil(
            config=config,
            exp_label=f'fake_mm-{model}',
            outcomes='os',
            events='death',
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            bags=f'tests/features/fake_mm_surv/bags'
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