import os
import sys
import requests
import traceback
import copy
from train.prepare_dataset import prepare_dataset
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import yaml
from train.run_training import run_training
from uploading.upload_results import (
    handle_hyperparam_final_model_upload,
    handle_cv_upload,
    handle_cv_stratified_upload,
    handle_standard_upload
)
from utils.pipeline_utils import plot_comparative_results, build_excel_path


def pipeline(base_path, excel_path, config, mods):
    print(f"\n=== Training: outcome={config['task_settings']['outcome']}, mods={mods}, subanalysis={config.get('current_filter', 'none')} ===")
    training_return = run_training(config, mods)
    print("Training completed.")

    ttype = config.get("training_type")
    if ttype == "hyperparameter_tuning":
        handle_hyperparam_final_model_upload(training_return, config, excel_path, mods)
    elif ttype == "cross_validation":
        handle_cv_upload(training_return["experiment_paths"], config, excel_path, mods=mods)
    elif ttype == "cross_validation_stratified":
        handle_cv_stratified_upload(training_return["experiment_paths"], config, excel_path, neptune_id=None)
    elif ttype == "standard":
        handle_standard_upload(training_return["experiment_paths"], config, excel_path, mods=mods)
    else:
        raise ValueError(f"Unknown training type: {ttype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--excel", required=True)
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    original_config = copy.deepcopy(config)

    try:
        # Dataset preparation if requested
        if config.get("prepare_dataset", False):
            for mods in config.get('mods', []):
                # Choose dataset based on radpy/radfm flags
                if mods.get("radpy", False):
                    train_data = "data/features_dataset_radpy_fixed.parquet"
                elif mods.get("radfm", False):
                    train_data = "data/features_dataset_radfm.parquet"
                else:
                    train_data = config.get("train_df")

                mod_string = "_".join([k for k, v in mods.items() if v])
                bag_path = f"bags_{mod_string}"

                prepare_dataset(train_data, config.get("annotation_file"), mods, bag_path)

            print("dataset prepared")

        # Main processing: iterate over outcomes
        for outcome in original_config['task_settings'].get('outcomes', []):
            print(f"\n🔄 Processing outcome: {outcome}")
            # Reset full config for each outcome
            config = copy.deepcopy(original_config)
            config['task_settings']['outcome'] = outcome

            for mods in config.get('mods', []):
                print(f"⚙️  Mods: {mods}")
                excel_path = build_excel_path(args.excel, config, mods)
                print(f"📁 Excel path: {excel_path}")
                pipeline(args.base_dir, excel_path, config, mods)

        # Notify success via hook
        hook_file = 'req_hook.txt'
        if os.path.exists(hook_file):
            url = open(hook_file).read().strip()
        else:
            url = 'https://ntfy.sh/invalid_hook'
        requests.post(url, data={'value1': 'task completed'})

    except Exception:
        # Notify exception via hook
        hook_file = 'req_hook.txt'
        if os.path.exists(hook_file):
            url = open(hook_file).read().strip()
        else:
            url = 'https://ntfy.sh/invalid_hook'
        exc_info = traceback.format_exc()
        requests.post(url, data={'value1': 'exception', 'value2': exc_info})
        raise

    # Plot comparative results
    plot_comparative_results(excel_path, config)

