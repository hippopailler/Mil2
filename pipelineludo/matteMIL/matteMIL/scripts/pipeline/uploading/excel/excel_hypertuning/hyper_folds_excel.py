import os
import pandas as pd
import json

def export_hyperparam_results(modality_path, config, excel_path, neptune_id):
    """
    Export all hyperparameter trial results to Excel.
    Each subfolder in modality_path is assumed to be a trial with scores and its config.
    """
    print(f"🔍 Scanning for hyperparameter trials in: {modality_path}")

    all_rows = []

    for trial_folder in os.listdir(modality_path):
        trial_path = os.path.join(modality_path, trial_folder)
        if not os.path.isdir(trial_path):
            continue

        # Look for scores file
        score_file = None
        for fname in ["scores.csv", "scores_test.csv"]:
            path = os.path.join(trial_path, fname)
            if os.path.exists(path):
                score_file = path
                break

        if not score_file:
            continue

        try:
            scores = pd.read_csv(score_file)
            if scores.shape[0] == 0:
                continue

            metrics = scores.iloc[0].to_dict()

            # Try to load trial-specific hyperparams
            param_file = os.path.join(trial_path, "mil_params.json")
            hyperparam_str = ""
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    params = json.load(f)
                    hyperparam_str = ", ".join([f"{k}={v}" for k, v in params.items()])

            row_data = {
                "OUTCOME": config.get("task"),
                "MODALITY": "_".join(config.get("modalities", [])),
                "NEPTUNE-ID": neptune_id,
                "IMP INDICATOR": config.get("imp"),
                "HYPERPARAM COMBO": hyperparam_str,
                **metrics
            }
            all_rows.append(row_data)
        except Exception as e:
            print(f"⚠️ Error reading trial in {trial_folder}: {e}")

    if not all_rows:
        print("❌ No valid trials found.")
        return

    df = pd.DataFrame(all_rows)

    # Write or append to Excel
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path, engine="openpyxl")
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"✅ Exported {len(all_rows)} trial results to Excel: {excel_path}")
