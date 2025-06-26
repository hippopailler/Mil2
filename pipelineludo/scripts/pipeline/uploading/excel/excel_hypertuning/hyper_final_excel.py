import os
import pandas as pd

def export_final_model_result(modality_path, config, excel_path, neptune_id):
    """
    Export the final model's performance (after hyperparameter tuning)
    to a global comparison Excel file.

    The function expects a file called 'final_model_scores.csv' in the modality_path,
    and adds a row to the Excel file to compare between modalities and configurations.
    If the file does not exist, it will be created with placeholder values.
    """
    score_file = os.path.join(modality_path, "final_model_scores.csv")

    # If the file doesn't exist, create it with dummy values
    if not os.path.exists(score_file):
        print(f"⚠️ final_model_scores.csv not found in {modality_path}, creating placeholder...")
        dummy_metrics = {"AUC": None, "F1": None, "Sensitivity": None, "Specificity": None}
        pd.DataFrame([dummy_metrics]).to_csv(score_file, index=False)

    try:
        scores = pd.read_csv(score_file)
        if scores.shape[0] == 0:
            print("⚠️ Score file is empty")
            return
        metrics = scores.iloc[0].to_dict()

        row_data = {
            "OUTCOME": config.get("task"),
            "MODALITY": "_".join(config.get("modalities", [])),
            "SOURCE": config.get("source"),
            "IMP INDICATOR": config.get("imp"),
            "NEPTUNE-ID": neptune_id,
            **metrics
        }

        # Write or append to Excel
        if os.path.exists(excel_path):
            existing_df = pd.read_excel(excel_path, engine="openpyxl")
            df = pd.concat([existing_df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            df = pd.DataFrame([row_data])

        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"✅ Final model result exported to: {excel_path}")

    except Exception as e:
        print(f"❌ Error exporting final model result: {e}")