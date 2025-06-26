def export_scores_to_excel(
    score_path,
    config,
    excel_path,
    neptune_id=None,
    cohort=None,
    hyperparam_combo=None,
    seed=None,
    mods=None,
    single_model=False
):
    """
    Exports a row of scores to Excel:
    - Supports both standard and averaged (mean/std) scores
    - For single models, combines train/test scores
    - Optionally includes: cohort info, neptune_id, hyperparameter combo, seed
    """
    import os
    import pandas as pd

    def fmt(m, s=None):
        if pd.isnull(m):
            return None
        return f"{m:.4f} ± {s:.4f}" if s is not None and pd.notnull(s) else f"{m:.4f}"

    active_mods = [k for k, v in mods.items() if v]
    print(f"Active modalities: {active_mods}")

    row = {
        "OUTCOME": config.get("task_settings", {}).get("outcome"),
        "MODALITY": "_".join(active_mods),
        "SOURCE": config.get("source"),
        "IMP INDICATOR": config.get("imp"),
        "NEPTUNE-ID": neptune_id,
    }

    # Add sub-analysis flags
    if "USE_COHORT2_FILTER" in config:
        row["COHORT2"] = int(bool(config["USE_COHORT2_FILTER"]))
    if "FILTER_SQUAMOUS" in config:
        row["SQUAMOUS"] = config["FILTER_SQUAMOUS"]
    if "FILTER_CHEMO_IMMUNO" in config:
        row["CHEMO_IMMUNO"] = int(config["FILTER_CHEMO_IMMUNO"])
    if "FILTER_INT" in config:
        row["INT_SUB"] = int(bool(config["FILTER_INT"]))
    if "FILTER_PDL1" in config:
        row["PDL1_GROUP"] = config["FILTER_PDL1"]
    if "FILTER_ALL_MODS" in config:
        row["ALL MODS"] = int(bool(config["FILTER_ALL_MODS"]))

    if cohort:
        row["COHORT"] = cohort
    if hyperparam_combo:
        row["HYPERPARAM COMBO"] = hyperparam_combo
    if seed is not None:
        row["SEED"] = seed  

    # 📊 Handle averaged scores (default case)
    average_file = os.path.join(score_path, "average_std_train_test_scores.csv")
    if os.path.exists(average_file):
        # now expecting 4 rows: Weighted Mean, Standard Error, 95% CI Lower, 95% CI Upper
        scores = pd.read_csv(average_file, index_col=0)
        if scores.shape[0] < 4:
            print(f"❌ Incomplete average score file: {average_file}")
            return

    mean     = scores.loc["Weighted Mean"]
    se       = scores.loc["Standard Error"]
    ci_lower = scores.loc["95% CI Lower"]
    ci_upper = scores.loc["95% CI Upper"]

    if config.get("task") == "classification":
        row.update({
            "TEST - F1":           fmt(mean["F1"], se["F1"]),
            "TEST - F1 CI":        f"[{ci_lower['F1']:.4f}, {ci_upper['F1']:.4f}]",
            "TEST - AUC":          fmt(mean["AUC"], se["AUC"]),
            "TEST - AUC CI":       f"[{ci_lower['AUC']:.4f}, {ci_upper['AUC']:.4f}]",
            "TEST - Sensitivity":  fmt(mean["Sensitivity"], se["Sensitivity"]),
            "TEST - Sensitivity CI": f"[{ci_lower['Sensitivity']:.4f}, {ci_upper['Sensitivity']:.4f}]",
            "TEST - Specificity":  fmt(mean["Specificity"], se["Specificity"]),
            "TEST - Specificity CI": f"[{ci_lower['Specificity']:.4f}, {ci_upper['Specificity']:.4f}]",
        })

    elif config.get("task") == "survival":
        row.update({
    

            "TEST - C-INDEX":        fmt(mean["C_INDEX_TEST"], se["C_INDEX_TEST"]),
            "TEST - C-INDEX CI":     f"[{ci_lower['C_INDEX_TEST']:.4f}, {ci_upper['C_INDEX_TEST']:.4f}]",
            "TEST - BRIER":          fmt(mean["BRIER_SCORE_TEST"], se["BRIER_SCORE_TEST"]),
            "TEST - BRIER CI":       f"[{ci_lower['BRIER_SCORE_TEST']:.4f}, {ci_upper['BRIER_SCORE_TEST']:.4f}]",
            "TEST - Mean AUC":       fmt(mean["MEAN_AUC_TEST"], se["MEAN_AUC_TEST"]),
            "TEST - Mean AUC CI":    f"[{ci_lower['MEAN_AUC_TEST']:.4f}, {ci_upper['MEAN_AUC_TEST']:.4f}]",
        })

    # 📥 Handle single model (train and test scores)
    elif single_model:
        train_path = os.path.join(score_path, "scores_train.csv")
        test_path = os.path.join(score_path, "scores_test.csv")

        train = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
        test = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

        if config.get("task") == "classification":
            for metric in ["F1", "AUC", "Sensitivity", "Specificity"]:
                if metric in train.columns:
                    row[f"TRAIN - {metric}"] = fmt(train[metric].iloc[0])
                if metric in test.columns:
                    row[f"TEST - {metric}"] = fmt(test[metric].iloc[0])

        elif config.get("task") == "survival":
            for metric in ["C_INDEX", "BRIER_SCORE", "MEAN_AUC"]:
                if f"{metric}_TRAIN" in train.columns:
                    row[f"TRAIN - {metric.replace('_', '-')}"] = fmt(train[f"{metric}_TRAIN"].iloc[0])
                if f"{metric}_TEST" in test.columns:
                    row[f"TEST - {metric.replace('_', '-')}"] = fmt(train[f"{metric}_TEST"].iloc[0])

    # 📂 Ensure the directory exists
    excel_dir = os.path.dirname(excel_path)
    if excel_dir and not os.path.exists(excel_dir):
        os.makedirs(excel_dir, exist_ok=True)

    # 📊 Append or create the Excel file
    try:
        df = pd.read_excel(excel_path, engine="openpyxl") if os.path.exists(excel_path) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"✅ Exported scores to Excel → {excel_path}")
    except Exception as e:
        print(f"❌ Failed to export scores to Excel: {e}")

