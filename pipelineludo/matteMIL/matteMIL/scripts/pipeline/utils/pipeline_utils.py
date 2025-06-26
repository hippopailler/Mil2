import os
from metrics import compute_weighted_average
from uploading import upload_experiment
from folders import group_folders_by_suffix_and_seed
from plotting import plot_ROC

def list_experiments(mods_path):
    """
    List valid experiment folders inside a modality path.
    Skips hidden files and non-directories.
    """
    return [
        name for name in os.listdir(mods_path)
        if not name.startswith(".") and os.path.isdir(os.path.join(mods_path, name))
    ]

def build_path(base_path, config, mods):
    """
    Build the path for the experiment based on the configuration and modalities.
    """
    source_imp = f"{config['source']}-{config['imp']}"
    modality_list = "_".join(mods)
    mods_path = os.path.join(base_path, config["task"], config["data-type"], config["training_type"], source_imp, modality_list)
    return mods_path
  
def finalize_results(mods_path, config, excel_path, updated_config=None):
    """
    Handles scoring, ROC plotting (only for classification + CV), Neptune upload,
    and Excel export depending on training type.
    """
    updated_config = updated_config or config
    is_classification = config["task"] == "classification"
    is_cv = config["training_type"] in ["cross_validation", "cross_validation_stratified"]

    if config["training_type"] == "cross_validation_stratified":
        print(f"\n📁 STEP 4: Organize by cohort for {mods_path}")
        group_folders_by_suffix_and_seed(mods_path)

        for seed in config["seed"]:
            for group_id in ["2", "3", "23"]:
                group_folder_name = f"group_{group_id}_seed_{seed}"
                group_path = os.path.join(mods_path, group_folder_name)

                if not os.path.exists(group_path):
                    print(f"⚠️ Skipping missing group folder: {group_folder_name}")
                    continue

                print(f"\n📊 STEP 5: Compute weighted averages for {group_folder_name}")
                compute_weighted_average(group_path)

                if is_classification:
                    print(f"📈 Plotting ROC for {group_folder_name}")
                    plot_ROC(group_path)

                if config.get("neptune", False):
                    print(f"\n🚀 STEP 6: Upload to Neptune for {group_folder_name}")
                    neptune_id = upload_experiment(group_path, config)
                else:
                    neptune_id = None

                if config.get("excel", False):
                    print(f"📤 STEP 7: Export to Excel for {group_folder_name}")
                    export_results_to_excel(group_path, updated_config, neptune_id, excel_path)

    else:
        print(f"\n📊 STEP 5: Compute weighted averages for {mods_path}")
        compute_weighted_average(mods_path)

        if is_classification and is_cv:
            print(f"📈 Plotting ROC for {mods_path}")
            plot_ROC(mods_path)

        if config.get("neptune", True):
            print(f"\n🚀 STEP 6: Upload to Neptune for {mods_path}")
            # neptune_id = upload_experiment(mods_path, config)
            neptune_id = "None"
        else:
            neptune_id = "None"

        if config.get("excel", True):
            print(f"📤 STEP 7: Export to Excel for {mods_path}")
            print(f"excel path: {excel_path}")
            export_results_to_excel(mods_path, updated_config, excel_path, neptune_id,)

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

sns.set_palette("husl")  # Use a diverse color palette for distinct bars

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparative_results(excel_path, config):
    if not os.path.exists(excel_path):
        print(f"⚠️ Excel file not found: {excel_path}")
        return

    df = pd.read_excel(excel_path)

    # Determine metric and its CI column
    task_type     = config.get("task")
    metric_column = "TEST - C-INDEX" if task_type == "survival" else "TEST - AUC"
    metric_ci_col = f"{metric_column} CI"

    required_columns = ["MODALITY", "SOURCE", "OUTCOME", metric_column, metric_ci_col]
    if not all(col in df.columns for col in required_columns):
        print(f"⚠️ Required columns not found in the Excel file: {', '.join(required_columns)}")
        return

    # Update the MODALITY column based on SOURCE
    df["MODALITY"] = df.apply(
        lambda row: row["MODALITY"].replace(
            "rad", "radF" if row["SOURCE"] == "foundation" else "pyrad"
        ) if "rad" in row["MODALITY"].split("_") else row["MODALITY"],
        axis=1
    )

    # Prepare sub-analysis info for title and filename
    sub_analysis_parts = []
    if config.get("USE_COHORT2_FILTER"):
        sub_analysis_parts.append("cohort2")
    if config.get("FILTER_SQUAMOUS") in {"0.0", "1.0"}:
        sub_analysis_parts.append(f"squamous_{config['FILTER_SQUAMOUS']}")
    if config.get("FILTER_CHEMO_IMMUNO") in {"0", "1"}:
        sub_analysis_parts.append(f"chemoio_{config['FILTER_CHEMO_IMMUNO']}")
    if config.get("FILTER_INT"):
        sub_analysis_parts.append("int_sub")
    if config.get("FILTER_PDL1"):
        sub_analysis_parts.append(f"pdl1_{config['FILTER_PDL1']}")
    if config.get("FILTER_ALL_MODS"):
        sub_analysis_parts.append("all_mods")

    sub_analysis_tag  = "_".join(sub_analysis_parts)
    sub_analysis_text = ", ".join(sub_analysis_parts).replace("_", " ").capitalize()

    # Create output directory
    output_dir_name = "comparative_plots" + (f"_{sub_analysis_tag}" if sub_analysis_tag else "")
    output_dir = os.path.join(os.path.dirname(excel_path), output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for plotting
    modalities, means, errors = [], [], []
    for modality, group in df.groupby("MODALITY"):
        fmt_val = str(group[metric_column].iat[0])   # e.g., "0.7543 ± 0.0123"
        ci_val  = str(group[metric_ci_col].iat[0])   # e.g., "[0.7300, 0.7786]"

        # Parse mean from "mean ± se"
        m = re.match(r"([0-9.]+)\s*±", fmt_val)
        if not m:
            print(f"⚠️ Bad format in {metric_column}: {fmt_val}. Skipping.")
            continue
        mean = float(m.group(1))

        # Parse lower/upper from "[low, high]"
        c = re.match(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", ci_val)
        if not c:
            print(f"⚠️ Bad format in {metric_ci_col}: {ci_val}. Skipping.")
            continue
        low, high = float(c.group(1)), float(c.group(2))
        error = (high - low) / 2

        modalities.append(modality.replace("_", ", "))
        means.append(mean)
        errors.append(error)

    # Plot results with error bars representing half the CI width
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("husl", len(modalities))
    plt.bar(modalities, means, yerr=errors, capsize=8, color=colors,
            edgecolor='black', linewidth=1.2)
    max_error = max(errors) if errors else 0
    for i, m in enumerate(means):
        plt.text(i, m + max_error * 0.05, f"{m:.4f}", ha='center', fontsize=10, fontweight='bold')

    outcome = df["OUTCOME"].unique()[0]
    title_parts = [f"Comparative {metric_column} for {outcome}"]
    if sub_analysis_text:
        title_parts.append(f"({sub_analysis_text})")

    plt.title(" ".join(title_parts), fontsize=18)
    plt.xlabel("Modality", fontsize=14)
    plt.ylabel(metric_column, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot
    filename = f"comparative_{metric_column.replace(' ', '_').lower()}_{outcome}"
    if sub_analysis_tag:
        filename += f"_{sub_analysis_tag}"
    plot_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparative plot saved at: {plot_path}")

def build_excel_path(base_dir, config, mods):
    print(f"\n🛠️ Building path with config: {config}")

    training_type = config.get("training_type", "")
    sub_dirs = []
    if training_type:
        sub_dirs.append(training_type)

    if config.get("USE_COHORT2_FILTER"):
        sub_dirs.append("cohort2")

    if "FILTER_SQUAMOUS" in config:
        val = config.get("FILTER_SQUAMOUS")
        sub_dirs.append(f"squamous_{val}")

    if "FILTER_CHEMO_IMMUNO" in config:
        val = config.get("FILTER_CHEMO_IMMUNO")
        sub_dirs.append(f"chemoio_{val}")
    
    if "FILTER_INT" in config:
        sub_dirs.append(f"int_sub")

    if "FILTER_PDL1" in config:
        val = config.get("FILTER_PDL1")
        sub_dirs.append(f"pdl1_{val}")
    
    if "FILTER_ALL_MODS" in config:
        sub_dirs.append(f"all_mods")

    outcome = config.get("task_settings", {}).get("outcome") or config.get("task")
    if isinstance(outcome, list):
        outcome = "_".join(outcome)

    filters_str = "_".join(sub_dirs) if sub_dirs else "all"
    directory = os.path.join(base_dir, *sub_dirs)
    filename = f"{outcome}_{filters_str}_results.xlsx"
    full_path = os.path.join(directory, filename)

    print(f"✅ Built path: {full_path}")
    os.makedirs(directory, exist_ok=True)
    return full_path
