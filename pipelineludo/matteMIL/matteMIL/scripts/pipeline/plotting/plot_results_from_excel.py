import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparative_results(excel_path, config):
    """
    Reads one Excel file at excel_path, expects columns:
      ["MODALITY","SOURCE","OUTCOME","TEST - AUC", "TEST - AUC CI"] (or "TEST - C-INDEX" if survival).
    Parses "mean ± SE" from the metric column, sorts modalities by descending mean, 
    and saves a bar‐plot with SE errorbars under:
      <same‐parent‐dir>/comparative_plots_{subanalysis}/...
    """
    if not os.path.exists(excel_path):
        print(f"⚠️ Excel file not found: {excel_path}")
        return

    df = pd.read_excel(excel_path, engine="openpyxl")

    task_type = config.get("task", "survival")
    metric_column = "TEST - C-INDEX" if task_type == "survival" else "TEST - AUC"
    metric_ci_col = f"{metric_column} CI"

    required = ["MODALITY", "SOURCE", "OUTCOME", metric_column, metric_ci_col]
    if not all(col in df.columns for col in required):
        print(f"⚠️ Missing columns in {excel_path}: {', '.join(required)}")
        return

    # Adjust "rad"→"radF" vs "pyrad" if needed:
    df["MODALITY"] = df.apply(
        lambda row: row["MODALITY"].replace(
            "rad", "radF" if row["SOURCE"] == "foundation" else "pyrad"
        ) if "rad" in row["MODALITY"].split("_") else row["MODALITY"],
        axis=1
    )

    # Build sub-analysis tag/text:
    parts = []
    if config.get("USE_COHORT2_FILTER"):
        parts.append("cohort2")
    if config.get("FILTER_SQUAMOUS") in {"0.0", "1.0"}:
        parts.append(f"squamous_{config['FILTER_SQUAMOUS']}")
    if config.get("FILTER_CHEMO_IMMUNO") in {"0", "1"}:
        parts.append(f"chemoio_{config['FILTER_CHEMO_IMMUNO']}")
    if config.get("FILTER_INT"):
        parts.append("int_sub")
    if config.get("FILTER_PDL1"):
        parts.append(f"pdl1_{config['FILTER_PDL1']}")
    if config.get("FILTER_ALL_MODS"):
        parts.append("all_mods")

    sub_tag = "_".join(parts)
    sub_text = ", ".join(parts).replace("_", " ").capitalize()

    # Create output folder next to the Excel file:
    parent = os.path.dirname(excel_path)
    outdir_name = "comparative_plots" + (f"_{sub_tag}" if sub_tag else "")
    outdir = os.path.join(parent, outdir_name)
    os.makedirs(outdir, exist_ok=True)

    # Collect (modality, mean, se) tuples, then sort by mean desc:
    rows = []
    for modality, grp in df.groupby("MODALITY"):
        fmt_val = str(grp[metric_column].iat[0])  # e.g. "0.7543 ± 0.0123"
        m = re.match(r"\s*([0-9.]+)\s*±\s*([0-9.]+)\s*", fmt_val)
        if not m:
            print(f"⚠️ Bad format in {metric_column}: {fmt_val}. Skipping.")
            continue
        mean, se = float(m.group(1)), float(m.group(2))
        rows.append((modality.replace("_", ", "), mean, se))

    if not rows:
        print(f"⚠️ No valid rows to plot in {excel_path}.")
        return

    # Sort by mean descending:
    rows.sort(key=lambda x: x[1], reverse=True)
    modalities, means, errors = zip(*rows)

    # Plot bar + SE in sorted order:
    plt.figure(figsize=(14, 10))
    colors = sns.color_palette("husl", len(modalities))
    plt.bar(modalities, means, yerr=errors, capsize=8,
            color=colors, edgecolor="black", linewidth=1.2)
    max_err = max(errors)

    for i, mval in enumerate(means):
        plt.text(i, mval + max_err * 0.05, f"{mval:.4f}",
                 ha="center", fontsize=10, fontweight="bold")

    outcome = df["OUTCOME"].unique()[0]
    title = f"Comparative {metric_column} for {outcome}"
    if sub_text:
        title += f" ({sub_text})"
    plt.title(title, fontsize=18)
    plt.xlabel("Modality", fontsize=14)
    plt.ylabel(metric_column, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    fn = f"comparative_{metric_column.replace(' ', '_').lower()}_{outcome}"
    if sub_tag:
        fn += f"_{sub_tag}"
    save_path = os.path.join(outdir, f"{fn}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

def infer_config_from_path(path):
    """
    Given a path like ".../results/.../cohort2/chemoio_1/squamous_0.0/XXX.xlsx",
    return a config dict:
      USE_COHORT2_FILTER  = True if "cohort2" in any component
      FILTER_CHEMO_IMMUNO = "0" or "1" if folder named "chemoio_0"/"chemoio_1"
      FILTER_SQUAMOUS     = "0.0" or "1.0" if folder "squamous_0.0"/"squamous_1.0"
      FILTER_INT          = True if "int_sub" in any folder
      FILTER_PDL1         = value after "pdl1_" if found
      FILTER_ALL_MODS     = True if "all_mods" in any folder
      task                = "classification" by default
    """
    cfg = {
        "task": "classification",  # Default task type, can be overridden
        "USE_COHORT2_FILTER": False,
        "FILTER_CHEMO_IMMUNO": None,
        "FILTER_SQUAMOUS": None,
        "FILTER_INT": False,
        "FILTER_PDL1": None,
        "FILTER_ALL_MODS": False,
    }
    comps = path.replace("\\", "/").split("/")

    for c in comps:
        if c.lower() == "cohort2":
            cfg["USE_COHORT2_FILTER"] = True
        if c.startswith("chemoio_"):
            # chemoio_0 or chemoio_1
            cfg["FILTER_CHEMO_IMMUNO"] = c.split("_")[1]
        if c.startswith("squamous_"):
            # squamous_0.0 or squamous_1.0
            cfg["FILTER_SQUAMOUS"] = c.split("_")[1]
        if c.lower() == "int_sub":
            cfg["FILTER_INT"] = True
        if c.startswith("pdl1_"):
            cfg["FILTER_PDL1"] = c.split("_", 1)[1]
        if c.lower() == "all_mods":
            cfg["FILTER_ALL_MODS"] = True

    # If you ever want to override for survival vs classification, you could inspect path or
    # pass an extra flag. By default it's classification → AUC.
    return cfg


def plot_all_comparative_results(base_dir="results"):
    """
    Walks through base_dir, finds every .xlsx (except anything already in a 'comparative_plots_*' folder),
    infers config from its path, and calls plot_comparative_results(...) for each one.
    """
    for root, dirs, files in os.walk(base_dir):
        # Skip any folder named "comparative_plots_*" to avoid infinite loops
        if os.path.basename(root).startswith("comparative_plots"):
            continue

        for fname in files:
            if not fname.lower().endswith(".xlsx"):
                continue
            excel_path = os.path.join(root, fname)

            # Infer config from the folder structure
            cfg = infer_config_from_path(root)

            # Finally call the plotting function
            plot_comparative_results(excel_path, cfg)


if __name__ == "__main__":
    # Simply call this script from the same directory that contains "results/"
    plot_all_comparative_results("/Users/ludole/Desktop/results/cross_validation/pdl1_low")