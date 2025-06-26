

def plot_ROC(group_path: str, seed: int = 0):
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, auc
    import os

    """
    Plot the ROC curves for all folds in a group experiment.

    This function:
    - Loads prediction files (with or without seed subfolders) for each fold
    - Extracts predicted probabilities and true labels
    - Computes per-fold ROC curves and AUC scores
    - Computes the weighted average ROC curve using fold sample sizes
    - Plots all ROC curves using Plotly and saves the result as an HTML file

    Args:
        group_path (str): Path to the group directory containing folds.
        seed (int): Seed index to select the appropriate subfolder (if present).

    Output:
        Saves the ROC plot to `roc_curve_plot.html` inside the group path.
    """

    # Softmax function to convert logits to probabilities
    def softmax(logits):
        logits = np.array(logits)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        if probs.shape[1] != 2:
            raise ValueError(f"Error: Expected shape (N, 2), got {probs.shape}")
        return probs

    # 🔍 Collect all fold directories
    fold_dirs = [f for f in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, f))]
    folds_mapping = {}
    all_dfs = []

    for fold in fold_dirs:
        fold_path = os.path.join(group_path, fold)
        found = False

        # Walk through subfolders to find predictions.parquet
        for root, _, files in os.walk(fold_path):
            if f"seed_{seed}" in root and "predictions.parquet" in files:
                pred_path = os.path.join(root, "predictions.parquet")
                found = True
                break
            elif "predictions.parquet" in files and not found:
                pred_path = os.path.join(root, "predictions.parquet")
                found = True  # fallback

        if not found:
            print(f"⚠️ File not found in any subfolder: {fold_path}")
            continue

        df = pd.read_parquet(pred_path)
        df['fold'] = fold  # track fold name
        all_dfs.append(df)

        for slide in df['slide']:
            folds_mapping[slide] = fold

    if not all_dfs:
        print("❌ No predictions found.")
        return

    # 📦 Merge all folds into a single DataFrame
    df_all = pd.concat(all_dfs, ignore_index=True)

    # 🎯 Get predicted probabilities for class 1
    if 'y_pred1' in df_all.columns and 'y_pred0' in df_all.columns:
        probs = softmax(df_all[['y_pred0', 'y_pred1']].values)[:, 1]
    elif 'y_pred0' in df_all.columns:
        probs = 1 - df_all['y_pred0'].values
    else:
        raise ValueError("❌ Missing y_pred0/y_pred1 — cannot compute probabilities.")

    # ✅ Ground truth labels (either y_true1 or y_true for classification)
    if 'y_true1' in df_all.columns:
        y_true_col = 'y_true1'
    elif 'y_true' in df_all.columns:
        y_true_col = 'y_true'
    else:
        raise ValueError("❌ Missing y_true or y_true1 column — required for ROC.")


    df_all['prob'] = probs

    # 📊 Prepare ROC plotting
    unique_folds = df_all['fold'].unique()
    fold_sizes = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = go.Figure()

    for fold in unique_folds:
        fold_df = df_all[df_all['fold'] == fold]
        fold_sizes.append(len(fold_df))

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(fold_df[y_true_col], fold_df['prob'])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR for consistent averaging
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Add fold-specific ROC to plot
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'{fold} (AUC = {roc_auc:.2f})',
            mode='lines'
        ))

    # 🧠 Compute weighted average ROC across folds
    weighted_tpr = np.average(tprs, axis=0, weights=fold_sizes)
    weighted_auc = auc(mean_fpr, weighted_tpr)
    std_auc = np.std(aucs)

    # Add mean ROC curve
    fig.add_trace(go.Scatter(
        x=mean_fpr,
        y=weighted_tpr,
        name=f'Mean (AUC = {weighted_auc:.2f} ± {std_auc:.2f})',
        mode='lines'
    ))

    # Add random classifier line (baseline)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random Classifier',
        line=dict(width=1, color='black', dash='dash'),
        mode='lines'
    ))

    # 🖼️ Update layout and save plot
    fig.update_layout(
        title=f'ROC Curve ({len(unique_folds)} folds)',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.7, y=0.1)
    )

    out_path = os.path.join(group_path, "roc_curve_plot.html")
    fig.write_html(out_path)
    print(f"📈 ROC curve saved to: {out_path}")
