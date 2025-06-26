def pick_best_hyperparams(base_results_path, config):
    """
    Itera su tutte le subfolder hparam_*/seed_*/,
    calcola i punteggi medi, e restituisce gli hyperparams del combo migliore.
    Supporta sia classification (AUC) che survival (C-INDEX).
    """
    import os
    import pandas as pd
    import re
    from metrics.compute_scores import compute_scores
    from metrics.calculate_average import compute_weighted_average

    print(f"\n🔍 Searching best hyperparameters in: {base_results_path}")

    task = config["task"]
    metric_col = "TEST - AUC" if task == "classification" else "C_INDEX_TEST"

    best_score = -1
    best_combo = None

    for combo_folder in os.listdir(base_results_path):
        combo_path = os.path.join(base_results_path, combo_folder)
        if not combo_folder.startswith("hparam_") or not os.path.isdir(combo_path):
            continue

        seed_scores = []
        for seed_folder in os.listdir(combo_path):
            seed_path = os.path.join(combo_path, seed_folder)  # ⬅️ now go into "eval"
            if not os.path.isdir(seed_path):
                continue

            # 🧮 Calcola scores se non esistono
            compute_scores(seed_path, task)
            compute_weighted_average(seed_path, config["task"])

            score_file = os.path.join(seed_path, "average_std_train_test_scores.csv")
            if not os.path.exists(score_file):
                print(f"⚠️ Missing summary in {seed_path}")
                continue

            try:
                df = pd.read_csv(score_file, index_col=0)
                metric = metric_col
                if metric_col not in df.columns:
                    fallback = metric_col.replace("TEST - ", "")
                    if fallback in df.columns:
                        metric = fallback
                    else:
                        print(f"⚠️ {metric_col} not found in {score_file}")
                        continue
                score = df.loc["Test Mean", metric]
                seed_scores.append(score)
            except Exception as e:
                print(f"⚠️ Error reading {score_file}: {e}")
                continue

        if seed_scores:
            avg_score = sum(seed_scores) / len(seed_scores)
            print(f"📊 {combo_folder} → {metric_col} (avg over {len(seed_scores)} seeds): {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_combo = combo_folder

    if not best_combo:
        raise RuntimeError("❌ No valid hyperparameter combo found.")

    print(f"\n🏆 Best combo: {best_combo} with {metric_col} = {best_score:.4f}")

    # 🔍 Estrai hyperparametri dal nome cartella
    match = re.findall(r"([a-z]{3})([0-9]+)", best_combo)
    hyperparams = {}
    for k, v in match:
        if k == "bat":
            hyperparams["batch_size"] = int(v)
        elif k == "rec":
            hyperparams["reconstruction_weight"] = float(f"{v[0]}.{v[1:]}")
        elif k == "n_l":
            hyperparams["n_layers"] = int(v)

    return hyperparams


def pick_best_final_model(final_model_path, task):
    import glob
    
    """
    Picks the best final model among seeds, using AUC (classification) or C-INDEX (survival),
    reading from either 'scores.csv' or 'scores_test.csv' inside each seed folder.
    """

    import os
    import pandas as pd

    metric_map = {
        "classification": "AUC",
        "survival": "C_INDEX_TEST"
    }

    if task not in metric_map:
        raise ValueError(f"❌ Unsupported task type: {task}")

    metric_col = metric_map[task]
    best_score = -1
    best_seed = None
    best_path = None

    print(f"\n🔍 Searching best final model in: {final_model_path}")

    for folder in os.listdir(final_model_path):
        if not folder.startswith("seed_"):
            continue

        seed_root = os.path.join(final_model_path, folder)

        # Cerca la cartella eval/... con uno score valido
        eval_subfolders = glob.glob(os.path.join(seed_root, "eval", "*"))
        if not eval_subfolders:
            print(f"⚠️ No eval folder found in {seed_root}")
            continue

        # Cerca il primo subfolder con scores
        score_path = None
        for sub in eval_subfolders:
            for name in ["average_std_train_test_scores.csv", "scores_test.csv", "scores.csv"]:
                candidate = os.path.join(sub, name)
                if os.path.exists(candidate):
                    score_path = candidate
                    break
            if score_path:
                break

        if not score_path:
            print(f"⚠️ No score files found in {seed_root}")
            continue

        try:
            df = pd.read_csv(score_path, index_col=0 if "average" in score_path else None)
            score = df.loc["Test Mean", metric_col] if "average" in score_path else df[metric_col].iloc[0]
        except Exception as e:
            print(f"⚠️ Failed to read {score_path}: {e}")
            continue

        print(f"📊 {folder} → {metric_col}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_seed = folder
            best_path = os.path.dirname(score_path)  # 👈 RETURN eval/.../ path!

    if best_seed is None:
        raise RuntimeError("❌ No valid final model found!")

    print(f"\n🏆 Best final model: {best_seed} with {metric_col} = {best_score:.4f}")
    return {
        "seed": best_seed,
        "score": best_score,
        "metric": metric_col,
        "path": best_path
    }
