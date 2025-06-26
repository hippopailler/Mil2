
def upload_experiment(base_folder, config, tags):
    """
    Uploads either a full CV experiment (with multiple subfolders) or a single experiment folder to Neptune.

    Args:
        base_folder (str): Path to the experiment folder.
        config (dict): Training config used to extract task metadata.

    Returns:
        str: Neptune run ID.
    """
    import os
    import neptune
    import json
    from dotenv import load_dotenv
    import yaml
    
    # Load Neptune API key
    load_dotenv(".env")
    API_KEY = os.getenv("NEPTUNE_API_KEY")
   # 🛠️ Fix outcomes
    outcomes = config["task_settings"]["outcome"]
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    folder_name = f"{outcomes}"

    base_tags = [
        config["task"],
        config["task_settings"]["loss"],
        *outcomes,
        config["training_type"],
        config["data-type"],
        config["source"],
        config["imp"],
        str(config["seed"])
    ]

    if tags:
        base_tags += tags

    print(f"🚀 Starting Neptune run for {folder_name} with tags:\n{base_tags}")

    run = neptune.init_run(
        project="albertus/Multimodal",
        api_token=API_KEY,
        tags=base_tags,
        name=folder_name
    )

    # Upload common summary files (always)
    for fname in ["average_test_scores.csv", "average_std_train_test_scores.csv", "roc_curve_plot.html"]:
        fpath = os.path.join(base_folder, fname)
        if os.path.exists(fpath):
            run[fname].upload(fpath)
            print(f"📤 Uploaded: {fname}")

    # Detect if it's a CV group folder or a single experiment
    subdirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    if subdirs:
        print("📂 Detected group folder, uploading all folds...")

        for cv_folder in subdirs:
            cv_folder_path = os.path.join(base_folder, cv_folder)

            if os.path.isdir(cv_folder_path):
                for root, _, files in os.walk(cv_folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_root = os.path.relpath(root, base_folder)

                        if "attention" in root and file.endswith(".npz"):
                            run[f"{rel_root}/attention/{file}"].upload(file_path)
                        elif "model" in root and file.endswith(".pth"):
                            run[f"{rel_root}/best_model/{file}"].upload(file_path)
                        elif file == "history.csv":
                            run[f"{rel_root}/history/{file}"].upload(file_path)
                        elif file == "mil_params.json":
                            with open(file_path, "r") as f:
                                params = json.load(f)
                                for key, value in params.items():
                                    run[f"{rel_root}/params/{key}"] = value
                        elif file.endswith(".png"):
                            run[f"{rel_root}/images/{file}"].upload(file_path)
                        elif file.endswith(".parquet"):
                            run[f"{rel_root}/predictions/{file}"].upload(file_path)
                        elif file == "slide_manifest.csv":
                            run[f"{rel_root}/slide_manifest/{file}"].upload(file_path)
                        elif file == "scores_test.csv":
                            run[f"{rel_root}/scores_test/{file}"].upload(file_path)
                        elif file == "scores_train.csv":
                            run[f"{rel_root}/scores_train/{file}"].upload(file_path)

    else:
        print("📁 Detected single experiment folder.")

        for file in os.listdir(base_folder):
            file_path = os.path.join(base_folder, file)

            if file == "mil_params.json":
                with open(file_path, "r") as f:
                    params = json.load(f)
                    for key, value in params.items():
                        run[f"params/{key}"] = value
            elif file.endswith(".csv") or file.endswith(".png") or file.endswith(".parquet") or file.endswith(".npz"):
                run[file] = neptune.types.File(file_path)
                print(f"📤 Uploaded: {file}")

    neptune_id = run["sys/id"].fetch()
    run.stop()
    print(f"✅ Neptune run complete for {folder_name}. ID: {neptune_id}")
    return neptune_id
