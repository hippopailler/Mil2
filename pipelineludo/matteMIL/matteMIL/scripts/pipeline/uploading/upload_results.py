import os
import re
import pandas as pd
from .excel import export_scores_to_excel
from plotting import plot_ROC
from .neptune import upload_experiment

# from utils import export_scores_to_excel, plot_auc_curve, upload_to_neptune
# These functions will be assumed available or mocked in real use

def handle_cv_upload(modality_paths, config, excel_path, mods):
    """
    Upload results from cross-validation training.
    - Upload the whole modality folder to Neptune
    - Export weighted average scores to Excel
    - If classification task, generate and upload ROC AUC plot
    """
    import os
    from .excel import export_scores_to_excel
    from .neptune import upload_experiment

    print("\n🚀 Handling CV upload...")

    for path in modality_paths:
        print(f"📁 Uploading to Neptune: {path}")
        try:
            tags = ["cv_training"]
            if "USE_COHORT2_FILTER" in config:
                tags.append(f"cohort2:{int(bool(config['USE_COHORT2_FILTER']))}")
            if "USE_FOUNDATION_MODEL" in config:
                src = "foundation" if config["USE_FOUNDATION_MODEL"] else "pyrad"
                tags.append(f"radiomics_source:{src}")
            if "FILTER_SQUAMOUS" in config:
                tags.append(f"squamous:{config['FILTER_SQUAMOUS']}")
            if "FILTER_CHEMO_IMMUNO" in config:
                tags.append(f"chemo_immuno:{config['FILTER_CHEMO_IMMUNO']}")
            if "FILTER_INT" in config:
                tags.append(f"int_sub:{config['FILTER_INT']}")
            if "FILTER_PDL1" in config:
                tags.append(f"pdl1_group:{config['FILTER_PDL1']}")
            if "FILTER_ALL_MODS" in config:
                tags.append(f"all_mods:{int(bool(config['FILTER_ALL_MODS']))}")

            # neptune_id = upload_experiment(path, config, tags=tags)
            neptune_id = None
        except Exception as e:
            print(f"❌ Failed to upload to Neptune: {e}")
            neptune_id = None

        print("📊 Exporting CV average scores to Excel...")
        export_scores_to_excel(
            score_path=path,
            config=config,
            excel_path=excel_path,
            neptune_id=neptune_id if neptune_id else None,
            cohort='23',
            mods=mods,
            single_model=False,
        )

def handle_cv_stratified_upload(modality_path, config, excel_path, neptune_id):
    """
    Upload results from cross-validation stratified training.
    - Each modality folder contains 3 group subfolders (2, 3, 23)
    - Export weighted average scores for each group to Excel
    - Upload entire modality folder to Neptune
    - If classification task, generate and upload ROC AUC plot
    """
    print("\n🚀 Handling Stratified CV upload...")
    # TODO: Upload to Neptune
    # TODO: Export group scores to Excel
    # TODO: Plot AUC if classification
    pass

def handle_standard_upload(modality_paths, config, excel_path, mods):
    """
    Upload results from standard training.
    - Upload each modality/seed/fold folder to Neptune
    - Export the score to Excel
    """

    import os
    from .excel import export_scores_to_excel
    from .neptune import upload_experiment


    print("\n🚀 Handling Standard upload...")

    for path in modality_paths:
        score_path = os.path.join(path)
        if not os.path.exists(score_path):
            print(f"⚠️ No test scores found in: {path}")
            continue

    print(f"📤 Uploading standard experiment: {path}")
    mod_tags = [f"{mod}:{val}" for mod, val in mods.items() if val]

    filter_tags = []
    if "USE_COHORT2_FILTER" in config:
        filter_tags.append(f"cohort2:{int(bool(config['USE_COHORT2_FILTER']))}")
    if "USE_FOUNDATION_MODEL" in config:
        src = "foundation" if config["USE_FOUNDATION_MODEL"] else "pyrad"
        filter_tags.append(f"radiomics_source:{src}")
    if "FILTER_SQUAMOUS" in config:
        filter_tags.append(f"squamous:{config['FILTER_SQUAMOUS']}")
    if "FILTER_CHEMO_IMMUNO" in config:
        filter_tags.append(f"chemo_immuno:{config['FILTER_CHEMO_IMMUNO']}")
    if "FILTER_INT" in config:
        filter_tags.append(f"int_sub:{config['FILTER_INT']}")
    if "FILTER_PDL1" in config:
        filter_tags.append(f"pdl1_group:{config['FILTER_PDL1']}")
    if "FILTER_ALL_MODS" in config:
        filter_tags.append(f"all_mods:{int(bool(config['FILTER_ALL_MODS']))}")


    tags = [
        "standard_training",
        *mod_tags,
        *filter_tags
        
    ]

    try:
        # neptune_id = upload_experiment(path, config, tags=tags)
        neptune_id = None
    except Exception as e:
        print(f"❌ Failed to upload to Neptune: {e}")
        neptune_id = None

    export_scores_to_excel(
        score_path=score_path,
        config=config,
        excel_path=excel_path,
        neptune_id=neptune_id if neptune_id else None,
        cohort='23',
        mods=mods,
        single_model=True,
    )

def handle_hyperparam_final_model_upload(
    training_return,
    config,
    excel_path_final,
    mods,
):
    """
    Upload results from hyperparameter tuning:
    - Upload each hparam_*/seed_X/ folder
    - Upload each best final model for each seed
    - Export scores to Excel
    """
    import os
    from .excel import export_scores_to_excel
    from .neptune import upload_experiment

    print("\n🚀 Uploading hyperparameter trials and best final models...")

    hyperparam_dir = training_return["hyperparam_dir"]
    final_model_dir = training_return["final_model_dir"]
    best_models_per_seed = training_return["best_models_per_seed"]  # Dict[seed] = path
    best_combos_per_seed = training_return["best_combos_per_seed"]  # Dict[seed] = combo

    excel_path_hparam = os.path.join(hyperparam_dir, "hyperparam_results.xlsx")

    
    # 1️⃣ Upload each hparam combo/seed
    for combo_folder in os.listdir(hyperparam_dir):
        combo_path = os.path.join(hyperparam_dir, combo_folder)
        if not os.path.isdir(combo_path) or not combo_folder.startswith("hparam_"):
            continue

        for seed_folder in os.listdir(combo_path):
            seed_path = os.path.join(combo_path, seed_folder)
            if not os.path.isdir(seed_path):
                continue
            
            score_path = os.path.join(seed_path, "average_std_train_test_scores.csv")
            if not os.path.exists(score_path):
                continue

            # Inside the loop over hyperparam trials:
            if config['task'] == 'classification':
                try:
                    plot_ROC(seed_path)  # Plot ROC across all folds for this seed
                except Exception as e:
                    print(f"⚠️ Could not generate ROC for {seed_path}: {e}")

            tags = [
                "hyperparam_trial",
                f"hyperparam_combo:{combo_folder}",
                f"seed:{seed_folder}"
            ]

            print(f"📤 Uploading hyperparam trial: {seed_path}")
            try:
                # neptune_id = upload_experiment(seed_path, config, tags=tags)
                print("no upload eheh")
                neptune_id = None
            except Exception as e:
                print(f"❌ Failed to upload to Neptune: {e}")
                neptune_id = None
            formatted = ", ".join(combo_folder.replace("hparam_", "").split("_"))
            print(f"Formatted hyperparam combo: {formatted}")

            export_scores_to_excel(
                score_path=score_path,
                config=config,
                excel_path=excel_path_hparam,
                neptune_id= neptune_id if neptune_id else None,
                cohort='23',
                hyperparam_combo=formatted,
                mods=mods,
                single_model=False,
            )

    # 2️ Upload best final model per seed
    print("\n👑 Uploading best final models per seed...")
    for seed, path in best_models_per_seed.items():
        score_path = path
        seed_tag = f"seed:seed_{seed}" if isinstance(seed, int) else f"seed:{seed}"
        if not os.path.exists(score_path):
            print(f"⚠️ Missing final model score for seed {seed}")
            continue

        combo = best_combos_per_seed[seed]
        print(combo)
        tags = [
            "final_model",
            seed_tag,
            "best_model",
            f"best_combo:{formatted}"
        ]
        print(f"📤 Uploading final model: {path}")
        try:
            neptune_id = upload_experiment(path, config, tags=tags)
        except Exception as e:
            print(f"Failed to upload to Neptune: {e}")
            neptune_id = None

        export_scores_to_excel(
            score_path=score_path,
            config=config,
            excel_path=excel_path_final,
            neptune_id= neptune_id if neptune_id else None,
            seed=seed,
            cohort='23',
            hyperparam_combo=combo,
            mods=mods,
            single_model=True,
        )
