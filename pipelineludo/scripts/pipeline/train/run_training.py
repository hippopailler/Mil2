from .training_loop import train_val
from .prepare_dataset import prepare_dataset
from .utils.paths_utils import build_base_path, build_full_path, get_group
from .utils.config_utils import get_folds
import slideflow as sf
import os
from .hyperparameters_tuning.grid_runner import run_grid_search_cv
from .hyperparameters_tuning.hyperparam_tuning import pick_best_hyperparams, pick_best_final_model
from metrics.compute_scores import compute_scores
from metrics.calculate_average import compute_weighted_average
import json


# TODO check the flow of stratitfied and cross validation
def run_training(config, mods):
    """
    Executes the full training pipeline for a given modality combo.

    - If training_type == "hyperparameter_tuning":
        → Runs CV for all hyperparameter combos
        → Picks the best combo
        → Retrains on full data
        → Returns 'hyperparam_dir' and 'final_model_dir'

    - Else (standard, cross_validation, or cross_validation_stratified):
        → Trains based on training_type, seeds, and folds
        → Returns 'experiment_paths'
    """

    # 1) Build a little “prefix” list based on your flags
    prefix_parts = []
    if config.get("FILTER_SQUAMOUS") is not None:
        prefix_parts.append(f"squamous_{config['FILTER_SQUAMOUS']}")
    if config.get("FILTER_CHEMO_IMMUNO") is not None:
        prefix_parts.append(f"chemoio_{config['FILTER_CHEMO_IMMUNO']}")
    if config.get("USE_COHORT2_FILTER"):
        prefix_parts.append("cohort2")
    if config.get("FILTER_INT"):
        prefix_parts.append(f"int")
    if config.get("FILTER_PDL1"):
        prefix_parts.append(f"pdl1_{config['FILTER_PDL1']}")
    if config.get("FILTER_ALL_MODS"):
        prefix_parts.append("all_mods")

    # Add the training type to the prefix
    if config.get("training_type"):
        prefix_parts.append(config["training_type"])

    # join into something like "cohort2/foundation/squamous"
    path_prefix = os.path.join(*prefix_parts) if prefix_parts else ""
    print(f"🔍 [DEBUG] path_prefix = '{path_prefix}'")

    mod_string = "_".join([k for k, v in mods.items() if v])
    bag_path = f"bags_{mod_string}"  
  

    project_path = '.'
    P = sf.Project(project_path)

    training_type = config["training_type"]
    folds = get_folds("cross_validation", config) if training_type == "hyperparameter_tuning" else get_folds(training_type, config)
    seeds = config["seed"]

    # 3️⃣ SKIP TRAINING if the flag is set
    if config.get("SKIP_TRAINING", False):
        print("🛑 SKIPPING TRAINING as 'SKIP_TRAINING' is set to True")
        return {"status": "dataset_prepared", "bag_path": bag_path}

    
    if training_type == "hyperparameter_tuning":
        base_path = build_base_path(config, mods)
        hyperparam_path   = os.path.join(path_prefix, "mil", base_path, "hyperparam")
        final_model_path  = os.path.join(path_prefix, "mil", base_path, "final_model")

        final_results = []

        for seed in seeds:
            for fold in folds:    
                          
                 run_grid_search_cv(
                    P,
                    config=config,
                    base_results_path=hyperparam_path,
                    bag_path=bag_path,
                    folds=folds,
                    mods=mods,
                    fold=fold,
                    seed=seed,
                    training_type="cross_validation",
                )   

            # Pick best hyperparameters for the current seed            
            best_combo = pick_best_hyperparams(
                base_results_path=hyperparam_path,
                config=config
            )

            print(f"🌱 Best hyperparameters for seed {seed}: {best_combo}")
            final_seed_path = os.path.join(final_model_path, f"seed_{seed}") 

            train_val(
                P,
                config={**config, "hyper_combo": best_combo},
                mods=mods,
                fold="ALL",
                seed=seed,
                results_path=os.path.abspath(final_seed_path),
                bag_path=bag_path,
                folds=folds,
                training_type="standard"
            )

            compute_scores(final_seed_path, config["task"])
            compute_weighted_average(final_seed_path)

            final_results.append({
                "seed": seed,
                "path": final_seed_path,
                "combo": best_combo
            })

         # Pick best final model
        best_model = pick_best_final_model(final_model_path, config["task"])
        best_seed = int(best_model["seed"].replace("seed_", ""))

        # Recupera il combo direttamente dal seed
        combo_map = {item["seed"]: item["combo"] for item in final_results}
        print("combo map", combo_map)
        if best_seed not in combo_map:
            raise RuntimeError(f"Best seed {best_seed} not found in final_results!")

        best_combo = combo_map[best_seed]

        # Salva riassunto del best model
        summary = {
            "seed": best_seed,
            "score": best_model["score"],
            "metric": best_model["metric"],
            "combo": best_combo
        }

        with open(os.path.join(final_model_path, "best_model.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "hyperparam_dir": os.path.abspath(hyperparam_path),
            "final_model_dir": os.path.abspath(final_model_path),
            "best_models_per_seed": {best_seed: best_model["path"]},
            "best_combos_per_seed": {best_seed: best_combo},
        }
    
    else:
        experiment_paths = []
        base_path = build_base_path(config, mods)

        for seed in seeds:
            # Handle the base seed path
            base_seed_path = os.path.abspath(os.path.join(path_prefix, "mil", base_path, f"seed_{seed}"))
            experiment_paths.append(base_seed_path)

            print(f"🔍 [DEBUG] Base seed path = {base_seed_path}")

            # Track which group directories we've processed
            group_dirs = set()

            for fold in folds:
                # Determine the group directory for stratified CV
                group = f"group_{get_group(fold)}" if training_type == "cross_validation_stratified" else None

                # Build the full training path
                full_path = build_full_path(
                    base_path=base_path,
                    fold=fold if fold != "ALL" else None,
                    seed=seed,
                    group=group,
                )

                abs_path = os.path.abspath(os.path.join(path_prefix, "mil", full_path))
                print(f"🔍 [DEBUG] About to run train_val → results_path = {abs_path}")

                train_val(
                    P,
                    config,
                    mods,
                    fold,
                    seed,
                    results_path=abs_path,
                    bag_path=bag_path,
                    folds=folds,
                    training_type=training_type
                )  

                print(f"Training completed for seed {seed}, fold {fold} at {abs_path}")
                compute_scores(abs_path, config["task"])

                # Track group directories for average calculation
                if group:
                    group_dir = os.path.join(base_seed_path, group)
                    group_dirs.add(group_dir)

            # 📊 Compute average at the **group** or **seed** level
            if config["task"] != "standard":
                if training_type == "cross_validation_stratified":
                    for group_dir in group_dirs:
                        print(f"📝 Computing weighted average for {group_dir}")
                        compute_weighted_average(group_dir, config["task"])
                else:
                    print(f"📝 Computing weighted average for {base_seed_path}")
                    compute_weighted_average(base_seed_path, config["task"])
               

        return {"experiment_paths": experiment_paths}