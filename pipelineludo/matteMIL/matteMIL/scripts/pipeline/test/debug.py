import sys
import traceback
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train.hyperparameters_tuning.hyperparam_tuning import pick_best_hyperparams
# CONFIGURATION ✨
base_results_path = "/Users/ludole/Desktop/PHD/matteMIL/mil/OS_MONTHS/survival/hyperparameter_tuning/hypothesis_driven/foundation-noimp/rwd_rad_dp_genomics/hyperparam"
config = {
    "task": "survival",  # or "classification"
}

# RUNNING & DEBUGGING 🐛
try:
    best_hparams = pick_best_hyperparams(base_results_path, config)
    print("\n🎉 Best Hyperparameters Found:")
    print(best_hparams)

except Exception as e:
    print("❌ Something went wrong during hyperparameter search!")
    print("-" * 60)
    traceback.print_exc(file=sys.stdout)
    print("-" * 60)