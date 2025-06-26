import os
import shutil
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
from .metrics_utils import calculate_classification_metrics, calculate_survival_metrics, softmax


REQUIRED_FILES = [
  # "mil_params.json",
]

REQUIRED_DIRS = [
   "attention"
]

def compute_scores(base_path, task, dry_run=True):
    """
    Processes all valid experiment folders recursively under base_path.
    base path given be like: hparam_*/seed_*/
    it only looks inside eval
    """
    print(f"🚀 Starting compute_scores in: {base_path}")
    SKIP_FOLDERS = ["attention", "models"]
    processed = 0

    for root, dirs, _ in os.walk(base_path, topdown=True):
        print(f"\n🔍 Visiting: {root}")
        if any(os.path.basename(root) == skip for skip in SKIP_FOLDERS):
            print(f"⛔ Skipped system folder: {root}")
            continue
        if "eval" not in root:
            print(f"⏭️ Skipped (no 'eval' in path): {root}")
            continue

        if folder_has_required_content(root):
            print(f"📂 Valid experiment found at: {root}")
            try:
                process_predictions(root, task)
                processed += 1
            except Exception as e:
                print(f"⚠️ Error processing {root}: {e}")
        elif not dry_run:
            if root != base_path:
                print(f"🗑️ Deleting invalid folder: {root}")
                shutil.rmtree(root, ignore_errors=True)
                print(f"   ✅ Deleted: {root}")
            else:
                print("🚫 Not deleting base_path itself.")

    if processed == 0:
        print(f"\n❗ No valid experiments found under: {base_path}")
    else:
        print(f"\n✅ Total valid experiments processed: {processed}")


def folder_has_required_content(folder_path):
    """Check for required files and folders in the given folder path."""
    print(f"\n🧪 Checking folder: {folder_path}")
    
    for filename in REQUIRED_FILES:
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            print(f"❌ Missing file: {full_path}")
            return False
        else:
            print(f"✅ Found file: {full_path}")
    
    for dirname in REQUIRED_DIRS:
        dir_path = os.path.join(folder_path, dirname)
        if not os.path.isdir(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        else:
            print(f"✅ Found directory: {dir_path}")

    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    print(f"🖼️ PNG count: {len(png_files)}")
    if not (0 <= len(png_files) <= 6):
        print(f"❌ PNG file count invalid in: {folder_path}")
        return False

    parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    print(f"📊 Parquet count: {len(parquet_files)}")
    if not parquet_files:
        print(f"❌ No parquet files found in: {folder_path}")
        return False

    print(f"✅ Folder is valid: {folder_path}")
    return True


def find_deepest_valid_folder(folder_path):
    """Cerca ricorsivamente la cartella valida più profonda."""
    print(f"\n🔎 Searching deepest valid folder in: {folder_path}")
    for root, dirs, files in os.walk(folder_path, topdown=True):
        print(f"🔍 Inspecting: {root}")
        if folder_has_required_content(root):
            print(f"✅ Deepest valid folder found: {root}")
            return root
    print("❌ No valid folder found.")
    return None

def process_predictions(folder_path, task):
    """Processa i file delle predizioni e salva gli score."""
    test_file = os.path.join(folder_path, "predictions.parquet")
    train_file = os.path.join(folder_path, "predictions_train.parquet")

    if task == "classification":
        if os.path.exists(test_file):
            df_test = pd.read_parquet(test_file)
            calculate_classification_metrics(df_test, os.path.join(folder_path, "scores_test.csv"))
        if os.path.exists(train_file):
            df_train = pd.read_parquet(train_file)
            calculate_classification_metrics(df_train, os.path.join(folder_path, "scores_train.csv"))

    elif task == "survival":
        print(f"🩺 Processing survival task in: {folder_path}")
        
        if os.path.exists(test_file):
            print(f"📥 Found test predictions file: {test_file}")
            test_data = pd.read_parquet(test_file)
            print(f"📊 Loaded test data: {len(test_data)} rows")
            
            try:
                scores_test = calculate_survival_metrics(None, test_file)
                scores_test["N_TEST"] = len(test_data)
                out_path = os.path.join(folder_path, "scores_test.csv")
                pd.DataFrame([scores_test]).to_csv(out_path, index=False)
                print(f"✅ Saved survival test scores to: {out_path}")
            except Exception as e:
                print(f"❌ Error while processing test survival scores: {e}")
        else:
            print(f"⚠️ Test file not found: {test_file}")
        
        if os.path.exists(train_file):
            print(f"📥 Found train predictions file: {train_file}")
            train_data = pd.read_parquet(train_file)
            print(f"📊 Loaded train data: {len(train_data)} rows")
            
            try:
                scores_train = calculate_survival_metrics(train_file, None)
                scores_train["N_TRAIN"] = len(train_data)
                out_path = os.path.join(folder_path, "scores_train.csv")
                pd.DataFrame([scores_train]).to_csv(out_path, index=False)
                print(f"✅ Saved survival train scores to: {out_path}")
            except Exception as e:
                print(f"❌ Error while processing train survival scores: {e}")
        else:
            print(f"⚠️ Train file not found: {train_file}")

