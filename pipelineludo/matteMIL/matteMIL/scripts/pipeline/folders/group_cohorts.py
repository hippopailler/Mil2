import os
import shutil
from collections import defaultdict

def group_folders_by_suffix_and_seed(base_path):
    print("\n📂 Scanning base path:", base_path)

    group_seed_center_map = defaultdict(lambda: defaultdict(list))

    for cross_val_folder in os.listdir(base_path):
        cross_val_path = os.path.join(base_path, cross_val_folder)
        if not os.path.isdir(cross_val_path): continue

        for fold_folder in os.listdir(cross_val_path):
            fold_path = os.path.join(cross_val_path, fold_folder)
            if not os.path.isdir(fold_path): continue

            parts = fold_folder.split("_")
            if len(parts) < 3 or not parts[-1].isdigit(): continue

            center = parts[1]
            group_number = parts[-1]  # '2' or '3'

            for seed_folder in os.listdir(fold_path):
                seed_path = os.path.join(fold_path, seed_folder)
                if not os.path.isdir(seed_path) or not seed_folder.startswith("seed_"):
                    continue

                # 1️⃣ COPY to group_2 or group_3
                group_folder = f"group_{group_number}_{seed_folder}"
                target_path = os.path.join(base_path, group_folder, fold_folder, seed_folder)

                if not os.path.exists(target_path):
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copytree(seed_path, target_path)
                    print(f"✅ Copied {seed_path} ➜ {target_path}")
                else:
                    print(f"⚠️ Skipped existing: {target_path}")

                # track for later
                group_seed_center_map[seed_folder][center].append(seed_path)

                # 2️⃣ COPY to group_23 (always)
                group23_path = os.path.join(base_path, f"group_23_{seed_folder}", fold_folder, seed_folder)
                if not os.path.exists(group23_path):
                    os.makedirs(os.path.dirname(group23_path), exist_ok=True)
                    shutil.copytree(seed_path, group23_path)
                    print(f"🌍 Copied to group_23: {group23_path}")
                else:
                    print(f"⚠️ Already exists in group_23: {group23_path}")

    print("\n🏁 Grouping complete! All folds copied into group_2, group_3, and group_23.")
