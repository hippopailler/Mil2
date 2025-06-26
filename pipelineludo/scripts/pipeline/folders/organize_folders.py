import os
import shutil
import re

def extract_components(folder_name, training_type):
    '''
    Extracts components from folder name based on naming convention.

    Returns:
        task (str), approach (str), subtype (from config), source_imp (str), modality_string (str)
    '''
    print(f"🔍 Extracting components from {folder_name}")
    match = re.match(r'^\d+-\s*(.+)', folder_name)
    if not match:
        return None

    parts = match.group(1).split('-')
    if len(parts) < 6:
        return None


    outcome = parts[1]  # outcome
    print(f"🧩 Task: {outcome}")

    task = parts[2]  # survival / classification
    print(f"🧩 Task: {task}")
    
    approach = parts[3]  # data-driven / hypothesis-driven
    print(f"🧩 Approach: {approach}")
    
    subtype = training_type  # pulled from config now
    print(f"🧩 Training Type (subtype): {subtype}")

    source = parts[4]
    print(f"🧩 Source: {source}")
    
    imp = parts[5]
    print(f"🧩 Imputation: {imp}")
    
    source_imp = f"{source}-{imp}"
    print(f"🧩 Source + Imputation: {source_imp}")

    mod_str = '-'.join(parts[6:])  # rwd_dp_rad

    return task, approach, subtype, source_imp, mod_str

def organize_folders(base_directory, config):
    '''
    Organizes unstructured experiment folders into a hierarchical format based on their names.

    Expected folder naming convention:
        "XX- {outcome}-{task}-{approach}-{data_type}-{source}-{imp}-{modalities}"
    Example:
        "12- os_months_6-survival-cross_validation-data-driven-foundation-imp-rwd_dp_rad"

    Extracted structure:
        base_directory/                             
         └── <outcome>/                              (e.g., os_months_6 )
            └── <task>/                              (e.g., survival, classification)
              └── <training_type>/                   (e.g., cross_validation, holdout)          
                └── <approach>/                      (e.g., data-driven, hypothesis-driven)
                        └── <source>-<imp>/          (e.g., foundation-imp)
                            └── <modalities>/        (e.g., rwd_dp_rad)
                                └── [original experiment folder contents]

    Parameters:
        base_directory (str): The root path containing unorganized experiment folders.

    Notes:
        - Only folders matching the expected pattern will be moved.
        - Folder contents are moved using shutil.move.
        - The function prints the source and destination for each operation.
    '''
    training_type = config["training_type"]
    all_folders = [f for f in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, f))]

    for folder in all_folders:
        components = extract_components(folder, training_type)
        if not components:
            print(f"⏭ Skipping: {folder} (not matching expected pattern)")
            continue

        task, approach, subtype, source_imp, mod_str = components
        print(f"🧩 Components: {task}, {approach}, {subtype}, {source_imp}, {mod_str}")

        destination = os.path.join(base_directory, task, approach, subtype, source_imp, mod_str)
        source_path = os.path.join(base_directory, folder)
        print(f"📂 Source: {source_path}")
        print(f"📂 Destination: {destination}")

        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)

        # Sposta tutto il contenuto dentro la nuova destinazione
        print(f"📂 Moving {folder} → {destination}")
        shutil.move(source_path, destination)
    
    print(f"✅ folder organization completed. moved {len(all_folders)} folders to {destination}.")

