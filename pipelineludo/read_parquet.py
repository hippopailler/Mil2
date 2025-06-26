import pandas as pd

# Paths to the generated parquet files
parquet_files = [
    "/Users/ludole/Desktop/PHD/alex code/features/model_train_6.parquet",
]

# Check each parquet file
for file in parquet_files:
    print(f"\n[INFO] Inspecting {file}...")
    try:
        # Load the Parquet file
        data = pd.read_parquet(file)

        # Display basic info
        print(f"[INFO] Number of instances (rows) in {file}: {len(data)}")
        print(f"[INFO] First 5 rows of {file}:\n{data.head()}")
        print(f"[INFO] Columns in {file}: {data.columns.tolist()}")

        # Check unique slide names to ensure they are not numeric
        if 'slide' in data.columns:
            unique_slides = data['slide'].unique()
            print(f"[INFO] Number of unique slides: {len(unique_slides)}")
            print(f"[INFO] Unique slides (first 10 shown): {unique_slides[:10]}")
        else:
            print("[WARNING] 'slide' column not found.")

    except Exception as e:
        print(f"[ERROR] Could not read {file}: {e}")
