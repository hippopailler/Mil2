import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# modality dictionary 
modality_dict = {
    "rwd": "RWD",
    "rwd_dp": "RWD\nDP",
    "rwd_radfm": "RWD\nFM RAD",
    "rwd_radpy": "RWD\nPYRAD",
    "rwd_radfm_dp": "RWD\nDP\nFM RAD",
    "rwd_radpy_dp": "RWD\nDP\nPYRAD",
    "rwd_radfm_dp_genomics": "RWD\nDP\nFM RAD\nGenomics",
    "rwd_radpy_dp_genomics": "RWD\nDP\nPYRAD\nGenomics",
}

# avg flag
AVG_FLAG = True

# path 
training_type = 'cross_validation'
sub1 = ''
sub2 = ''
path_pre = f'{training_type}/mil' # new_path
path_suf = f'survival/{training_type}/hypothesis_driven/pyrad-noimp'
outcomes = ['OS_MONTHS'] # 'DCR', 'ORR', 

for outcome in outcomes:
    path = os.path.join(sub1, sub2, path_pre, outcome, path_suf)

    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Get all folders in the path
    all_folders = [f.name for f in os.scandir(path) if f.is_dir()]

    # Check that all folder names are in the modality dictionary
    for folder in all_folders:
        if folder not in modality_dict:
            raise ValueError(f"Folder '{folder}' is not in the modality dictionary")

    # Collect data for plotting
    modalities = []
    modality_keys = []  # Keep track of keys for CSV
    score_values = []
    ci_lowers = []
    ci_uppers = []

    # Process folders in the order of the modality_dict
    for key, label in modality_dict.items():
        folder_path = os.path.join(path, key)
        
        if os.path.exists(folder_path):
            # Check for seed_0 folder
            seed_path = os.path.join(folder_path, 'seed_0')
            if not os.path.exists(seed_path):
                raise FileNotFoundError(f"seed_0 folder not found in: {folder_path}")
            
            if AVG_FLAG:
                # Use average_std_train_test_scores.csv
                csv_path = os.path.join(seed_path, 'average_std_train_test_scores.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"average_std_train_test_scores.csv not found in: {seed_path}")
                
                # Read the CSV file
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    c_index_mean = df.loc['Weighted Mean', 'C_INDEX_TEST']
                    c_index_std = df.loc['Standard Error', 'C_INDEX_TEST']
                    
                    # Calculate 95% confidence interval assuming 5 samples
                    # t_critical for 95% CI with df=4 is approximately 2.776
                    n_samples = 5
                    t_critical = 2.776
                    margin_of_error = t_critical * c_index_std / np.sqrt(n_samples)
                    
                    ci_lower = c_index_mean - margin_of_error
                    ci_upper = c_index_mean + margin_of_error
                    
                    modalities.append(label)
                    modality_keys.append(key)
                    score_values.append(c_index_mean)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                    
                except Exception as e:
                    raise ValueError(f"Error reading CSV file {csv_path}: {e}")
            else:
                # Check for eval_auc_ci.csv
                csv_path = os.path.join(seed_path, 'eval_auc_ci.csv')
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"eval_auc_ci.csv not found in: {seed_path}")
                
                # Read the CSV file
                try:
                    df = pd.read_csv(csv_path)
                    auc = df['auc'].iloc[0]
                    ci_lower = df[' ci_lower'].iloc[0]  # Note the space in column name
                    ci_upper = df[' ci_upper'].iloc[0]  # Note the space in column name
                    
                    modalities.append(label)
                    modality_keys.append(key)
                    score_values.append(auc)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                    
                except Exception as e:
                    raise ValueError(f"Error reading CSV file {csv_path}: {e}")

    # Convert to numpy arrays for plotting
    score_values = np.array(score_values)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)

    # Create the plot
    plt.figure(figsize=(12, 6))

    if AVG_FLAG:
        metric_label = 'CV C-Index'
        plot_title = 'Cross-Validation C-Index'
        y_label = 'C-Index'
    else:
        metric_label = 'CV AUC'
        plot_title = 'Cross-Validation AUC'
        y_label = 'AUC'

    plt.plot(modalities, score_values, marker='o', linestyle='-', color='#1a80bb', label=metric_label)
    plt.fill_between(modalities, ci_lowers, ci_uppers, color='#8cc5e3', alpha=0.3, label='Confidence interval')

    for i, (val, lower, upper) in enumerate(zip(score_values, ci_lowers, ci_uppers)):
        ci_range = val - lower  # Since CI is symmetric, this equals upper - val
        plt.text(i, 0.02, f"{val:.2f} ± {ci_range:.2f}", fontsize=10, ha='center', color='#1a80bb')

    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Save as PNG
    plt.savefig(os.path.join(path, f'line_plot-{sub1}-{sub2}-{training_type}-{outcome}.png'), dpi=300, bbox_inches='tight')
    
    # Save plot data as CSV
    plot_data = pd.DataFrame({
        'Modality': modality_keys,
        'Score': score_values,
        'CI_Lower': ci_lowers,
        'CI_Upper': ci_uppers
    })
    plot_data.to_csv(os.path.join(path, f'line_plot-{sub1}-{sub2}-{training_type}-{outcome}.csv'), index=False)
