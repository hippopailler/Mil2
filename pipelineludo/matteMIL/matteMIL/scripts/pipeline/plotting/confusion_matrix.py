import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Path to the parquet file

path = 'cross_validation/mil/os_months_24/classification/cross_validation/hypothesis_driven/pyrad-noimp/rwd_radpy_dp/seed_0/fold_GHD/predictions.parquet'


# -1: 00000-mb_attention_mil
# -2: eval
# -3: seed_0
# -4: rwd
# -5: pyrad-noimp
# -6: hypothesis_driven
# -7: standard
# -8: classification
# -9: DCR
# -10: mil

# Extract specific folder names for title
path_parts = Path(path).parts
title_parts = [
    path_parts[-10],  # DCR
    path_parts[-8],   # standard
    path_parts[-6],   # pyrad-noimp
    path_parts[-5]    # rwd
]
title = ' - '.join(title_parts)
confusion_matrix_name = '_'.join(title_parts)

# in title replace the word 'standard with 'test'
title = title.replace('standard', 'test')

# confusion_matrix_name is the title with the word 'standard' replaced with 'test'
confusion_matrix_name = confusion_matrix_name.replace('standard', 'test')

# Load the parquet file
df = pd.read_parquet(path)

# Apply softmax to y_pred0 and y_pred1
def softmax(x):
    """Apply softmax function to convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Stack y_pred0 and y_pred1 into a 2D array for softmax
logits = np.column_stack([df['y_pred0'], df['y_pred1']])

# Apply softmax
probabilities = softmax(logits)

# Get predicted class (class with higher probability)
y_pred = np.argmax(probabilities, axis=1)

# Create confusion matrix
cm = confusion_matrix(df['y_true'], y_pred)

# Create the plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.title(f'{title}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the plot as PNG in the same directory as the parquet file
output_path = Path(path).parent / f'ConMat_{confusion_matrix_name}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Confusion matrix saved to: {output_path}")
print(f"Confusion Matrix:")
print(cm)
print(f"\nAccuracy: {np.trace(cm) / np.sum(cm):.3f}")



