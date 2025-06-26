import numpy as np
import matplotlib.pyplot as plt

modalities = ['RWD\n(n. 1200)', 'RWD\nDP\n(n. 700)', 'RWD\nFM RAD\n(n. 600)', 'RWD\nPYRAD\n(n. 600)', 'RWD\nDP\nFM RAD\n(n. 230)', 'RWD\nDP\nPYRAD\n(n. 230)']

score_values = np.array(
    [0.67, 0.56, 0.58, 0.57, 0.58, 0.62]
)
score_std = np.array(
    [0.03, 0.04, 0.04, 0.06, 0.1, 0.03]
)

plt.figure(figsize=(12, 6))

plt.plot(modalities, score_values, marker='o', linestyle='-', color='#1a80bb', label='CV AUC')
plt.fill_between(modalities, score_values - score_std, score_values + score_std, color='#8cc5e3', alpha=0.3, label='Confidence interval')

for i, (val, std) in enumerate(zip(score_values, score_std)):
    plt.text(i, 0.02, f"{val:.2f} ± {std:.2f}", fontsize=10, ha='center', color='#1a80bb')

plt.title('Cross-Validation AUC')
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
# plt.show()

# save as png
plt.savefig('line_plot_albi.png', dpi=300, bbox_inches='tight')

