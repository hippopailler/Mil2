import torch
import os

# Chemin vers les fichiers .pt
bags_dir = 'fake_mm_surv/bags'

for modality in ['mod1', 'mod2', 'mod3']:
    file_path = os.path.join(bags_dir, f'{modality}.pt')
    try:
        data = torch.load(file_path)
        print(f"{modality}: {len(data)} entries loaded.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")