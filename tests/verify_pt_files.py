import torch
import os

def verify_pt_files():
    bags_dir = 'fake_mm_surv/bags'
    
    for modality in ['mod1', 'mod2', 'mod3']:
        pt_file = os.path.join(bags_dir, f'{modality}.pt')
        
        if os.path.exists(pt_file):
            data = torch.load(pt_file)
            print(f"\nFichier {modality}.pt:")
            print(f"- Shape: {data.shape}")
            print(f"- Type: {data.dtype}")
            print(f"- Valeurs non nulles: {torch.count_nonzero(data)}")
            print(f"- Quelques valeurs: {data[0:3]}")
        else:
            print(f"\nFichier {modality}.pt non trouvé!")

if __name__ == '__main__':
    verify_pt_files()