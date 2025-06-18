import pandas as pd
import torch
import os
import numpy as np

# Charger le fichier parquet
df = pd.read_parquet('fake_mm_surv/df.parquet')
print("Structure du DataFrame:")
print(df.head())

# Créer un dossier pour les fichiers .pt
output_dir = 'fake_mm_surv/bags'
os.makedirs(output_dir, exist_ok=True)

def convert_to_tensor(data_list, max_len):
    if data_list is None or len(data_list) == 0:
        return torch.zeros(max_len, dtype=torch.float32)
    
    # Convertir en liste si c'est un numpy array
    if isinstance(data_list, np.ndarray):
        data_list = data_list.tolist()
    
    # S'assurer que c'est une liste plate
    if isinstance(data_list[0], list):
        data_list = data_list[0]
    
    # Padding avec des zéros
    padded_list = data_list + [0] * (max_len - len(data_list))
    return torch.tensor(padded_list, dtype=torch.float32)

# Dimensions maximales pour chaque modalité
max_dims = {
    'mod1': 1,
    'mod2': 2,
    'mod3': 3
}

# Traiter chaque modalité séparément
for modality in ['mod1', 'mod2', 'mod3']:
    print(f"\nTraitement de {modality}:")
    modality_tensors = []
    
    for idx, row in df.iterrows():
        current_value = row[modality]
        
        # Debug print
        print(f"Slide {idx}, valeur: {current_value} (type: {type(current_value)})")
        
        try:
            if pd.isna(current_value).any():
                tensor = torch.zeros(max_dims[modality], dtype=torch.float32)
            else:
                tensor = convert_to_tensor(current_value, max_dims[modality])
            modality_tensors.append(tensor)
        except Exception as e:
            print(f"Erreur pour slide {idx}: {e}")
            tensor = torch.zeros(max_dims[modality], dtype=torch.float32)
            modality_tensors.append(tensor)
    
    # Empiler tous les tensors
    modality_data = torch.stack(modality_tensors)
    
    # Vérifier les données
    print(f"Shape finale de {modality}: {modality_data.shape}")
    print(f"Nombre de valeurs non nulles: {torch.count_nonzero(modality_data)}")
    
    # Sauvegarder le tensor
    output_path = os.path.join(output_dir, f'{modality}.pt')
    torch.save(modality_data, output_path)
    print(f"Sauvegarde de {modality}.pt terminée")