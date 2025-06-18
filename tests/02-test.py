


from slideflow.util import prepare_multimodal_mixed_bags




# Chemins des données
bags_dir ='bags'
df_path = 'tests/features/fake_mm_surv/df.parquet'    

# Lecture des features depuis le parquet
#df_features = pd.read_parquet('tests/features/fake_mm_surv/df.parquet')

prepare_multimodal_mixed_bags(
    path=df_path,        # Chemin vers le fichier de features
    bags_path=bags_dir   # Dossier où sauvegarder les bags
)