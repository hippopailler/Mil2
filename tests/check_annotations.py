import pandas as pd

# Charger les annotations
ann_df = pd.read_csv('annotations/ann_mm_surv.csv')
print("Structure des annotations:")
print(ann_df.head())
print("\nColonnes:", ann_df.columns.tolist())
print("\nDistribution dataset:")
print(ann_df['dataset'].value_counts())
print("\nVérification des slides:")
print(ann_df['slide'].head())