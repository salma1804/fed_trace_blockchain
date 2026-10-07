# split_cliniques.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

SEED = 42
DATA_DIR = Path(".")            # dossier où se trouvent tes CSV
OUT_DIR = Path("cliniques")
OUT_DIR.mkdir(exist_ok=True)

# 1. Charger les données d'entraînement
df = pd.read_csv(DATA_DIR / "train_numeric.csv")
print("Shape :", df.shape)
print("Colonnes :", list(df.columns))
print(df["label"].value_counts().sort_index(), "\n")

# 2. Découpage stratifié en 3 cliniques
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for i, (_, idx) in enumerate(skf.split(df, df["label"]), start=1):
    clinique = df.iloc[idx].reset_index(drop=True)
    clinique.to_csv(OUT_DIR / f"clinique_{i}.csv", index=False)
    print(f"Clinique {i} : {len(clinique)} lignes")
    print(clinique["label"].value_counts().sort_index().to_dict())

# 3. Copier val et test tels quels (globaux)
for name in ["val_numeric", "test_numeric"]:
    pd.read_csv(DATA_DIR / f"{name}.csv").to_csv(OUT_DIR / f"{name}.csv", index=False)

print("\nFichiers créés dans", OUT_DIR.resolve())