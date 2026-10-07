import pandas as pd
import numpy as np
from model import FEATURES, load_scaler, _clean

for name in ["clinique_1", "val_numeric"]:
    df = pd.read_csv(f"cliniques/{name}.csv")
    print("=====", name, "=====")
    print(df.dtypes)
    print("Valeurs manquantes (brut) :")
    print(df.isna().sum())
    print("Valeurs de Gender :", df["Gender"].unique()[:10])
    print("Après nettoyage, NaN :")
    print(_clean(df)[FEATURES].isna().sum())
    print()

print("Scaler :", load_scaler())
print("Poids après entraînement contiennent NaN ?")
from model import SoftmaxRegression, load_dataset
X, y = load_dataset("clinique_1")
print("NaN dans X :", np.isnan(X).sum())
w = SoftmaxRegression().fit(X, y, epochs=30).get_weights()
print(w)