# test_model.py
import numpy as np
from model import SoftmaxRegression, load_dataset

X1, y1 = load_dataset("clinique_1")
Xv, yv = load_dataset("val_numeric")
Xt, yt = load_dataset("test_numeric")
print("Clinique 1 :", X1.shape, "| val :", Xv.shape, "| test :", Xt.shape)

m1 = SoftmaxRegression().fit(X1, y1, epochs=30)
print("Val  :", m1.evaluate(Xv, yv))
print("Test :", m1.evaluate(Xt, yt))

# Déterminisme : deux entraînements identiques => poids strictement identiques
m2 = SoftmaxRegression().fit(X1, y1, epochs=30)
print("Poids identiques ?", np.array_equal(m1.get_weights(), m2.get_weights()))
print("Taille du vecteur de poids :", m1.get_weights().shape)  # attendu : (21,)