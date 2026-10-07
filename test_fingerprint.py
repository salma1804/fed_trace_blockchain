import json
import numpy as np
from fingerprint import fingerprint_weights, fingerprint_text
from federation import run_federation

# 1. Mêmes poids => même hash
w = np.arange(21, dtype=np.float64) / 10
h = fingerprint_weights(w)
print("Hash :", h)
print("1. Même poids, même hash :", h == fingerprint_weights(w.copy()))

# 2. Un changement minuscule => hash différent
w2 = w.copy()
w2[5] += 1e-12
print("2. Changement de 1e-12 détecté :", h != fingerprint_weights(w2))

# 3. -0.0 et 0.0 donnent le même hash
a, b = np.zeros(21), np.zeros(21)
b[0] = -0.0
print("3. -0.0 == 0.0 :", fingerprint_weights(a) == fingerprint_weights(b))

# 4. NaN refusé
try:
    fingerprint_weights(np.full(21, np.nan))
    print("4. NaN refusé : False")
except ValueError:
    print("4. NaN refusé : True")

# 5. Le hash survit à l'écriture/lecture JSON (history.json)
w3 = json.loads(json.dumps(w.tolist()))
print("5. Stable après JSON :", fingerprint_weights(w3) == h)

# 6. Deux fédérations complètes identiques => même hash final
w_a, _ = run_federation(n_rounds=3)
w_b, _ = run_federation(n_rounds=3)
print("6. Fédération reproductible :", fingerprint_weights(w_a) == fingerprint_weights(w_b))

# 7. Texte
print("7. Texte modifié détecté :",
      fingerprint_text("RF élevé") != fingerprint_text("RF élevé."))