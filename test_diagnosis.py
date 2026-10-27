import json
import os
import shutil
import pandas as pd
from ledger import Ledger
from model import SoftmaxRegression
from diagnosis import diagnose, verify_explanation, trace, OFF

if os.path.exists("ledger_diag.jsonl"):
    os.remove("ledger_diag.jsonl")
shutil.copy("ledger.jsonl", "ledger_diag.jsonl")
L = Ledger("ledger_diag.jsonl")
n0 = len(L)

# Chargement du modèle global final (round 10)
final = L.find("global_model", round=10)[0]["payload"]["model_hash"]
model = SoftmaxRegression()
model.set_weights(json.loads((OFF / "weights" / f"{final}.json").read_text()))

# Trois diagnostics sur des cas du jeu de test
cases = pd.read_csv("cliniques/test_numeric.csv")
results = []
for i in range(3):
    raw = cases.iloc[i].to_dict()
    res = diagnose(L, model, raw, f"test-row-{i}", clinic_id=1)
    results.append(res)
    print(f"{res['case']} | vrai label = {int(raw['label'])} | prédit = {res['prediction']} ({res['probability']})")
    print("   ", res["explanation"])

# --- Vérifications ---
print()
print("1. 3 diagnostics ancrés :", len(L) == n0 + 3 and len(L.find("diagnosis")) == 3)
d = L.find("diagnosis")[-1]["payload"]
print("2. Hash du modèle ancré = modèle du round 10 :", d["model_hash"] == final)

case = results[-1]["case"]
print("3. Explication intacte :", verify_explanation(L, case)[0])

# 4. Falsification : on retire les facteurs de l'explication stockée hors chaîne
p = OFF / "explanations" / f"{d['explanation_hash']}.json"
original = p.read_text(encoding="utf-8")
data = json.loads(original)
data["text"] = data["text"].split(" Facteurs")[0]
p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
ok, msg = verify_explanation(L, case)
print("4. Falsification détectée :", (not ok), "|", msg)

# 5. Restauration
p.write_text(original, encoding="utf-8")
print("5. Après restauration, intacte :", verify_explanation(L, case)[0])

# 6. Confidentialité : ni explication ni valeurs médicales dans le registre
contenu = open("ledger_diag.jsonl", encoding="utf-8").read()
print("6. Aucun texte d'explication sur la chaîne :",
      "Facteurs" not in contenu and "Anti_CCP" not in contenu)

# 7. Chemin de provenance
print("\n7. Chemin de provenance du dernier diagnostic :")
lines = trace(L, case)
print("\n".join(lines))
print("   Atteint le round 0 :", any("round 0" in l for l in lines))