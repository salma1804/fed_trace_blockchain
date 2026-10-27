import json
import os
from ledger import Ledger

for f in ["test_ledger.jsonl", "test_ledger_tampered.jsonl", "test_ledger_deleted.jsonl"]:
    if os.path.exists(f):
        os.remove(f)

# 1. Fonctionnement normal
L = Ledger("test_ledger.jsonl")
L.append("contribution", {"round": 1, "clinic_id": 1, "hash": "aaa"})
L.append("contribution", {"round": 1, "clinic_id": 2, "hash": "bbb"})
L.append("global_model", {"round": 1, "hash": "ccc"})
print("1. Chaîne valide, 4 blocs :", L.verify_chain() == (True, None) and len(L) == 4)

# 2. Redémarrage : on relit le fichier, rien n'est perdu
L2 = Ledger("test_ledger.jsonl")
print("2. Après redémarrage, 4 blocs et chaîne valide :",
      len(L2) == 4 and L2.verify_chain() == (True, None))

# 3. Modification du contenu d'un ancien bloc
lines = open("test_ledger.jsonl", encoding="utf-8").read().splitlines()
d = json.loads(lines[2])
d["payload"]["hash"] = "FAUX"
lines[2] = json.dumps(d, ensure_ascii=False)
open("test_ledger_tampered.jsonl", "w", encoding="utf-8").write("\n".join(lines) + "\n")
ok, msg = Ledger("test_ledger_tampered.jsonl").verify_chain()
print("3. Modification détectée :", (not ok), "|", msg)

# 4. Suppression d'un bloc au milieu
lines = open("test_ledger.jsonl", encoding="utf-8").read().splitlines()
del lines[2]
open("test_ledger_deleted.jsonl", "w", encoding="utf-8").write("\n".join(lines) + "\n")
ok, msg = Ledger("test_ledger_deleted.jsonl").verify_chain()
print("4. Suppression détectée :", (not ok), "|", msg)

# 5. Recherche
print("5. Recherche des contributions :", len(L2.find("contribution")) == 2)
print("   Recherche du round 1 global_model :", len(L2.find("global_model", round=1)) == 1)