#  Le diagnostic  [diagnosis.py]
#   On donne un patient au modele. Il repond (par exemple
#   "Seropositive, 95%") et il ECRIT UNE EXPLICATION en francais :
#   les 3 informations qui ont le plus compte. Exemple :
#     "Diagnostic : PR seropositive (probabilite 0.95). Facteurs les
#      plus influents : Anti_CCP = 145 (renforce), RF = 80
#      (renforce), CRP = 12 (attenue)."
#   Puis on ecrit dans le cahier :
#     - le code-barres du modele utilise ;
#     - le code-barres de l'explication ;
#     - un faux nom pour le patient (un "pseudonyme", par exemple
#       CAS-4B406AF682), jamais son vrai nom.
#   Le texte de l'explication, lui, reste dans le coffre (offchain/).

import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from fingerprint import fingerprint_text, fingerprint_weights
from model import FEATURES, load_scaler, _clean

OFF = Path("offchain")
CLASS_NAMES = {0: "Sain", 1: "PR séropositive", 2: "PR séronégative"}


def pseudonymize(patient_ref) -> str:
    """Pseudonyme de cas : hash salé, impossible à relier au patient sans le sel."""
    OFF.mkdir(exist_ok=True)
    p = OFF / "pseudo_salt.txt"
    if not p.exists():
        p.write_text(os.urandom(16).hex())
    salt = p.read_text()
    return "CAS-" + hashlib.sha256((salt + str(patient_ref)).encode()).hexdigest()[:10].upper()


def prepare_case(raw: dict):
    """Même prétraitement que pour l'entraînement : imputation médiane + normalisation."""
    sc = load_scaler()
    df = _clean(pd.DataFrame([raw]))
    vals = df[FEATURES].astype(float).iloc[0]
    filled = vals.fillna(pd.Series(sc["median"], index=FEATURES))
    x = (filled.to_numpy() - np.array(sc["mean"])) / np.array(sc["std"])
    return x.reshape(1, -1), vals


def build_explanation(model, x, raw_vals, pred, proba) -> str:
    W = model.W
    others = [k for k in range(W.shape[1]) if k != pred]
    contrib = x[0] * (W[:, pred] - W[:, others].mean(axis=1))
    order = np.argsort(-np.abs(contrib), kind="stable")[:3]
    parts = []
    for j in order:
        name = FEATURES[j]
        v = raw_vals[name]
        val = "non renseigné" if pd.isna(v) else f"{v:g}"
        effet = "renforce" if contrib[j] > 0 else "atténue"
        parts.append(f"{name} = {val} ({effet})")
    return (f"Diagnostic : {CLASS_NAMES[pred]} (probabilité {proba:.2f}). "
            f"Facteurs les plus influents : " + ", ".join(parts) + ".")


def diagnose(ledger, model, raw_case: dict, patient_ref, clinic_id=1):
    x, raw_vals = prepare_case(raw_case)
    p = model.predict_proba(x)[0]
    pred = int(p.argmax())
    proba = float(p[pred])
    text = build_explanation(model, x, raw_vals, pred, proba)

    case = pseudonymize(patient_ref)
    model_hash = fingerprint_weights(model.get_weights())   # hash du modèle RÉELLEMENT utilisé
    expl_hash = fingerprint_text(text)

    folder = OFF / "explanations"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{expl_hash}.json").write_text(
        json.dumps({"case": case, "text": text}, ensure_ascii=False), encoding="utf-8")

    block = ledger.append("diagnosis", {
        "case": case, "clinic_id": clinic_id, "model_hash": model_hash,
        "explanation_hash": expl_hash, "predicted_class": pred,
        "probability": round(proba, 4), "validation_status": "pending_review"})
    return {"case": case, "prediction": CLASS_NAMES[pred], "probability": round(proba, 4),
            "explanation": text, "block": block["index"]}


def verify_explanation(ledger, case):
    """Recalcule le hash de l'explication stockée et le compare à celui ancré."""
    blocks = ledger.find("diagnosis", case=case)
    if not blocks:
        return False, "Diagnostic introuvable dans le registre"
    h = blocks[-1]["payload"]["explanation_hash"]
    path = OFF / "explanations" / f"{h}.json"
    if not path.exists():
        return False, "Explication absente du stockage hors chaîne"
    text = json.loads(path.read_text(encoding="utf-8"))["text"]
    if fingerprint_text(text) != h:
        return False, "FALSIFICATION : l'explication a été modifiée après son ancrage"
    return True, "Explication intacte"


def trace(ledger, case):
    """Remonte du diagnostic jusqu'aux cliniques contributrices puis au modèle initial."""
    d = ledger.find("diagnosis", case=case)[-1]
    h = d["payload"]["model_hash"]
    lines = [f"Diagnostic {case} (bloc {d['index']}, clinique émettrice {d['payload']['clinic_id']})"]
    while h is not None:
        found = ledger.find("global_model", model_hash=h)
        if not found:
            lines.append("  !! modèle absent du registre : NON CERTIFIÉ")
            break
        m = found[0]
        p = m["payload"]
        lines.append(f"  └─ modèle global round {p['round']} (bloc {m['index']}, {p['strategy']}) {h[:12]}...")
        if p["round"] > 0:
            for c in ledger.find("contribution", round=p["round"]):
                if c["payload"]["base_model_hash"] == p["parent_hash"]:
                    lines.append(f"       ├─ contribution clinique {c['payload']['clinic_id']} "
                                 f"(bloc {c['index']}, {c['payload']['n_samples']} patients)")
        h = p["parent_hash"]
    return lines