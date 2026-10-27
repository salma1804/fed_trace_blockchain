# anchored_federation.py fait tourner les 10 rounds de fédération et, à chaque round, 
# écrit dans le registre le hash de chaque contribution avant l'agrégation, puis le hash 
# du modèle global (avec ses métriques et son modèle parent) 
# après, en rangeant les poids réels hors chaîne.


import json
import os
from pathlib import Path
from federation import local_train, CLINIQUES, N_ROUNDS, LOCAL_EPOCHS
from aggregation import weighted_average
from fingerprint import fingerprint_weights
from ledger import Ledger
from model import SoftmaxRegression, load_dataset

OFFCHAIN = Path("offchain/weights")
TRAIN_PARAMS = {"local_epochs": LOCAL_EPOCHS, "lr": 0.1, "batch_size": 64,
                "l2": 1e-4, "privacy": "none"}


def save_offchain(weights):
    """Sauve les poids hors chaîne, nommés par leur empreinte. Renvoie l'empreinte."""
    h = fingerprint_weights(weights)
    OFFCHAIN.mkdir(parents=True, exist_ok=True)
    (OFFCHAIN / f"{h}.json").write_text(json.dumps([float(x) for x in weights]))
    return h


def run_anchored(ledger, n_rounds=N_ROUNDS, aggregate=weighted_average,
                 strategy="weighted_average", attacks=None, verbose=True):
    """aggregate(updates) -> (poids, ids_retenus, ids_écartés)
       attacks = {clinic_id: fonction(poids_locaux, poids_de_départ) -> poids_empoisonnés}"""
    attacks = attacks or {}
    Xv, yv = load_dataset("val_numeric")

    # Round 0 : modèle initial = racine de l'arbre de provenance
    w = SoftmaxRegression().get_weights()
    parent_hash = save_offchain(w)
    b = ledger.append("global_model", {
        "round": 0, "model_hash": parent_hash, "strategy": "init",
        "retained": [], "discarded": [], "metrics": {}, "parent_hash": None})
    if verbose:
        print(f"Round  0 | modèle initial ancré  | bloc {b['index']} | {parent_hash[:12]}...")

    for r in range(1, n_rounds + 1):
        base = w
        # 1. Entraînement local (une clinique malveillante peut empoisonner sa mise à jour)
        updates = [local_train(c, base, r) for c in CLINIQUES]
        for u in updates:
            if u["clinic_id"] in attacks:
                u["weights"] = attacks[u["clinic_id"]](u["weights"], base)

        # 2. Ancrage des contributions AVANT l'agrégation (empoisonnées ou non)
        for u in updates:
            ch = save_offchain(u["weights"])
            b = ledger.append("contribution", {
                "round": r, "clinic_id": u["clinic_id"], "n_samples": u["n_samples"],
                "weights_hash": ch, "base_model_hash": parent_hash,
                "params": TRAIN_PARAMS})
            if verbose:
                print(f"Round {r:2d} | contribution clinique {u['clinic_id']} | bloc {b['index']} | {ch[:12]}...")

        # 3. Agrégation (renvoie aussi le sort de chaque contribution)
        w, retained, discarded = aggregate(updates)

        # 4. Ancrage du modèle global, des contributions retenues ET écartées
        m = SoftmaxRegression()
        m.set_weights(w)
        metrics = m.evaluate(Xv, yv)
        mh = save_offchain(w)
        b = ledger.append("global_model", {
            "round": r, "model_hash": mh, "strategy": strategy,
            "retained": retained, "discarded": discarded,
            "metrics": metrics, "parent_hash": parent_hash})
        if verbose:
            print(f"Round {r:2d} | MODELE GLOBAL        | bloc {b['index']} | {mh[:12]}... | acc={metrics['accuracy']:.4f}")
            if discarded:
                print(f"Round {r:2d} | REJET ANCRÉ : clinique(s) {discarded} écartée(s)")
        parent_hash = mh

    return w


if __name__ == "__main__":
    if os.path.exists("ledger.jsonl"):
        os.remove("ledger.jsonl")          # nouvelle exécution propre
    L = Ledger("ledger.jsonl")
    run_anchored(L)
    ok, msg = L.verify_chain()
    print("\nBlocs dans le registre :", len(L))
    print("Chaîne valide :", ok)