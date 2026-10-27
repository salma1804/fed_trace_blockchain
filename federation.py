# [federation.py : local_train kol clinique mch idawar fl local w yatinikn les weight finale bad namlo lagéregation msh nwn3 serveur global f kol round 
import json
import numpy as np
from model import SoftmaxRegression, load_dataset

CLINIQUES = [1, 2, 3]
N_ROUNDS = 10
LOCAL_EPOCHS = 5


def local_train(clinic_id, global_weights, round_no):
    """Étape 3 : une clinique s'entraîne sur ses propres données."""
    X, y = load_dataset(f"clinique_{clinic_id}")
    model = SoftmaxRegression()
    model.set_weights(global_weights)
    # graine déterministe, différente pour chaque (round, clinique)
    model.fit(X, y, epochs=LOCAL_EPOCHS, seed=1000 * round_no + clinic_id)
    return {
        "clinic_id": clinic_id,
        "weights": model.get_weights(),
        "n_samples": len(X),
    }


def aggregate_weighted(updates):
    """Étape 4 : moyenne pondérée par le nombre d'échantillons."""
    total = sum(u["n_samples"] for u in updates)
    agg = sum(u["n_samples"] * u["weights"] for u in updates) / total
    return agg


def run_federation(n_rounds=N_ROUNDS, aggregate=aggregate_weighted):
    """Étape 5 : on répète entraînement local + agrégation."""
    Xv, yv = load_dataset("val_numeric")
    global_weights = SoftmaxRegression().get_weights()  # départ : zéros
    history = []

    for r in range(1, n_rounds + 1):
        updates = [local_train(c, global_weights, r) for c in CLINIQUES]
        global_weights = aggregate(updates)

        m = SoftmaxRegression()
        m.set_weights(global_weights)
        metrics = m.evaluate(Xv, yv)
        print(f"Round {r:2d} | val acc = {metrics['accuracy']:.4f} | f1 = {metrics['f1_macro']:.4f}")

        history.append({
            "round": r,
            "contributions": [
                {"clinic_id": u["clinic_id"], "n_samples": u["n_samples"],
                 "weights": u["weights"].tolist()} for u in updates
            ],
            "global_weights": global_weights.tolist(),
            "metrics": metrics,
        })
    return global_weights, history


if __name__ == "__main__":
    final_w, history = run_federation()

    # Évaluation finale sur le jeu de test
    Xt, yt = load_dataset("test_numeric")
    m = SoftmaxRegression()
    m.set_weights(final_w)
    print("\nModèle fédéré  | TEST :", m.evaluate(Xt, yt))

    # Référence : modèle centralisé (les 3 jeux regroupés)
    data = [load_dataset(f"clinique_{c}") for c in CLINIQUES]
    Xall = np.vstack([d[0] for d in data])
    yall = np.concatenate([d[1] for d in data])
    central = SoftmaxRegression().fit(Xall, yall, epochs=N_ROUNDS * LOCAL_EPOCHS)
    print("Modèle centralisé | TEST :", central.evaluate(Xt, yt))

    # Sauvegarde de l'historique (servira pour le hash et le registre)
    with open("history.json", "w") as f:
        json.dump(history, f)
    print("\nhistory.json sauvegardé")