# //nlamed les round mta les clinique f serveur global 

import numpy as np

OUTLIER_FACTOR = 10.0


def weighted_average(updates):
    """Moyenne pondérée (non robuste). Renvoie (poids, retenues, écartées)."""
    total = sum(u["n_samples"] for u in updates)
    w = sum(u["n_samples"] * u["weights"] for u in updates) / total
    return w, [u["clinic_id"] for u in updates], []


def median_with_exclusion(updates, factor=OUTLIER_FACTOR):
    """Médiane coordonnée par coordonnée, après exclusion d'une contribution aberrante.

    Avec 3 cliniques, on tolère au plus 1 clinique malveillante.
    """
    W = np.stack([u["weights"] for u in updates])
    m = np.median(W, axis=0)
    d = np.linalg.norm(W - m, axis=1)          # distance de chaque contribution à la médiane
    order = np.argsort(-d, kind="stable")      # de la plus éloignée à la plus proche

    discarded = []
    if len(updates) >= 3:
        far, second = order[0], order[1]
        if d[far] > factor * max(d[second], 1e-9):
            discarded = [int(far)]

    keep = [i for i in range(len(updates)) if i not in discarded]
    w = np.median(W[keep], axis=0)
    retained_ids = [updates[i]["clinic_id"] for i in keep]
    discarded_ids = [updates[i]["clinic_id"] for i in discarded]
    return w, retained_ids, discarded_ids


def sign_flip_attack(factor=50.0):
    """Attaque : la clinique renvoie l'inverse amplifié de sa mise à jour."""
    def attack(local_w, base_w):
        return base_w - factor * (local_w - base_w)
    return attack