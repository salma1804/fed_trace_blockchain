# Question du cahier des charges	Où benchmark.py y répond
# Combien de temps l’ancrage ajoute-t-il par round ?	Section 1
# Combien d’octets sont écrits par round, et est-ce constant si le modèle grossit ?	Sections 1 et 3
# Combien de temps faut-il pour reconstituer la provenance ?	Section 4
# Combien de falsifications sont détectées ?	Section 5
# Combien de modèles légitimes la porte refuse-t-elle à tort ?	Section 5
# À quelle vitesse le registre grossit-il ?	Section 4




import json
import os
import statistics
import time

import numpy as np
import pandas as pd

from aggregation import weighted_average
from anchored_federation import run_anchored, save_offchain, OFFCHAIN
from diagnosis import OFF, diagnose, verify_explanation, trace
from federation import local_train, CLINIQUES
from fingerprint import fingerprint_weights
from gate import certify_model
from ledger import Ledger
from model import SoftmaxRegression, load_dataset

RAW = pd.read_csv("cliniques/test_numeric.csv").iloc[0].to_dict()
XV, YV = load_dataset("val_numeric")
RESULTS = {}
TRAIN_PARAMS = {"local_epochs": 5, "lr": 0.1, "batch_size": 64, "l2": 1e-4, "privacy": "none"}


# ------------------------------------------------------------------ utilitaires
def rm(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def med(vals):
    return statistics.median(vals)


def mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def load_model(L, r):
    h = L.find("global_model", round=r)[0]["payload"]["model_hash"]
    m = SoftmaxRegression()
    m.set_weights(json.loads((OFFCHAIN / f"{h}.json").read_text()))
    return m


def timed_ms(fn, repeats):
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return med(out)


# ------------------------------------------------------------------ 1. configurations
def one_run(mode, n_rounds=10):
    """mode : none (fédération seule), models (ancrage des modèles), full (contributions + modèles)."""
    path = "bench_cfg.jsonl"
    rm(path)
    L = Ledger(path) if mode != "none" else None
    t0 = time.perf_counter()
    w = SoftmaxRegression().get_weights()
    parent, off_bytes, size0 = None, 0, 0
    if L:
        parent = save_offchain(w)
        off_bytes += len(json.dumps([float(x) for x in w]))
        L.append("global_model", {"round": 0, "model_hash": parent, "strategy": "init",
                                  "retained": [], "discarded": [], "metrics": {}, "parent_hash": None})
        size0 = os.path.getsize(path)
    for r in range(1, n_rounds + 1):
        base = w
        updates = [local_train(c, base, r) for c in CLINIQUES]
        if mode == "full":
            for u in updates:
                L.append("contribution", {
                    "round": r, "clinic_id": u["clinic_id"], "n_samples": u["n_samples"],
                    "weights_hash": save_offchain(u["weights"]), "base_model_hash": parent,
                    "params": TRAIN_PARAMS})
                off_bytes += len(json.dumps([float(x) for x in u["weights"]]))
        w, retained, discarded = weighted_average(updates)
        m = SoftmaxRegression()
        m.set_weights(w)
        metrics = m.evaluate(XV, YV)
        if L:
            mh = save_offchain(w)
            off_bytes += len(json.dumps([float(x) for x in w]))
            L.append("global_model", {"round": r, "model_hash": mh, "strategy": "weighted_average",
                                      "retained": retained, "discarded": discarded,
                                      "metrics": metrics, "parent_hash": parent})
            parent = mh
    elapsed = time.perf_counter() - t0
    ledger_bytes = (os.path.getsize(path) - size0) if L else 0
    return elapsed, ledger_bytes, off_bytes


def exp_configs(repeats=7, n_rounds=10):
    modes = [("none", "Fédération seule"),
             ("models", "Fédération + ancrage des modèles"),
             ("full", "Fédération + ancrage contributions + modèles")]
    one_run("full", n_rounds)                                  # échauffement, ignoré
    data = {m: {"t": [], "led": [], "off": []} for m, _ in modes}
    for _ in range(repeats):                                   # modes entrelacés : moins de dérive
        for m, _ in modes:
            t, led, off = one_run(m, n_rounds)
            data[m]["t"].append(t / n_rounds * 1000)
            data[m]["led"].append(led / n_rounds)
            data[m]["off"].append(off / n_rounds)
    res = {}
    base = statistics.mean(data["none"]["t"])
    print(f"\n1. SURCOÛT TEMPOREL ET VOLUME ÉCRIT PAR ROUND ({n_rounds} rounds, {repeats} exécutions)")
    print(f"{'Configuration':<48}{'ms/round':>9}{'écart':>7}{'surcoût':>9}{'registre':>10}{'hors chaîne':>13}")
    for m, label in modes:
        mean, sd = mean_std(data[m]["t"])
        led, off = statistics.mean(data[m]["led"]), statistics.mean(data[m]["off"])
        res[m] = {"label": label, "ms_par_round": mean, "ecart_type_ms": sd,
                  "surcout_ms": mean - base, "surcout_pct": (mean - base) / base * 100,
                  "octets_registre_par_round": led, "octets_hors_chaine_par_round": off}
        print(f"{label:<48}{mean:>9.1f}{sd:>7.1f}{mean - base:>+9.1f}{led:>9.0f} o{off:>11.0f} o")
    pd.DataFrame(res).T.to_csv("mesures_configurations.csv", index=False)
    RESULTS["configurations"] = res
    print("   (surcoût en ms par round ; octets écrits par round dans le registre / hors chaîne)")


# ------------------------------------------------------------------ 2. coût unitaire
def exp_micro():
    w = np.random.default_rng(0).normal(size=21)
    rm("bench_micro.jsonl")
    L = Ledger("bench_micro.jsonl")
    payload = {"round": 1, "clinic_id": 1, "n_samples": 886, "weights_hash": "a" * 64,
               "base_model_hash": "b" * 64, "params": TRAIN_PARAMS}
    fp = timed_ms(lambda: fingerprint_weights(w), 2000) * 1000
    ap = timed_ms(lambda: L.append("contribution", payload), 100)
    so = timed_ms(lambda: save_offchain(w), 100)
    print("\n2. COÛT UNITAIRE")
    print(f"   Empreinte SHA-256 de 21 poids       : {fp:8.1f} µs")
    print(f"   Écriture d'un bloc (avec fsync)     : {ap:8.2f} ms   <- l'essentiel du surcoût")
    print(f"   Sauvegarde des poids hors chaîne    : {so:8.2f} ms")
    RESULTS["micro"] = {"empreinte_us": fp, "ecriture_bloc_ms": ap, "sauvegarde_hors_chaine_ms": so}


# ------------------------------------------------------------------ 3. taille du modèle
def exp_size():
    rng = np.random.default_rng(1)
    rows = []
    print("\n3. VOLUME ÉCRIT SUR LA CHAÎNE QUAND LA TAILLE DU MODÈLE VARIE")
    print(f"{'Poids':>12}{'hors chaîne':>16}{'bloc sur chaîne':>18}{'empreinte':>14}")
    for n in (21, 2_100, 210_000, 2_100_000):
        w = rng.normal(size=n)
        ms_fp = timed_ms(lambda: fingerprint_weights(w), 5)
        rm("bench_size.jsonl")
        L = Ledger("bench_size.jsonl")
        b = L.append("contribution", {"round": 1, "clinic_id": 1, "n_samples": 886,
                                      "weights_hash": fingerprint_weights(w),
                                      "base_model_hash": "b" * 64, "params": TRAIN_PARAMS})
        bloc = len(json.dumps(b, ensure_ascii=False).encode("utf-8")) + 1
        rows.append({"poids": n, "octets_hors_chaine": 8 * n, "octets_bloc": bloc, "empreinte_ms": ms_fp})
        print(f"{n:>12,}{8 * n:>14,} o{bloc:>16} o{ms_fp:>11.2f} ms".replace(",", " "))
    RESULTS["taille_modele"] = rows
    print("   (hors chaîne = poids en float64 binaire : 8 octets par poids)")


# ------------------------------------------------------------------ 4. croissance, porte, provenance
def exp_growth():
    rm("bench_100.jsonl", "bench_10.jsonl")
    L100 = Ledger("bench_100.jsonl")
    t = time.perf_counter()
    run_anchored(L100, n_rounds=100, verbose=False)
    duree = time.perf_counter() - t
    pts = []
    print("\n4. CROISSANCE DU REGISTRE ET COÛT DE LA VÉRIFICATION")
    print(f"{'Rounds':>8}{'Blocs':>8}{'Taille':>12}{'verify_chain':>16}")
    for r in (10, 25, 50, 100):
        nb = L100.find("global_model", round=r)[0]["index"] + 1
        size = sum(len(json.dumps(b, ensure_ascii=False).encode("utf-8")) + 1 for b in L100.blocks[:nb])
        sub = Ledger("bench_100.jsonl")
        sub.blocks = sub.blocks[:nb]
        v = timed_ms(sub.verify_chain, 15)
        pts.append({"rounds": r, "blocs": nb, "octets": size, "verify_chain_ms": v})
        print(f"{r:>8}{nb:>8}{size / 1024:>9.1f} Ko{v:>13.2f} ms")
    print(f"   Fédération ancrée de 100 rounds : {duree:.1f} s")

    L10 = Ledger("bench_10.jsonl")
    run_anchored(L10, n_rounds=10, verbose=False)
    gp, prov = {}, {}
    print("\n   Porte de certification et provenance")
    print(f"{'Registre':>12}{'porte':>12}{'diagnostic':>14}{'provenance':>14}")
    for name, LL, rr in (("10 rounds", L10, 10), ("100 rounds", L100, 100)):
        model = load_model(LL, rr)
        t_gate = timed_ms(lambda: certify_model(LL, model), 15)
        cases = []

        def diag():
            cases.append(diagnose(LL, model, RAW, f"bench-{name}-{len(cases)}")["case"])

        t_diag = timed_ms(diag, 5)
        t_tr = timed_ms(lambda: trace(LL, cases[-1]), 10)
        gp[name] = {"porte_ms": t_gate, "diagnostic_ms": t_diag, "provenance_ms": t_tr}
        print(f"{name:>12}{t_gate:>9.2f} ms{t_diag:>11.2f} ms{t_tr:>11.2f} ms")
    RESULTS["croissance"] = {"duree_100_rounds_s": duree, "points": pts, "porte_provenance": gp}
    return L10


# ------------------------------------------------------------------ 5. détection
def exp_detection(L10):
    rng = np.random.default_rng(42)
    print("\n5. TAUX DE DÉTECTION")

    # a. explications falsifiées
    model = load_model(L10, 10)
    out = diagnose(L10, model, RAW, "bench-falsif")
    h = L10.find("diagnosis", case=out["case"])[-1]["payload"]["explanation_hash"]
    path = OFF / "explanations" / f"{h}.json"
    original = path.read_text(encoding="utf-8")
    text = json.loads(original)["text"]
    kinds = {"retrait des facteurs": 0, "un caractère modifié": 0, "un espace ajouté": 0}
    n_per = 70
    for kind in kinds:
        for _ in range(n_per):
            if kind == "retrait des facteurs":
                new = text.split(" Facteurs")[0]
            elif kind == "un caractère modifié":
                i = int(rng.integers(0, len(text)))
                new = text[:i] + ("X" if text[i] != "X" else "Y") + text[i + 1:]
            else:
                new = text + " "
            path.write_text(json.dumps({"case": out["case"], "text": new}, ensure_ascii=False),
                            encoding="utf-8")
            if not verify_explanation(L10, out["case"])[0]:
                kinds[kind] += 1
    path.write_text(original, encoding="utf-8")
    fausse_alarme = sum(1 for _ in range(200) if not verify_explanation(L10, out["case"])[0])
    tot = sum(kinds.values())
    print(f"   Explications falsifiées détectées : {tot}/{n_per * 3} "
          f"({', '.join(f'{k} {v}/{n_per}' for k, v in kinds.items())})")
    print(f"   Fausses alarmes sur explications intactes : {fausse_alarme}/200")

    # b. substitution de modèle et faux refus de la porte
    models = {r: load_model(L10, r) for r in range(1, 11)}
    refused, n_sub = 0, 200
    for _ in range(n_sub):
        m = SoftmaxRegression()
        w = models[int(rng.integers(1, 11))].get_weights().copy()
        w[int(rng.integers(0, len(w)))] += rng.choice([-1, 1]) * 10 ** rng.uniform(-12, -1)
        m.set_weights(w)
        if not certify_model(L10, m)[0]:
            refused += 1
    faux_refus = sum(1 for r in range(1, 11) for _ in range(20)
                     if not certify_model(L10, models[r])[0])
    X1, y1 = load_dataset("clinique_1")
    fake_ref = sum(1 for s in range(20)
                   if not certify_model(L10, SoftmaxRegression().fit(X1, y1, epochs=5 + s, seed=s))[0])
    print(f"   Modèles substitués refusés (écart de 1e-12 à 1e-1 sur un poids) : {refused}/{n_sub}")
    print(f"   Modèles entraînés hors fédération refusés : {fake_ref}/20")
    print(f"   Faux refus (modèles légitimes refusés) : {faux_refus}/200")

    # c. falsification du registre
    lines = open("bench_10.jsonl", encoding="utf-8").read().splitlines()
    n = len(lines)
    mod, sup = 0, 0
    for _ in range(50):
        i = int(rng.integers(1, n - 1))
        d = json.loads(lines[i])
        k = next(iter(d["payload"]))
        d["payload"][k] = "falsifie"
        alt = lines[:i] + [json.dumps(d, ensure_ascii=False)] + lines[i + 1:]
        open("bench_tamper.jsonl", "w", encoding="utf-8").write("\n".join(alt) + "\n")
        if not Ledger("bench_tamper.jsonl").verify_chain()[0]:
            mod += 1
        alt = lines[:i] + lines[i + 1:]
        open("bench_tamper.jsonl", "w", encoding="utf-8").write("\n".join(alt) + "\n")
        if not Ledger("bench_tamper.jsonl").verify_chain()[0]:
            sup += 1
    open("bench_tamper.jsonl", "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
    tronque = not Ledger("bench_tamper.jsonl").verify_chain()[0]
    print(f"   Blocs modifiés dans le registre détectés : {mod}/50")
    print(f"   Blocs supprimés au milieu détectés : {sup}/50")
    print(f"   Suppression du DERNIER bloc détectée : {'oui' if tronque else 'NON (limite connue)'}")
    RESULTS["detection"] = {
        "explications_falsifiees": f"{tot}/{n_per * 3}", "fausses_alarmes": f"{fausse_alarme}/200",
        "substitutions_refusees": f"{refused}/{n_sub}", "faux_modeles_refuses": f"{fake_ref}/20",
        "faux_refus": f"{faux_refus}/200", "blocs_modifies_detectes": f"{mod}/50",
        "blocs_supprimes_detectes": f"{sup}/50", "troncature_detectee": tronque}


if __name__ == "__main__":
    t0 = time.time()
    print("FedTrace - mesures (registre local simplifié, une seule machine)")
    exp_configs()
    exp_micro()
    exp_size()
    L10 = exp_growth()
    exp_detection(L10)
    with open("mesures.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRésultats enregistrés dans mesures.json et mesures_configurations.csv "
          f"({time.time() - t0:.0f} s)")