import hashlib
import json
import time

import pandas as pd

from aggregation import weighted_average, median_with_exclusion, sign_flip_attack
from anchored_federation import save_offchain, OFFCHAIN
from chain_client import ChainClient
from diagnosis import OFF, prepare_case, build_explanation, pseudonymize
from federation import local_train, CLINIQUES
from fingerprint import fingerprint_weights, fingerprint_text
from model import SoftmaxRegression, load_dataset

RAW = pd.read_csv("cliniques/test_numeric.csv").iloc[0].to_dict()
METRICS = {"accuracy": 0.9, "f1_macro": 0.8}


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


def check(label, cond):
    print(f"   [{'OK' if cond else 'ÉCHEC'}] {label}")
    return bool(cond)


def expect_revert(label, fn):
    """Le contrat doit refuser la transaction."""
    try:
        fn()
    except Exception as e:
        msg = str(e).replace("\n", " ")
        ok = "revert" in msg.lower() or type(e).__name__ == "ContractLogicError"
        i = msg.find("FedTrace:")
        reason = msg[i:i + 80] if i >= 0 else msg[:80]
        print(f"   [{'OK' if ok else 'ÉCHEC'}] {label} : refusé -> {reason}")
        return ok
    print(f"   [ÉCHEC] {label} : le contrat aurait dû refuser")
    return False


# ====================================================================== PARTIE A
def part_a():
    print("\nPARTIE A : règles du contrat (droits d'écriture et validation des écritures)")
    c = ChainClient()
    c.deploy()
    r = []
    init, m1, expl = H("init"), H("m1"), H("explication")

    r.append(expect_revert("modèle initial ancré par un inconnu",
                           lambda: c.anchor_initial_model(init, sender=c.outsider)))
    c.anchor_initial_model(init)
    r.append(check("modèle initial ancré par le serveur", c.provenance(init) == [(0, init)]))
    r.append(expect_revert("second modèle initial", lambda: c.anchor_initial_model(H("autre"))))

    r.append(expect_revert("contribution ancrée par un inconnu",
                           lambda: c.anchor_contribution(1, 1, H("w1"), init, 886, sender=c.outsider)))
    r.append(expect_revert("contribution ancrée par le serveur",
                           lambda: c.anchor_contribution(1, 1, H("w1"), init, 886, sender=c.server)))
    r.append(expect_revert("contribution avec un modèle de départ inconnu",
                           lambda: c.anchor_contribution(1, 1, H("w1"), H("inconnu"), 886)))
    r.append(expect_revert("contribution du mauvais round",
                           lambda: c.anchor_contribution(1, 2, H("w1"), init, 886)))
    c.anchor_contribution(1, 1, H("w1"), init, 886)
    r.append(check("contribution de la clinique 1 ancrée", len(c.round_contributions(1)) == 1))
    r.append(expect_revert("même clinique, même round, deux fois",
                           lambda: c.anchor_contribution(1, 1, H("w1bis"), init, 886)))

    r.append(expect_revert("modèle ancré par une clinique",
                           lambda: c.anchor_model(1, m1, init, "weighted_average", [1], [], METRICS,
                                                  sender=c.clinics[1])))
    r.append(expect_revert("modèle ancré AVANT les contributions des cliniques 2 et 3",
                           lambda: c.anchor_model(1, m1, init, "weighted_average", [1, 2, 3], [], METRICS)))
    c.anchor_contribution(2, 1, H("w2"), init, 886)
    c.anchor_contribution(3, 1, H("w3"), init, 886)
    r.append(expect_revert("modèle avec un parent inconnu",
                           lambda: c.anchor_model(1, m1, H("autre"), "weighted_average", [1, 2, 3], [], METRICS)))
    r.append(expect_revert("clinique à la fois retenue et écartée",
                           lambda: c.anchor_model(1, H("m1x"), init, "median", [1, 2], [2], METRICS)))
    r.append(expect_revert("round qui ne suit pas le parent",
                           lambda: c.anchor_model(2, H("m2x"), init, "weighted_average", [1, 2, 3], [], METRICS)))
    c.anchor_model(1, m1, init, "weighted_average", [1, 2, 3], [], METRICS)
    v = c.verify_model(m1)
    r.append(check("modèle du round 1 ancré et certifié",
                   v["certified"] and v["round"] == 1 and v["retained"] == [1, 2, 3] and v["accuracy"] == 0.9))
    r.append(expect_revert("même modèle ancré deux fois",
                           lambda: c.anchor_model(1, m1, init, "weighted_average", [1, 2, 3], [], METRICS)))

    r.append(expect_revert("diagnostic avec un modèle inconnu",
                           lambda: c.anchor_decision(1, "CAS-TEST", H("zzz"), expl, 1, 0.95)))
    r.append(expect_revert("diagnostic avec le modèle initial (non entraîné)",
                           lambda: c.anchor_decision(1, "CAS-TEST", init, expl, 1, 0.95)))
    r.append(expect_revert("diagnostic ancré par un inconnu",
                           lambda: c.anchor_decision(1, "CAS-TEST", m1, expl, 1, 0.95, sender=c.outsider)))
    c.anchor_decision(1, "CAS-TEST", m1, expl, 1, 0.95)
    r.append(check("diagnostic ancré avec un modèle certifié",
                   c.verify_decision("CAS-TEST", expl) and not c.verify_decision("CAS-TEST", H("autre"))
                   and len(c.decisions("CAS-TEST")) == 1))
    r.append(check("provenance du round 1 jusqu'au modèle initial", c.provenance(m1) == [(1, m1), (0, init)]))
    r.append(check("modèle initial et modèle inconnu non certifiés",
                   not c.verify_model(init)["certified"] and not c.verify_model(H("zzz"))["certified"]))
    return all(r)


# ====================================================================== PARTIE B
def federate_on_chain(client, aggregate, strategy, attacks=None, n_rounds=10):
    """Même boucle que run_anchored, mais les ancrages sont des transactions signées
    par la bonne identité (cliniques pour les contributions, serveur pour les modèles)."""
    client.deploy()
    start = len(client.gas_used)
    Xv, yv = load_dataset("val_numeric")
    w = SoftmaxRegression().get_weights()
    parent = save_offchain(w)
    client.anchor_initial_model(parent)
    hashes, metrics = [parent], {}
    for rd in range(1, n_rounds + 1):
        base = w
        updates = [local_train(cid, base, rd) for cid in CLINIQUES]
        for u in updates:
            if attacks and u["clinic_id"] in attacks:
                u["weights"] = attacks[u["clinic_id"]](u["weights"], base)
        for u in updates:                                           # AVANT l'agrégation
            client.anchor_contribution(u["clinic_id"], rd, save_offchain(u["weights"]),
                                       parent, u["n_samples"])
        w, retained, discarded = aggregate(updates)
        m = SoftmaxRegression()
        m.set_weights(w)
        metrics = m.evaluate(Xv, yv)
        mh = save_offchain(w)
        client.anchor_model(rd, mh, parent, strategy, retained, discarded, metrics)  # APRÈS
        parent = mh
        hashes.append(mh)
    return w, hashes, metrics, sum(client.gas_used[start:])


def diagnose_on_chain(client, model, raw, ref, clinic_id=1):
    x, vals = prepare_case(raw)
    p = model.predict_proba(x)[0]
    pred = int(p.argmax())
    proba = float(p[pred])
    text = build_explanation(model, x, vals, pred, proba)
    case = pseudonymize(ref)
    mh = fingerprint_weights(model.get_weights())
    eh = fingerprint_text(text)
    if not client.is_certified(mh):                                  # porte de certification
        raise PermissionError("modèle non certifié : diagnostic refusé par la porte")
    folder = OFF / "explanations"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{eh}.json").write_text(json.dumps({"case": case, "text": text}, ensure_ascii=False),
                                       encoding="utf-8")
    client.anchor_decision(clinic_id, case, mh, eh, pred, proba)
    return {"case": case, "explanation_hash": eh, "model_hash": mh, "text": text}


def part_b():
    print("\nPARTIE B : fédération réelle ancrée sur la chaîne locale")
    c = ChainClient()
    r = []

    t0 = time.perf_counter()
    w1, hashes1, metrics1, gas1 = federate_on_chain(c, weighted_average, "weighted_average")
    dt = time.perf_counter() - t0
    final = hashes1[-1]
    v = c.verify_model(final)
    r.append(check("modèle du round 10 ancré et certifié sur la chaîne", v["certified"] and v["round"] == 10))

    prov = c.provenance(final)
    r.append(check("provenance on-chain : rounds 10 -> 0, mêmes empreintes que la fédération",
                   [x for x, _ in prov] == list(range(10, -1, -1)) and [h for _, h in prov] == hashes1[::-1]))

    contribs = c.round_contributions(10)
    r.append(check("3 contributions du round 10 ancrées, poids conservés hors chaîne",
                   len(contribs) == 3 and all((OFFCHAIN / f"{x['weights_hash']}.json").exists()
                                               and x["base_hash"] == hashes1[9] for x in contribs)))
    r.append(check(f"métriques ancrées (accuracy {v['accuracy']:.4f})",
                   abs(v["accuracy"] - metrics1["accuracy"]) < 1e-4))

    model = SoftmaxRegression()
    model.set_weights(w1)
    X1, y1 = load_dataset("clinique_1")
    fake = SoftmaxRegression().fit(X1, y1, epochs=30)
    sub = SoftmaxRegression()
    wsub = w1.copy()
    wsub[0] += 1e-9
    sub.set_weights(wsub)
    r.append(check("porte : modèle fédéré certifié",
                   c.is_certified(fingerprint_weights(w1))))
    r.append(check("porte : faux modèle et substitution discrète (1e-9) non certifiés",
                   not c.is_certified(fingerprint_weights(fake.get_weights()))
                   and not c.is_certified(fingerprint_weights(sub.get_weights()))))

    out = diagnose_on_chain(c, model, RAW, "chain-1")
    altered = out["text"].split(" Facteurs")[0]
    r.append(check("diagnostic ancré, explication intacte vérifiée sur la chaîne",
                   c.verify_decision(out["case"], out["explanation_hash"])))
    r.append(check("explication falsifiée : empreinte absente de la chaîne",
                   not c.verify_decision(out["case"], fingerprint_text(altered))))
    try:
        diagnose_on_chain(c, fake, RAW, "chain-2")
        refused = False
    except PermissionError:
        refused = True
    r.append(check("faux modèle : diagnostic refusé par la porte", refused))
    r.append(expect_revert("faux modèle : ancrage direct refusé par le contrat lui-même",
                           lambda: c.anchor_decision(1, "CAS-FAUX", fingerprint_weights(fake.get_weights()),
                                                     H("x"), 1, 0.9)))

    print("\n   Clinique 3 malveillante, agrégation robuste (nouveau déploiement)")
    w2, hashes2, metrics2, gas2 = federate_on_chain(
        c, median_with_exclusion, "median_with_exclusion", attacks={3: sign_flip_attack(50.0)})
    ok_excl = 0
    for h in hashes2[1:]:
        vm = c.verify_model(h)
        if vm["discarded"] == [3] and vm["retained"] == [1, 2]:
            ok_excl += 1
    r.append(check(f"rejet de la clinique 3 ancré sur la chaîne à chaque round ({ok_excl}/10)", ok_excl == 10))
    c3 = [x for x in c.round_contributions(1) if x["clinic_id"] == 3]
    r.append(check("sa contribution empoisonnée est bien ancrée (preuve conservée)",
                   len(c3) == 1 and (OFFCHAIN / f"{c3[0]['weights_hash']}.json").exists()))
    r.append(check(f"qualité préservée (accuracy {metrics2['accuracy']:.3f} vs {metrics1['accuracy']:.3f})",
                   metrics2["accuracy"] >= metrics1["accuracy"] - 0.05))

    print(f"\n   Coût sur la chaîne locale (fédération honnête de 10 rounds) :")
    print(f"      durée : {dt / 10 * 1000:.0f} ms par round (entraînement inclus)")
    print(f"      gas   : {gas1 / 10:,.0f} par round (hors déploiement)".replace(",", " "))
    return all(r)


if __name__ == "__main__":
    ok_a = part_a()
    ok_b = part_b()
    print("\n" + "=" * 52)
    print(f"Partie A (règles du contrat)      : {'RÉUSSIE' if ok_a else 'ÉCHEC'}")
    print(f"Partie B (fédération sur chaîne)  : {'RÉUSSIE' if ok_b else 'ÉCHEC'}")