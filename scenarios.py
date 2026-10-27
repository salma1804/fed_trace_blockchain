import json
import os
import time
import pandas as pd
from aggregation import weighted_average, median_with_exclusion, sign_flip_attack
from anchor_agent import AnchorAgent, RegistryUnavailable
from anchored_federation import run_anchored, OFFCHAIN
from diagnosis import OFF, verify_explanation, trace
from gate import certified_diagnose, CertificationError
from ledger import Ledger
from model import SoftmaxRegression, load_dataset

RAW = pd.read_csv("cliniques/test_numeric.csv").iloc[0].to_dict()


def rm(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def check(label, cond):
    print(f"   [{'OK' if cond else 'ÉCHEC'}] {label}")
    return bool(cond)


def load_model(L, r):
    h = L.find("global_model", round=r)[0]["payload"]["model_hash"]
    m = SoftmaxRegression()
    m.set_weights(json.loads((OFFCHAIN / f"{h}.json").read_text()))
    return m


def final_acc(L):
    return L.find("global_model", round=10)[0]["payload"]["metrics"]["accuracy"]


def discards(L):
    return [(b["payload"]["round"], b["payload"]["discarded"])
            for b in L.find("global_model") if b["payload"]["round"] > 0]


def payload_seq(L):
    return [(b["record_type"], {k: v for k, v in b["payload"].items() if k != "anchor_id"})
            for b in L.blocks]


class FlakyLedger:
    """Registre dont on peut simuler l'arrêt (les écritures échouent)."""
    def __init__(self, ledger):
        self._l = ledger
        self.down = False

    def append(self, record_type, payload):
        if self.down:
            raise RegistryUnavailable("registre arrêté")
        return self._l.append(record_type, payload)

    def __len__(self):
        return len(self._l)

    def __getattr__(self, name):
        return getattr(self._l, name)


# ---------------------------------------------------------------- Scénario 1
def scenario_1():
    print("\nSCÉNARIO 1 : fonctionnement normal")
    rm("scenario_1.jsonl")
    L = Ledger("scenario_1.jsonl")
    run_anchored(L, verbose=False)
    r = [check("fédération de 3 cliniques sur 10 rounds, 42 blocs, chaîne valide",
               len(L) == 42 and L.verify_chain()[0])]
    out = certified_diagnose(L, load_model(L, 10), RAW, "scn-1")
    d = L.find("diagnosis")[0]["payload"]
    r.append(check(f"modèle certifié, diagnostic produit : {out['prediction']} ({out['probability']})",
                   len(L.find("diagnosis")) == 1))
    r.append(check("empreintes du modèle et de l'explication ancrées",
                   bool(d["model_hash"]) and bool(d["explanation_hash"])))
    r.append(check("provenance reconstituée jusqu'au round 0",
                   any("round 0" in l for l in trace(L, out["case"]))))
    return all(r), L, out


# ---------------------------------------------------------------- Scénario 2
def scenario_2(L, out):
    print("\nSCÉNARIO 2 : explication falsifiée")
    h = L.find("diagnosis", case=out["case"])[-1]["payload"]["explanation_hash"]
    p = OFF / "explanations" / f"{h}.json"
    original = p.read_text(encoding="utf-8")
    r = [check("avant modification : explication intacte", verify_explanation(L, out["case"])[0])]
    data = json.loads(original)
    data["text"] = data["text"].split(" Facteurs")[0]       # on retire un facteur gênant
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    ok, msg = verify_explanation(L, out["case"])
    r.append(check(f"après modification : falsification détectée ({msg})", not ok))
    p.write_text(original, encoding="utf-8")                # restauration
    return all(r)


# ---------------------------------------------------------------- Scénario 3
def scenario_3(L):
    print("\nSCÉNARIO 3 : modèle non certifié")
    n = len(L.find("diagnosis"))
    X1, y1 = load_dataset("clinique_1")
    Xv, yv = load_dataset("val_numeric")
    fake = SoftmaxRegression().fit(X1, y1, epochs=30)
    try:
        certified_diagnose(L, fake, RAW, "scn-3")
        refused, msg = False, ""
    except CertificationError as e:
        refused, msg = True, str(e)
    r = [check(f"modèle entraîné hors fédération (accuracy {fake.evaluate(Xv, yv)['accuracy']:.2f}) : "
               "diagnostic refusé", refused),
         check(f"motif : {msg[:70]}...", refused),
         check("aucun diagnostic ancré pour ce modèle", len(L.find("diagnosis")) == n)]
    return all(r)


# ---------------------------------------------------------------- Scénario 4
def scenario_4(acc_nominal):
    print("\nSCÉNARIO 4 : clinique malveillante (clinique 3)")
    attack = {3: sign_flip_attack(50.0)}
    rm("scenario_4_robuste.jsonl", "scenario_4_moyenne.jsonl")
    R = Ledger("scenario_4_robuste.jsonl")
    run_anchored(R, aggregate=median_with_exclusion, strategy="median_with_exclusion",
                 attacks=attack, verbose=False)
    W = Ledger("scenario_4_moyenne.jsonl")
    run_anchored(W, aggregate=weighted_average, strategy="weighted_average",
                 attacks=attack, verbose=False)
    accR, accW = final_acc(R), final_acc(W)
    det = sum(1 for _, d in discards(R) if d == [3])
    ok_avant = True
    for rd in range(1, 11):
        c3 = [c for c in R.find("contribution", round=rd) if c["payload"]["clinic_id"] == 3][0]
        m = R.find("global_model", round=rd)[0]
        if not (c3["index"] < m["index"]
                and (OFFCHAIN / f"{c3['payload']['weights_hash']}.json").exists()):
            ok_avant = False
    r = [check(f"agrégation robuste : clinique 3 écartée à chaque round ({det}/10), rejet ancré", det == 10),
         check("sa contribution était ancrée AVANT l'agrégation (non-répudiation)", ok_avant),
         check(f"agrégation robuste : qualité préservée (acc {accR:.3f} vs {acc_nominal:.3f} sans attaque)",
               accR >= acc_nominal - 0.05),
         check(f"agrégation NON robuste : rien d'écarté, contribution intégrée au modèle (acc {accW:.3f})",
               all(d == [] for _, d in discards(W)) and accW < accR - 0.10)]
    return all(r)


# ---------------------------------------------------------------- Scénario 5
def scenario_5(L_ref):
    print("\nSCÉNARIO 5 : registre arrêté pendant des rounds, puis relancé")
    rm("scenario_5.jsonl", "scenario_5_queue.jsonl", "scenario_5b.jsonl", "scenario_5b_queue.jsonl")
    L5 = Ledger("scenario_5.jsonl")
    flaky = FlakyLedger(L5)
    agent = AnchorAgent(flaky, "scenario_5_queue.jsonl")
    state = {"c": 0, "len_during": None, "peak": 0}

    def hooked(updates):                       # appelé une fois par round
        state["c"] += 1
        if state["c"] == 3:
            flaky.down = True                  # le registre s'arrête
        if state["c"] == 5:
            state["peak"] = agent.pending()
            state["len_during"] = len(L5)
            flaky.down = False                 # le registre redémarre
        return weighted_average(updates)

    run_anchored(agent, aggregate=hooked, verbose=False)
    # Le registre de référence (scénario 1) contient en plus le diagnostic : on compare les 42 premiers blocs
    r = [check(f"pendant l'arrêt : {state['peak']} ancrages mis en file "
               f"(registre figé à {state['len_during']} blocs)",
               state["peak"] > 0 and state["len_during"] < 42),
         check("après redémarrage : file vide", agent.pending() == 0),
         check(f"aucun ancrage perdu : {len(L5)} blocs, chaîne valide",
               len(L5) == 42 and L5.verify_chain()[0]),
         check("contenu et ordre identiques à une exécution sans panne",
               payload_seq(L5) == payload_seq(L_ref)[:len(L5)])]

    # Arrêt avec plantage du programme : la file survit sur disque
    LB = Ledger("scenario_5b.jsonl")
    fb = FlakyLedger(LB)
    fb.down = True
    a1 = AnchorAgent(fb, "scenario_5b_queue.jsonl")
    for i in range(3):
        a1.append("contribution", {"round": 1, "clinic_id": i + 1, "hash": f"h{i}"})
    r.append(check("registre arrêté : 3 ancrages en file, 0 écrit", a1.pending() == 3 and len(LB) == 1))
    del a1                                      # le programme "plante"
    fb.down = False
    a2 = AnchorAgent(fb, "scenario_5b_queue.jsonl")
    r.append(check("nouveau programme : la file est retrouvée", a2.pending() == 3))
    a2.flush()
    r.append(check("rejeu : 3 blocs écrits, file vide, chaîne valide",
                   a2.pending() == 0 and len(LB) == 4 and LB.verify_chain()[0]))
    # Plantage entre l'écriture et le nettoyage de la file : pas de doublon
    item = {"anchor_id": "deja-ecrit", "record_type": "contribution",
            "payload": {"round": 9, "clinic_id": 1, "hash": "dup"}}
    LB.append("contribution", dict(item["payload"], anchor_id="deja-ecrit"))
    a2._save_queue([item])
    n = len(LB)
    a2.flush()
    r.append(check("rejeu d'un ancrage déjà écrit : pas de doublon", len(LB) == n and a2.pending() == 0))
    return all(r)


if __name__ == "__main__":
    t0 = time.time()
    ok1, L1, out1 = scenario_1()
    ok2 = scenario_2(L1, out1)
    ok3 = scenario_3(L1)
    ok4 = scenario_4(final_acc(L1))
    ok5 = scenario_5(L1)

    print("\n" + "=" * 52)
    for i, ok in enumerate([ok1, ok2, ok3, ok4, ok5], start=1):
        print(f"Scénario {i} : {'RÉUSSI' if ok else 'ÉCHEC'}")
    print(f"Durée totale : {time.time() - t0:.0f} s")