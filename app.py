import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from anchored_federation import OFFCHAIN
from diagnosis import OFF, trace
from fingerprint import fingerprint_text
from gate import certified_diagnose, CertificationError
from ledger import Ledger
from model import SoftmaxRegression, load_dataset

UI_LEDGER = "ledger_ui.jsonl"

st.set_page_config(page_title="FedTrace - Audit", layout="wide")
st.title("FedTrace - Interface d'audit")
st.caption("Prototype pédagogique : ne pas utiliser avec de vraies données de patients.")


# ------------------------------------------------------------ utilitaires
def short(h, n=16):
    return (h[:n] + "...") if h else ""


def load_round_model(L, r):
    h = L.find("global_model", round=r)[0]["payload"]["model_hash"]
    m = SoftmaxRegression()
    m.set_weights(json.loads((OFFCHAIN / f"{h}.json").read_text()))
    return m


@st.cache_resource
def fake_model():
    X1, y1 = load_dataset("clinique_1")
    return SoftmaxRegression().fit(X1, y1, epochs=30)


def explanation_info(p):
    path = OFF / "explanations" / f"{p['explanation_hash']}.json"
    if not path.exists():
        return path, None, None
    text = json.loads(path.read_text(encoding="utf-8"))["text"]
    return path, text, fingerprint_text(text)


# ------------------------------------------------------------ registre
if not Path(UI_LEDGER).exists():
    if Path("ledger.jsonl").exists():
        shutil.copy("ledger.jsonl", UI_LEDGER)
    else:
        st.error("Registre introuvable. Lance d'abord : python anchored_federation.py")
        st.stop()

others = sorted(p.name for p in Path(".").glob("*.jsonl")
                if p.name != UI_LEDGER and "queue" not in p.name)
choice = st.sidebar.selectbox("Registre affiché", [UI_LEDGER] + others)
if st.sidebar.button("Réinitialiser le registre de démo"):
    shutil.copy("ledger.jsonl", UI_LEDGER)
    st.session_state.pop("last", None)
    st.rerun()
st.sidebar.caption("Les diagnostics ne sont possibles que sur " + UI_LEDGER
                   + ". Les autres registres sont en lecture seule.")

L = Ledger(choice)
ok, msg = L.verify_chain()
models = L.find("global_model")
nb_rounds = max([b["payload"]["round"] for b in models], default=0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Blocs", len(L))
c2.metric("Chaîne", "valide" if ok else "ROMPUE")
c3.metric("Rounds", nb_rounds)
c4.metric("Diagnostics ancrés", len(L.find("diagnosis")))
if not ok:
    st.error(msg)

tabs = st.tabs(["Rounds", "Diagnostic", "Provenance et audit", "Blocs du registre"])

# ------------------------------------------------------------ onglet 1 : rounds
with tabs[0]:
    if not models:
        st.info("Aucun modèle ancré dans ce registre.")
    else:
        rows = []
        for b in models:
            p = b["payload"]
            m = p.get("metrics", {})
            rows.append({
                "Round": p["round"], "Bloc (transaction)": b["index"],
                "Stratégie": p["strategy"], "Empreinte du modèle": short(p["model_hash"]),
                "Accuracy (val)": m.get("accuracy"), "F1 macro": m.get("f1_macro"),
                "Retenues": ", ".join(map(str, p["retained"])),
                "Écartées": ", ".join(map(str, p["discarded"]))})
        st.dataframe(pd.DataFrame(rows))

        excl = [(b["payload"]["round"], b["payload"]["discarded"])
                for b in models if b["payload"]["discarded"]]
        if excl:
            st.warning("Rejets ancrés dans le registre : "
                       + " ; ".join(f"round {r} : clinique(s) {d}" for r, d in excl))

        rounds = [b["payload"]["round"] for b in models if b["payload"]["round"] > 0]
        if rounds:
            r = st.selectbox("Détail d'un round", rounds, index=len(rounds) - 1)
            mb = L.find("global_model", round=r)[0]
            contribs = [c for c in L.find("contribution", round=r)
                        if c["payload"]["base_model_hash"] == mb["payload"]["parent_hash"]]
            st.dataframe(pd.DataFrame([{
                "Clinique": c["payload"]["clinic_id"],
                "Patients déclarés": c["payload"]["n_samples"],
                "Empreinte de la mise à jour": short(c["payload"]["weights_hash"]),
                "Bloc (transaction)": c["index"],
                "Écartée ?": "OUI" if c["payload"]["clinic_id"] in mb["payload"]["discarded"] else "non",
            } for c in contribs]))
            if contribs:
                st.caption("Contributions ancrées aux blocs "
                           + ", ".join(str(c["index"]) for c in contribs)
                           + f" AVANT le modèle global (bloc {mb['index']}).")

# ------------------------------------------------------------ onglet 2 : diagnostic
with tabs[1]:
    rounds_ok = sorted(b["payload"]["round"] for b in models if b["payload"]["round"] > 0)
    if choice != UI_LEDGER:
        st.info(f"Sélectionne {UI_LEDGER} dans la barre latérale pour lancer des diagnostics.")
    elif not rounds_ok:
        st.warning("Aucun modèle fédéré dans ce registre.")
    else:
        cases = pd.read_csv("cliniques/test_numeric.csv")
        idx = int(st.number_input("Numéro du cas (jeu de test)", min_value=0,
                                  max_value=len(cases) - 1, value=0, step=1))
        st.dataframe(cases.iloc[[idx]])
        options = ([f"Modèle fédéré - round {r}" for r in rounds_ok]
                   + ["Faux modèle (entraîné hors fédération)",
                      "Modèle fédéré modifié discrètement (1 poids + 0,000001)"])
        sel = st.selectbox("Modèle déployé", options, index=len(rounds_ok) - 1)

        if st.button("Lancer le diagnostic"):
            raw = cases.iloc[idx].to_dict()
            if sel.startswith("Modèle fédéré - round"):
                model = load_round_model(L, int(sel.split()[-1]))
            elif sel.startswith("Faux"):
                model = fake_model()
            else:
                model = load_round_model(L, rounds_ok[-1])
                w = model.get_weights()
                w[0] += 1e-6
                model.set_weights(w)
            try:
                out = certified_diagnose(L, model, raw, f"ui-{idx}")
                st.session_state["last"] = ("ok", out, sel)
            except CertificationError as e:
                st.session_state["last"] = ("refused", str(e), sel)
            st.rerun()

        last = st.session_state.get("last")
        if last:
            kind, data, used = last
            if kind == "ok":
                st.success(f"Diagnostic AUTORISÉ ({used}) : {data['prediction']} "
                           f"- probabilité {data['probability']}")
                st.write(data["explanation"])
                st.caption(f"Pseudonyme du cas : {data['case']} - ancré au bloc {data['block']} "
                           f"- modèle certifié (round {data['certified_round']})")
            else:
                st.error(f"Diagnostic REFUSÉ ({used}) : {data}")

# ------------------------------------------------------------ onglet 3 : provenance
with tabs[2]:
    latest = {}
    for b in L.find("diagnosis"):
        latest[b["payload"]["case"]] = b
    if not latest:
        st.info("Aucun diagnostic dans ce registre. Lance-en un dans l'onglet Diagnostic.")
    else:
        case = st.selectbox("Diagnostic (pseudonyme du cas)", list(latest.keys()),
                            index=len(latest) - 1)
        b = latest[case]
        p = b["payload"]
        st.write(f"Bloc {b['index']} - clinique émettrice {p['clinic_id']} - "
                 f"classe prédite {p['predicted_class']} (probabilité {p['probability']}) - "
                 f"statut : {p['validation_status']}")

        st.subheader("Arbre de provenance")
        st.code("\n".join(trace(L, case)))

        st.subheader("Intégrité de l'explication")
        path, text, recalc = explanation_info(p)
        if path is None or text is None:
            st.error("Explication absente du stockage hors chaîne.")
        else:
            a, c = st.columns(2)
            a.caption("Empreinte ancrée dans le registre")
            a.code(p["explanation_hash"])
            c.caption("Empreinte recalculée sur le texte stocké")
            c.code(recalc)
            if recalc == p["explanation_hash"]:
                st.success("Explication intacte.")
            else:
                st.error("FALSIFICATION DÉTECTÉE : le texte a été modifié après son ancrage.")
            st.info(text)

            bak = path.with_suffix(".bak")
            f1, f2 = st.columns(2)
            if f1.button("Falsifier l'explication (démo)") and not bak.exists():
                shutil.copy(path, bak)
                data = json.loads(path.read_text(encoding="utf-8"))
                data["text"] = data["text"].split(" Facteurs")[0]
                path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                st.rerun()
            if f2.button("Restaurer l'explication") and bak.exists():
                shutil.copy(bak, path)
                bak.unlink()
                st.rerun()
    st.caption("Seules les empreintes sont dans le registre : ni poids, ni données de patients.")

# ------------------------------------------------------------ onglet 4 : blocs
with tabs[3]:
    st.dataframe(pd.DataFrame([{
        "Bloc": b["index"], "Heure (UTC)": b["timestamp"], "Type": b["record_type"],
        "Hash du bloc": short(b["block_hash"]), "Hash du bloc précédent": short(b["prev_hash"]),
    } for b in L.blocks]))
    st.caption("Chaque bloc contient le hash du précédent : modifier un ancien bloc casse la chaîne.")