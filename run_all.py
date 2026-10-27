"""Valide tout le projet d'un coup.

    python run_all.py            vérifie la version locale (registre Python)
    python run_all.py --chain    vérifie aussi le smart contract (nœud Hardhat lancé)

Écrit le détail complet dans validation_log.txt.
"""
import contextlib
import io
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

CORE_FILES = [
    "split_cliniques.py", "model.py", "federation.py", "fingerprint.py", "ledger.py",
    "anchored_federation.py", "diagnosis.py", "gate.py", "aggregation.py", "anchor_agent.py",
    "scenarios.py", "app.py", "benchmark.py", "chain_client.py", "test_chain.py",
    "chain/contracts/FedTrace.sol",
]
DATA_FILES = ["cliniques/clinique_1.csv", "cliniques/clinique_2.csv", "cliniques/clinique_3.csv",
              "cliniques/val_numeric.csv", "cliniques/test_numeric.csv"]

# (libellé, script, textes attendus dans la sortie, nombre minimum de "True")
STEPS = [
    ("Modèle : entraînement et déterminisme", "test_model.py", ["Poids identiques ? True"], 0),
    ("Empreintes SHA-256 (7 tests)", "test_fingerprint.py", [], 7),
    ("Fédération : 10 rounds, fédéré vs centralisé", "federation.py",
     ["Modèle fédéré", "Modèle centralisé", "history.json sauvegardé"], 0),
    ("Registre : chaîne de blocs (tests de falsification)", "test_ledger.py", [], 6),
    ("Fédération ancrée (42 blocs)", "anchored_federation.py",
     ["Blocs dans le registre : 42", "Chaîne valide : True"], 0),
    ("Diagnostic, explication, provenance", "test_diagnosis.py", [], 7),
    ("Les 5 scénarios du cahier des charges", "scenarios.py",
     [f"Scénario {i} : RÉUSSI" for i in range(1, 6)], 0),
]
if "--chain" in sys.argv:
    STEPS.append(("Smart contract sur nœud Hardhat", "test_chain.py", ["Partie A", "Partie B"], 0))

LOG = []


def run_script(script, required, min_true, timeout=900):
    if not Path(script).exists():
        return "ABSENT", 0.0, ""
    t0 = time.time()
    try:
        p = subprocess.run([PY, script], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=ENV, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "ÉCHEC : délai dépassé", time.time() - t0, ""
    out = p.stdout + p.stderr
    problems = []
    if p.returncode != 0:
        problems.append(f"code de sortie {p.returncode}")
    if "Traceback" in out:
        problems.append("erreur Python (voir le journal)")
    if "ÉCHEC" in out:
        problems.append("une vérification affiche ÉCHEC")
    if re.search(r"\bFalse\b", out):
        problems.append("une vérification affiche False")
    for r in required:
        if r not in out:
            problems.append(f"texte attendu absent : {r}")
    if min_true and out.count("True") < min_true:
        problems.append(f"moins de {min_true} résultats True")
    status = "OK" if not problems else "ÉCHEC : " + " ; ".join(problems)
    return status, time.time() - t0, out


def check_robust_no_false_rejection():
    """Agrégation robuste sur une fédération 100 % honnête : aucune clinique ne doit être écartée."""
    from aggregation import median_with_exclusion
    from federation import run_federation

    rejets = []

    def agg(updates):
        w, _retained, discarded = median_with_exclusion(updates)
        rejets.append(discarded)
        return w

    with contextlib.redirect_stdout(io.StringIO()):
        run_federation(n_rounds=10, aggregate=agg)
    n = sum(1 for d in rejets if d)
    return n == 0, f"faux rejets : {n}/10 rounds"


def main():
    t_all = time.time()
    print(f"Validation FedTrace - Python {platform.python_version()} - {time.strftime('%Y-%m-%d %H:%M')}")

    manquants = [f for f in CORE_FILES if not Path(f).exists()]
    data_manquantes = [f for f in DATA_FILES if not Path(f).exists()]
    print("\nFichiers du projet absents :", ", ".join(manquants) if manquants else "aucun")
    if data_manquantes:
        print("Données absentes :", ", ".join(data_manquantes), "\n -> lance d'abord : python split_cliniques.py")

    rows = []
    for label, script, required, min_true in STEPS:
        print(f"... {label} ({script})", flush=True)
        status, dt, out = run_script(script, required, min_true)
        rows.append((label, script, status, dt))
        LOG.append(f"{'=' * 70}\n{label} ({script}) -> {status}\n{'=' * 70}\n{out}\n")

    print("... Agrégation robuste sans attaque (faux rejets)", flush=True)
    try:
        ok, detail = check_robust_no_false_rejection()
        status = "OK" if ok else f"ÉCHEC : {detail}"
    except Exception as e:                                   # noqa: BLE001
        status, detail = f"ÉCHEC : {e}", str(e)
    rows.append(("Agrégation robuste sans attaque", "(en direct)", status, 0.0))
    LOG.append(f"Agrégation robuste sans attaque -> {status} ({detail})\n")

    print("\n" + "=" * 78)
    print(f"{'Vérification':<52}{'Résultat':<12}{'Durée':>8}")
    print("-" * 78)
    for label, script, status, dt in rows:
        court = status if len(status) <= 10 else status.split(" :")[0]
        print(f"{label:<52}{court:<12}{dt:>6.1f} s")
        if status not in ("OK", "ABSENT"):
            print(f"     -> {status}")
    print("=" * 78)

    ok = sum(1 for r in rows if r[2] == "OK")
    ko = sum(1 for r in rows if r[2].startswith("ÉCHEC"))
    ab = sum(1 for r in rows if r[2] == "ABSENT")
    print(f"{ok} réussie(s), {ko} échec(s), {ab} absente(s) - {time.time() - t_all:.0f} s")
    Path("validation_log.txt").write_text(
        "\n".join(LOG) + f"\nRésumé : {ok} OK, {ko} ÉCHEC, {ab} ABSENT\n", encoding="utf-8")
    print("Détail complet dans validation_log.txt")
    sys.exit(1 if ko else 0)


if __name__ == "__main__":
    main()