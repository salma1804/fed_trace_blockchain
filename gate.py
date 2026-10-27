# // 9bal mayt3ada diagnostic ithabet keno jey mn modele certifié mawjoud fl legneder=registre ou nn

from fingerprint import fingerprint_weights
from diagnosis import diagnose


class CertificationError(Exception):
    """Levée quand la porte refuse un diagnostic."""


def certify_model(ledger, model, check_chain=True):
    """Renvoie (True, bloc_du_modèle) si certifié, sinon (False, raison)."""
    if check_chain:
        ok, msg = ledger.verify_chain()
        if not ok:
            return False, f"Registre invalide, impossible de certifier : {msg}"
    try:
        h = fingerprint_weights(model.get_weights())
    except ValueError as e:
        return False, f"Poids invalides : {e}"
    found = ledger.find("global_model", model_hash=h)
    if not found:
        return False, f"Modèle NON CERTIFIÉ : l'empreinte {h[:12]}... n'est ancrée dans aucun round"
    block = found[0]
    if block["payload"]["round"] == 0:
        return False, "Modèle initial non entraîné : aucun round de fédération ne l'a produit"
    return True, block


def certified_diagnose(ledger, model, raw_case, patient_ref, clinic_id=1):
    """Diagnostic protégé par la porte : refusé si le modèle n'est pas certifié."""
    ok, info = certify_model(ledger, model)
    if not ok:
        raise CertificationError(info)
    result = diagnose(ledger, model, raw_case, patient_ref, clinic_id)
    result["certified_round"] = info["payload"]["round"]
    return result