import hashlib
import json
import numpy as np

HEADER = b"FEDTRACE-WEIGHTS-v1|float64-le|"


def fingerprint_weights(weights) -> str:
    """SHA-256 (hexadécimal) d'un vecteur de poids, canonicalisation stricte."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if not np.all(np.isfinite(w)):
        raise ValueError("Poids invalides : NaN ou infini, empreinte refusée")
    w = w + 0.0                      # transforme -0.0 en 0.0
    payload = HEADER + str(w.size).encode() + b"|" + w.astype("<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def fingerprint_text(text: str) -> str:
    """SHA-256 d'un texte (ex. l'explication d'un diagnostic)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_json(obj) -> str:
    """SHA-256 d'un dictionnaire (JSON canonique : clés triées, sans espaces)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()