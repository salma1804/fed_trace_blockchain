# // hetha how l registre =bdd mtablockchain linsajlfih les lhash mta kol clinique w hash global  f kol round
import json
import os
import time
from pathlib import Path
from fingerprint import fingerprint_json

GENESIS_PREV = "0" * 64


def _block_hash(block: dict) -> str:
    """Hash d'un bloc = SHA-256 de son contenu (sans le champ block_hash)."""
    content = {k: v for k, v in block.items() if k != "block_hash"}
    return fingerprint_json(content)


class Ledger:
    def __init__(self, path="ledger.jsonl"):
        self.path = Path(path)
        self.blocks = []
        if self.path.exists():
            self._load()
        else:
            self.path.touch()
        if not self.blocks:
            self.append("genesis", {"note": "FedTrace ledger"})

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.blocks.append(json.loads(line))

    def append(self, record_type: str, payload: dict) -> dict:
        """Ajoute un bloc à la fin de la chaîne et l'écrit sur disque."""
        block = {
            "index": len(self.blocks),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prev_hash": self.blocks[-1]["block_hash"] if self.blocks else GENESIS_PREV,
            "record_type": record_type,
            "payload": payload,
        }
        block["block_hash"] = _block_hash(block)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())          # écriture réellement sur disque
        self.blocks.append(block)
        return block

    def verify_chain(self):
        """Renvoie (True, None) si la chaîne est intacte, sinon (False, message)."""
        prev = GENESIS_PREV
        for i, b in enumerate(self.blocks):
            if b["index"] != i:
                return False, f"Bloc {i} : index incorrect (bloc supprimé ou inséré ?)"
            if b["prev_hash"] != prev:
                return False, f"Bloc {i} : le lien avec le bloc précédent est rompu"
            if _block_hash(b) != b["block_hash"]:
                return False, f"Bloc {i} : contenu modifié (hash recalculé différent)"
            prev = b["block_hash"]
        return True, None

    def find(self, record_type=None, **payload_filters):
        """Cherche des blocs par type et/ou par valeurs dans le payload."""
        out = []
        for b in self.blocks:
            if record_type and b["record_type"] != record_type:
                continue
            if all(b["payload"].get(k) == v for k, v in payload_filters.items()):
                out.append(b)
        return out

    def __len__(self):
        return len(self.blocks)