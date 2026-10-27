# L’anchor_agent.py est comme un facteur qui garantit que rien ne se perd lors de l’enregistrement.
# Il fait 3 choses :
# 1. Il prépare l’ancrage et le sauvegarde dans un fichier anchor_queue.jsonl.
# 2. Il essaie de l’enregistrer dans le registre.
# 3. S’il échoue, il le garde en attente et réessaie lorsque le registre redevient disponible.
# mellokhr kbal maydkhl fl registre  yarmih f anchor_queue w bad ki idakhl ithbat kn lid mta li
# mch idakhlo mawjoud aka may3wdch idakhlo snn idakhl mch info matyt9aydch martin 



import json
import os
import uuid
from pathlib import Path


class RegistryUnavailable(Exception):
    """Le registre ne répond pas."""


class AnchorAgent:
    """Intercalé entre la fédération et le registre. Même interface que Ledger.append."""

    def __init__(self, ledger, queue_path="anchor_queue.jsonl"):
        self.ledger = ledger
        self.queue_path = Path(queue_path)
        self.queue_path.touch(exist_ok=True)

    def _load_queue(self):
        text = self.queue_path.read_text(encoding="utf-8")
        return [json.loads(l) for l in text.splitlines() if l.strip()]

    def _save_queue(self, items):
        tmp = self.queue_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.queue_path)

    def pending(self):
        return len(self._load_queue())

    def append(self, record_type, payload):
        item = {"anchor_id": uuid.uuid4().hex, "record_type": record_type, "payload": payload}
        with open(self.queue_path, "a", encoding="utf-8") as f:   # 1. durable d'abord
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.flush()                                               # 2. puis on essaie d'écrire
        found = self.ledger.find(record_type, anchor_id=item["anchor_id"])
        return found[0] if found else {"index": None, "queued": True}

    def flush(self):
        """Rejoue la file dans l'ordre. Renvoie le nombre d'éléments écrits."""
        items = self._load_queue()
        remaining = list(items)
        for it in items:
            try:
                # idempotence : déjà écrit (plantage avant le nettoyage de la file) ?
                if not self.ledger.find(it["record_type"], anchor_id=it["anchor_id"]):
                    self.ledger.append(it["record_type"],
                                       dict(it["payload"], anchor_id=it["anchor_id"]))
            except (RegistryUnavailable, OSError):
                break
            remaining.pop(0)
        self._save_queue(remaining)
        return len(items) - len(remaining)