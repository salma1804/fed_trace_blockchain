# t5ali lcode python y3raf win l smart contact(festrace.sol)


import json
from pathlib import Path

from web3 import Web3

RPC_URL = "http://127.0.0.1:8545"


def _find_artifact():
    cands = [p for p in Path("chain/artifacts").rglob("FedTrace.json") if "FedTrace.sol" in str(p)]
    if not cands:
        raise FileNotFoundError(
            "Artefact FedTrace.json introuvable. Compile le contrat : "
            "cd chain puis npx hardhat build (ou npx hardhat compile)")
    return json.loads(cands[0].read_text(encoding="utf-8"))


def h2b(h: str) -> bytes:
    """Empreinte SHA-256 hexadécimale (64 caractères) -> bytes32."""
    return bytes.fromhex(h)


def b2h(b) -> str:
    return bytes(b).hex()


def mask(ids) -> int:
    m = 0
    for i in ids:
        m |= 1 << int(i)
    return m


def ids_from_mask(m: int):
    return [i for i in range(1, 256) if (m >> i) & 1]


def case_id(label: str) -> bytes:
    """Clé d'un cas = hash de son pseudonyme (bytes32)."""
    return Web3.keccak(text=label)


class ChainClient:
    """Rôles : compte 0 = administrateur, 1 = serveur d'agrégation,
    2/3/4 = cliniques 1/2/3, 5 = inconnu (aucun droit)."""

    def __init__(self, url=RPC_URL):
        self.w3 = Web3(Web3.HTTPProvider(url))
        if not self.w3.is_connected():
            raise ConnectionError("Noeud injoignable. Lance 'npx hardhat node' dans le dossier chain "
                                  "(dans un autre terminal) puis relance.")
        art = _find_artifact()
        self.abi, self.bytecode = art["abi"], art["bytecode"]
        acc = self.w3.eth.accounts
        self.owner, self.server = acc[0], acc[1]
        self.clinics = {1: acc[2], 2: acc[3], 3: acc[4]}
        self.outsider = acc[5]
        self.contract = None
        self.gas_used = []

    # ------------------------------------------------------------ transactions
    def _send(self, fn, sender):
        txh = fn.transact({"from": sender})
        rc = self.w3.eth.wait_for_transaction_receipt(txh)
        if rc.status != 1:
            raise RuntimeError("transaction échouée")
        self.gas_used.append(rc.gasUsed)
        return rc

    def deploy(self):
        """Déploie une nouvelle instance du contrat et configure les identités."""
        C = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        rc = self._send(C.constructor(), self.owner)
        self.contract = self.w3.eth.contract(address=rc.contractAddress, abi=self.abi)
        self._send(self.contract.functions.setServer(self.server), self.owner)
        for cid, account in self.clinics.items():
            self._send(self.contract.functions.registerClinic(account, cid), self.owner)
        return rc.contractAddress

    def anchor_initial_model(self, model_hash, sender=None):
        return self._send(self.contract.functions.anchorInitialModel(h2b(model_hash)),
                          sender or self.server)

    def anchor_contribution(self, clinic_id, round_, weights_hash, base_hash, n_samples, sender=None):
        return self._send(
            self.contract.functions.anchorContribution(
                round_, h2b(weights_hash), h2b(base_hash), int(n_samples)),
            sender or self.clinics[clinic_id])

    def anchor_model(self, round_, model_hash, parent_hash, strategy, retained, discarded,
                     metrics, sender=None):
        data = (round_, h2b(model_hash), h2b(parent_hash), strategy, mask(retained), mask(discarded),
                int(round(metrics.get("accuracy", 0) * 10000)),
                int(round(metrics.get("f1_macro", 0) * 10000)))
        return self._send(self.contract.functions.anchorModel(data), sender or self.server)

    def anchor_decision(self, clinic_id, case, model_hash, explanation_hash, predicted_class,
                        probability, status=0, sender=None):
        return self._send(
            self.contract.functions.anchorDecision(
                case_id(case), h2b(model_hash), h2b(explanation_hash),
                int(predicted_class), int(round(probability * 10000)), int(status)),
            sender or self.clinics[clinic_id])

    # ------------------------------------------------------------ lectures
    def is_certified(self, model_hash) -> bool:
        return self.contract.functions.isCertified(h2b(model_hash)).call()

    def verify_model(self, model_hash) -> dict:
        certified, round_, retained, discarded, acc = \
            self.contract.functions.verifyModel(h2b(model_hash)).call()
        return {"certified": certified, "round": round_, "retained": ids_from_mask(retained),
                "discarded": ids_from_mask(discarded), "accuracy": acc / 10000}

    def verify_decision(self, case, explanation_hash) -> bool:
        return self.contract.functions.verifyDecision(case_id(case), h2b(explanation_hash)).call()

    def provenance(self, model_hash):
        hashes, rounds = self.contract.functions.getProvenance(h2b(model_hash)).call()
        return [(int(r), b2h(h)) for h, r in zip(hashes, rounds)]

    def round_contributions(self, round_):
        ids, wh, bh, ns = self.contract.functions.getRoundContributions(round_).call()
        return [{"clinic_id": int(i), "weights_hash": b2h(w), "base_hash": b2h(b), "n_samples": int(n)}
                for i, w, b, n in zip(ids, wh, bh, ns)]

    def decisions(self, case):
        return self.contract.functions.getDecisions(case_id(case)).call()
        