# tsna3 l cle prive f key/cliniuque 1 w cle public f team/clinique1.json
"""Crée la clé privée d'un participant et publie son adresse publique.

    python make_keys.py clinique1      (ou clinique2, clinique3, serveur)
    python make_keys.py --list         (voir quelles adresses sont déjà publiées)

- La clé privée est écrite dans keys/<rôle>.key : elle reste sur TON portable.
  Ne la partage jamais, ne la mets jamais dans Git (le .gitignore ignore *.key).
- L'adresse publique est écrite dans team/<rôle>.json : c'est CE fichier qu'on partage.
- Si la clé existe déjà, elle n'est jamais écrasée.
"""
import json
import sys
from pathlib import Path

from eth_account import Account

ROLES = ["clinique1", "clinique2", "clinique3", "serveur"]
KEY_DIR = Path("keys")
TEAM_DIR = Path("team")


def choisir_role():
    if len(sys.argv) > 1:
        role = sys.argv[1].strip().lower()
    else:
        print("Rôles possibles :", ", ".join(ROLES))
        role = input("Ton rôle : ").strip().lower()
    if role not in ROLES:
        sys.exit(f"Rôle inconnu : {role}. Choisis parmi : {', '.join(ROLES)}")
    return role


def verifier_gitignore():
    gi = Path(".gitignore")
    if not gi.exists() or "*.key" not in gi.read_text(encoding="utf-8"):
        print("\nATTENTION : ton .gitignore ne contient pas '*.key'.")
        print("Lance d'abord : git pull origin main  (ou python setup_repo.py)")
        print("avant de faire un 'git add', sinon ta clé privée pourrait partir dans Git.")


def lister():
    print("Adresses publiées dans team/ :")
    for role in ROLES:
        p = TEAM_DIR / f"{role}.json"
        if p.exists():
            print(f"  {role:<10} {json.loads(p.read_text(encoding='utf-8'))['address']}")
        else:
            print(f"  {role:<10} (pas encore publiée)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        lister()
        return
    role = choisir_role()
    KEY_DIR.mkdir(exist_ok=True)
    TEAM_DIR.mkdir(exist_ok=True)
    key_path = KEY_DIR / f"{role}.key"

    if key_path.exists():
        acct = Account.from_key(key_path.read_text(encoding="utf-8").strip())
        print("Une clé existe déjà pour ce rôle : elle n'est PAS écrasée.")
    else:
        acct = Account.create()
        key_path.write_text("0x" + bytes(acct.key).hex(), encoding="utf-8")
        print("Nouvelle clé créée.")

    public = TEAM_DIR / f"{role}.json"
    public.write_text(json.dumps({"role": role, "address": acct.address}, indent=2), encoding="utf-8")

    print(f"\nRôle              : {role}")
    print(f"Adresse publique  : {acct.address}")
    print(f"Clé privée        : {key_path}   <- SECRÈTE, reste sur ton portable")
    print(f"Fichier à partager: {public}   <- le seul à ajouter dans Git")
    verifier_gitignore()


if __name__ == "__main__":
    main()