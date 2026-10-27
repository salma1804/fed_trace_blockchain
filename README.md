# FedTrace

Prototype pédagogique : **ancrage et vérification de la provenance d'un apprentissage fédéré**
pour le diagnostic de maladies auto-immunes (classes : sain, polyarthrite rhumatoïde
séropositive, polyarthrite rhumatoïde séronégative).

Trois cliniques simulées entraînent le même modèle sur leurs propres données. Un registre
(blockchain) garde l'empreinte de chaque contribution (avant l'agrégation), du modèle global
(après) et de chaque diagnostic. Une porte de certification refuse tout diagnostic fait avec un
modèle dont l'empreinte n'est pas dans le registre.

> **Prototype d'école : ne pas utiliser avec de vraies données de patients.**

## Prérequis

- Python 3.13 (version utilisée pour les tests)
- Git
- Node.js 22 ou plus (uniquement pour la partie smart contract)

## Installation (CMD Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Placer à la racine du projet `train_numeric.csv`, `val_numeric.csv` et `test_numeric.csv`
(colonnes : Age, Gender, RF, Anti_CCP, CRP, ESR, label ; label 0 = sain, 1 = séropositive,
2 = séronégative).

Source et licence du jeu de données : **à compléter**.

## Ordre d'exécution

```
python split_cliniques.py      # découpe l'entraînement en 3 cliniques (dossier cliniques/)
python run_all.py              # valide tout : tests, 5 scénarios, agrégation robuste
python benchmark.py            # mesures (surcoût, volume, détection) -> mesures.json
streamlit run app.py           # interface d'audit
```

## Smart contract (Hardhat + Solidity)

```
cd chain
npm install
npx hardhat build
npx hardhat node               # à laisser ouvert dans un autre terminal
```

Puis, dans le dossier principal :

```
python run_all.py --chain
```

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| `split_cliniques.py` | Découpe les données en 3 cliniques (stratifié) |
| `model.py` | Prétraitement et modèle (régression logistique, 21 poids) |
| `federation.py` | Entraînement local et agrégation par moyenne pondérée |
| `fingerprint.py` | Empreintes SHA-256 reproductibles |
| `ledger.py` | Registre local : blocs chaînés par hash |
| `anchored_federation.py` | Fédération avec ancrage (contributions avant, modèle après) |
| `aggregation.py` | Agrégation robuste (médiane avec exclusion) et attaque simulée |
| `diagnosis.py` | Diagnostic, explication, pseudonyme, provenance |
| `gate.py` | Porte de certification |
| `anchor_agent.py` | Agent d'ancrage avec file de reprise |
| `scenarios.py` | Les 5 scénarios du cahier des charges |
| `app.py` | Interface d'audit (Streamlit) |
| `benchmark.py` | Mesures pour le rapport |
| `run_all.py` | Validation complète en une commande |
| `chain/contracts/FedTrace.sol` | Smart contract |
| `chain_client.py`, `test_chain.py` | Client Python du contrat et ses tests |

## Limites connues

- Fédération simulée : une seule machine, données découpées au hasard (cas IID).
- Registre local : un seul fichier ; la suppression du dernier bloc n'est pas détectée.
- Nœud Hardhat unique : ni réplication ni consensus démontrés.
- L'agrégation robuste tolère 1 clinique malveillante sur 3 ; le seuil d'exclusion est un réglage.
- La blockchain prouve la provenance, pas la qualité médicale du modèle.
- Pas de révocation de modèle, pas de signature des contributions par les cliniques, contrat non audité.
