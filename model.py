# model.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["Age", "Gender", "RF", "Anti_CCP", "CRP", "ESR"]
LABEL = "label"
N_CLASSES = 3
CLINIQUES_DIR = Path("cliniques")
SCALER_PATH = CLINIQUES_DIR / "scaler.json"


# ---------- Données ----------
def _encode_gender(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    m = {"male": 1, "m": 1, "homme": 1, "female": 0, "f": 0, "femme": 0}
    return s.astype(str).str.strip().str.lower().map(m)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Gender"] = _encode_gender(df["Gender"])
    return df


def build_scaler():
    df = pd.concat(
        [_clean(pd.read_csv(CLINIQUES_DIR / f"clinique_{i}.csv")) for i in (1, 2, 3)]
    )
    med = df[FEATURES].median()
    filled = df[FEATURES].fillna(med)
    scaler = {
        "median": med.tolist(),
        "mean": filled.mean().tolist(),
        "std": filled.std(ddof=0).replace(0, 1).tolist(),
    }
    SCALER_PATH.write_text(json.dumps(scaler, indent=2))
    return scaler


def load_scaler():
    if SCALER_PATH.exists():
        sc = json.loads(SCALER_PATH.read_text())
        if "median" in sc:
            return sc
    return build_scaler()


def load_dataset(name):
    sc = load_scaler()
    df = _clean(pd.read_csv(CLINIQUES_DIR / f"{name}.csv"))
    Xdf = df[FEATURES].fillna(pd.Series(sc["median"], index=FEATURES))
    X = (Xdf.to_numpy(dtype=np.float64) - np.array(sc["mean"])) / np.array(sc["std"])
    y = df[LABEL].to_numpy(dtype=np.int64)
    return X, y

# ---------- Modèle ----------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_features=len(FEATURES), n_classes=N_CLASSES):
        self.n_features = n_features
        self.n_classes = n_classes
        self.W = np.zeros((n_features, n_classes), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

    # --- poids sous forme de vecteur plat (pour agréger et hasher) ---
    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b]).astype(np.float64)

    def set_weights(self, vec: np.ndarray):
        vec = np.asarray(vec, dtype=np.float64)
        k = self.n_features * self.n_classes
        self.W = vec[:k].reshape(self.n_features, self.n_classes).copy()
        self.b = vec[k:].copy()

    # --- entraînement local ---
    def fit(self, X, y, epochs=20, lr=0.1, batch_size=64, l2=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        n = len(X)
        Y = np.eye(self.n_classes)[y]
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b_idx = idx[s:s + batch_size]
                Xb, Yb = X[b_idx], Y[b_idx]
                P = softmax(Xb @ self.W + self.b)
                err = (P - Yb) / len(b_idx)
                self.W -= lr * (Xb.T @ err + l2 * self.W)
                self.b -= lr * err.sum(axis=0)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro")),
        }