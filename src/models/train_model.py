# src/models/train_model.py
from __future__ import annotations
from pathlib import Path
import joblib
from typing import Iterable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd

# Ruta a la raíz del repo (este archivo está en repo/src/models/train_model.py)
REPO_ROOT = Path(__file__).resolve().parents[2]

def train_random_forest(
    df: pd.DataFrame,
    target_col: str,
    features: Iterable[str],
    *,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    random_state: int = 42,
    model_path: str | Path | None = None,
):
    X = df[features]
    y = df[target_col]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)

    f1 = f1_score(y, clf.predict(X))

    # guardar en <repo>/models/random_forest.joblib por defecto
    if model_path is None:
        model_path = REPO_ROOT / "models" / "random_forest.joblib"
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (REPO_ROOT / model_path).resolve()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    return clf, f1, model_path

