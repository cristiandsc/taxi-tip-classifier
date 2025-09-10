# src/models/train_model.py
from __future__ import annotations
from typing import Iterable
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import MODEL_PATH  # importamos la ruta centralizada


def train_random_forest(
    df: pd.DataFrame,
    target_col: str,
    features: Iterable[str],
    *,
    n_estimators: int = 100,
    max_depth: int | None = 10,
    random_state: int = 42,
    model_path: str | None = None,
):
    """
    Entrena un RandomForestClassifier y guarda el modelo en disco.

    Args:
        df: DataFrame con datos preprocesados.
        target_col: Nombre de la columna objetivo (ej. 'high_tip').
        features: Lista de columnas predictoras.
        n_estimators: Número de árboles en el bosque.
        max_depth: Profundidad máxima de cada árbol.
        random_state: Semilla aleatoria para reproducibilidad.
        model_path: Ruta opcional donde guardar el modelo.
                    Si None, se usa config.MODEL_PATH.

    Returns:
        clf: Modelo entrenado.
        f1: F1-score en el conjunto de entrenamiento.
        model_path: Ruta final donde se guardó el modelo.
    """
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

    # Usar config.MODEL_PATH si no se pasa ruta
    if model_path is None:
        model_path = MODEL_PATH

    model_path = str(model_path)  # joblib.dump espera string o Path
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    return clf, f1, model_path

