# src/models/predict_model.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# Módulos del repo
from config import MODEL_PATH, REPORTS_DIR  # 👈 rutas centralizadas
from data.dataset import load_data
from features.build_features import preprocess, features as FEATURE_LIST


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELO
# ──────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str | Path | None = None):
    """
    Carga el modelo .joblib. Por defecto usa config.MODEL_PATH.
    Acepta rutas relativas al repo o absolutas.
    """
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.is_absolute():
        # Si el usuario pasa algo relativo, resolvemos a absoluta respecto al cwd
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────────
# PREPARACIÓN DE DATASETS
# ──────────────────────────────────────────────────────────────────────────────
def make_dataset_from_raw(
    df_raw: pd.DataFrame, target_col: str = "high_tip"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Preprocesa un dataframe crudo (como el de la URL de febrero) y devuelve:
      - X: dataframe con features,
      - y: vector objetivo,
      - df_proc: dataframe completo preprocesado (por inspección/guardado).
    """
    df_proc = preprocess(df_raw, target_col=target_col)
    X = df_proc[FEATURE_LIST]
    y = df_proc[target_col]
    return X, y, df_proc


def make_dataset_from_parquet(
    path: str | Path, target_col: str = "high_tip"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Carga un parquet YA preprocesado (p. ej. data/processed/taxi_train.parquet)
    y devuelve X, y y el dataframe.
    """
    df_proc = load_data(str(path), filetype="parquet")
    X = df_proc[FEATURE_LIST]
    y = df_proc[target_col]
    return X, y, df_proc


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────────────
def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Devuelve la probabilidad de clase positiva (columna 1 de predict_proba)."""
    proba = model.predict_proba(X)
    return proba[:, 1]


def predict_labels_from_proba(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades en etiquetas 0/1 usando un umbral."""
    return (proba >= threshold).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_binary(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: Optional[Iterable[float]] = None,
) -> Dict:
    """
    Calcula métricas de clasificación binaria y devuelve un dict con:
      precision, recall, f1, support_pos, support_total, confusion_matrix, classification_report
    """
    y_true_arr = np.asarray(list(y_true))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred)
    report = classification_report(y_true_arr, y_pred, digits=3, zero_division=0)

    # soportes
    support_pos = int((y_true_arr == 1).sum())
    support_total = int(y_true_arr.size)

    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support_pos": support_pos,
        "support_total": support_total,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    if y_proba is not None:
        y_proba = np.asarray(list(y_proba))
        out["proba_head"] = [float(x) for x in y_proba[:10]]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS (guardar artefactos)
# ──────────────────────────────────────────────────────────────────────────────
def save_metrics_json(metrics: Dict, path: str | Path):
    """Guarda métricas como JSON (útil en reports/metrics_eval.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def save_predictions_csv(
    df_source: pd.DataFrame,
    proba: Iterable[float],
    labels: Iterable[int],
    path: str | Path,
    id_cols: Optional[list[str]] = None,
):
    """
    Guarda un CSV con columnas: id_cols (opcionales) + proba + label.
    Si no hay id_cols, incluirá el índice como 'row_id'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame({"proba": list(proba), "label": list(labels)})
    if id_cols:
        out = pd.concat([df_source[id_cols].reset_index(drop=True), out], axis=1)
    else:
        out.insert(0, "row_id", np.arange(len(out)))

    out.to_csv(path, index=False)

