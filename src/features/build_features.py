# src/features/build_features.py
from __future__ import annotations
import pandas as pd

# === Listas de features ===
numeric_feat = [
    "pickup_weekday",
    "pickup_hour",
    "work_hours",
    "pickup_minute",
    "passenger_count",
    "trip_distance",
    "trip_time",
    "trip_speed",
]

categorical_feat = [
    "PULocationID",
    "DOLocationID",
    "RatecodeID",
]

# Unión para el modelo
features = numeric_feat + categorical_feat

# Constante pequeña para evitar división por cero
EPS = 1e-7


def preprocess(df: pd.DataFrame, target_col: str = "high_tip") -> pd.DataFrame:
    """
    Aplica el mismo preprocesamiento que en el notebook 00:

    - Filtra filas con fare_amount > 0
    - Crea 'tip_fraction' = tip_amount / fare_amount
    - Crea variable binaria objetivo target_col (propina > 20%)
    - Genera features de tiempo: weekday, hour, minute, work_hours
    - Calcula trip_time (segundos) y trip_speed (distance / segundos)
    - Convierte features a float32 (rellena NaN con -1.0) y target a int32
    - Devuelve solo columnas del modelo: features + target_col

    Notes:
    - Usa las columnas originales 'tpep_pickup_datetime' y 'tpep_dropoff_datetime'.
    """
    df = df.copy()

    # --- Limpieza básica ---
    df = df[df["fare_amount"] > 0].reset_index(drop=True)

    # --- Target ---
    df["tip_fraction"] = df["tip_amount"] / df["fare_amount"]
    df[target_col] = (df["tip_fraction"] > 0.20).astype(int)  # binaria 0/1

    # --- Asegurar datetime ---
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # --- Features de tiempo  ---
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_minute"] = df["tpep_pickup_datetime"].dt.minute
    # Días hábiles (0-4) y horas de trabajo [8, 18]
    df["work_hours"] = (
        (df["pickup_weekday"] >= 0)
        & (df["pickup_weekday"] <= 4)
        & (df["pickup_hour"] >= 8)
        & (df["pickup_hour"] <= 18)
    ).astype(int)

    # Duración del viaje (segundos) y velocidad aproximada
    df["trip_time"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.seconds
    df["trip_speed"] = df["trip_distance"] / (df["trip_time"] + EPS)

    # --- Conservar solo columnas del modelo ---
    # (opcionalmente puedo descartar columnas que no use)
    cols = features + [target_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el modelo: {missing}")

    df = df[cols].copy()

    # --- Tipos y NaN ---
    df[features] = df[features].astype("float32").fillna(-1.0)
    df[target_col] = df[target_col].astype("int32")

    return df.reset_index(drop=True)
