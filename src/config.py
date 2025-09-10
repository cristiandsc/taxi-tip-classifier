# src/config.py
from pathlib import Path

# Raíz del proyecto (…/taxi-tip-classifier)
ROOT = Path(__file__).resolve().parents[1]

# Carpetas importantes
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Rutas por defecto útiles (opcional)
TRAIN_PARQUET = PROCESSED_DIR / "taxi_train.parquet"
MODEL_PATH    = MODELS_DIR / "random_forest.joblib"
