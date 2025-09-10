# src/data/dataset.py
from pathlib import Path
import pandas as pd
from src import config

def load_data(path: str | Path, filetype: str = "parquet") -> pd.DataFrame:
    """
    Carga un dataset de taxis desde archivo local o URL.

    Args:
        path (str | Path): ruta local o URL al archivo.
        filetype (str): 'parquet' o 'csv'.

    Returns:
        pd.DataFrame: dataframe con los datos cargados.
    """
    path = Path(path)

    try:
        if filetype == "parquet":
            return pd.read_parquet(path)
        elif filetype == "csv":
            return pd.read_csv(path)
        else:
            raise ValueError("filetype debe ser 'parquet' o 'csv'")
    except Exception as e:
        raise RuntimeError(f"Error cargando datos desde {path} ({filetype}): {e}")


