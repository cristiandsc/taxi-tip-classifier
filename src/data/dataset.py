# src/data/dataset.py
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

def load_data(path: str | Path, filetype: str = "parquet") -> pd.DataFrame:
    """
    Carga un dataset de taxis desde archivo local o URL (http/https).

    Args:
        path (str | Path): ruta local o URL al archivo.
        filetype (str): 'parquet' o 'csv'.

    Returns:
        pd.DataFrame: dataframe con los datos cargados.
    """
    # Detectar si es URL
    is_url = isinstance(path, str) and urlparse(path).scheme in {"http", "https"}
    path_in = path if is_url else str(Path(path))

    if filetype == "parquet":
        return pd.read_parquet(path_in)  # requiere pyarrow o fastparquet
    elif filetype == "csv":
        return pd.read_csv(path_in)
    else:
        raise ValueError("filetype debe ser 'parquet' o 'csv'")

