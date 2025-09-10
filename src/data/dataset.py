import pandas as pd

def load_data(path, filetype="parquet"):
    """
    Carga un dataset de taxis desde archivo local o URL.
    
    Args:
        path (str): ruta local o URL al archivo.
        filetype (str): 'parquet' o 'csv'.
    
    Returns:
        pd.DataFrame: dataframe con los datos cargados.
    """
    if filetype == "parquet":
        return pd.read_parquet(path)
    elif filetype == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError("filetype debe ser 'parquet' o 'csv'")

