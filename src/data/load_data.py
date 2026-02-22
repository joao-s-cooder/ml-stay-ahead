import pandas as pd
import os
from src.utils.paths import RAW_DATA_FILE

def load_raw_data(file_path=RAW_DATA_FILE):
    """
    Carrega os dados brutos do Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {file_path}")
    
    print(f"Carregando dados de {file_path}...")
    df = pd.read_excel(file_path)
    return df

if __name__ == "__main__":
    try:
        df = load_raw_data()
        print(f"Dados carregados. Shape: {df.shape}")
    except Exception as e:
        print(f"Erro: {e}")
