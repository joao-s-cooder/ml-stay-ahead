import pytest
import pandas as pd
from unittest.mock import patch
from src.data.load_data import load_raw_data

@patch('src.data.load_data.pd.read_excel') 
def test_load_raw_data_mocked(mock_read):
    # Criamos um DataFrame de exemplo para ser retornado pelo mock
    mock_df = pd.DataFrame({
        'RA': [123, 456], 
        'Fase': [1, 2], 
        'Turma': ['A', 'B']
    })
    mock_read.return_value = mock_df
    
    # Chamamos a função
    df = load_raw_data()
    
    # Verificamos
    assert not df.empty
    assert 'RA' in df.columns
    mock_read.assert_called_once()