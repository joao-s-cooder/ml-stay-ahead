from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np
from api.app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Passos Mágicos Lag Prediction API"}

@patch('api.app.model')
def test_predict_endpoint_success(mock_model):
    # 1. Mock do modelo para retornar uma previsão fixa
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    
    # 2. Payload de teste (com dados fictícios, mas no formato esperado)
    payload = {
        "idade_22": 15.0,
        "genero": "Menino",
        "instituicao_ensino": "Escola Pública",
        "pedra_22": "Ametista",
        "inde_22": 7.5,
        "iaa": 8.0,
        "ieg": 6.5,
        "ips": 7.0,
        "ida": 7.2,
        "matem": 6.0,
        "portug": 6.5,
        "ingles": 8.0
    }
    
    # 3. Enviamos a requisição POST para o endpoint de previsão
    response = client.post("/predict", json=payload)
    
    # 4. Validamos se passou pelo caminho feliz da API
    assert response.status_code == 200
    data = response.json()
    assert "risk_of_lag" in data
    assert "risk_probability" in data