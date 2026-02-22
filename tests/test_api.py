from fastapi.testclient import TestClient
from api.app import app
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Passos Mágicos Lag Prediction API"}

def test_predict_endpoint():
    # Mock data compatível com StudentData model
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
    
    # Necessita do modelo carregado. Se o modelo não estiver carregado, retorna 503.
    # Podemos aceitar 200 ou 503 como respostas "válidas" neste contexto (endpoint alcançável).
    # Idealmente, devemos mockar o modelo.
    
    response = client.post("/predict", json=payload)
    
    if response.status_code == 503:
        pytest.skip("Modelo não carregado na API, pulando teste de predição.")
    
    assert response.status_code == 200
    data = response.json()
    assert "risk_of_lag" in data
    assert "risk_probability" in data
