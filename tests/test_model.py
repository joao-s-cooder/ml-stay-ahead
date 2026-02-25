import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from src.models.predict_model import make_prediction
from src.data.preprocess import preprocess_data, build_preprocessing_pipeline
from src.models.train_model import train_model
from src.models.monitor_drift import generate_drift_dashboard
from src.utils.paths import ARTIFACTS_DIR

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

@pytest.fixture
# Cria um DataFrame de exemplo para os testes
def sample_data():
    data = {
        'Defas': [-1.0, 0.0, -2.0, 0.0, -0.5],
        'Idade 22': [10, 11, 12, 10, 11],
        'Gênero': ['M', 'F', 'M', 'F', 'M'],
        'Instituição de ensino': ['Publica', 'Privada', 'Publica', 'Publica', 'Privada'],
        'Pedra 22': ['Ametista', 'Topazio', 'Agata', 'Quartzo', 'Ametista'],
        'INDE 22': [7.5, 8.5, 6.0, 5.0, 7.0],
        'IAA': [8.0, 9.0, 7.0, 6.0, 8.0],
        'IEG': [7.0, 8.0, 6.0, 5.0, 7.0],
        'IPS': [7.5, 8.5, 6.5, 5.5, 7.5],
        'IDA': [7.0, 8.0, 6.0, 5.0, 7.0],
        'Matem': [6.0, 7.0, 5.0, 4.0, 6.0],
        'Portug': [6.0, 7.0, 5.0, 4.0, 6.0],
        'Inglês': [6.0, 7.0, 5.0, 4.0, 6.0]
    }
    return pd.DataFrame(data)

# Testa a função de pré-processamento de dados
def test_preprocess_data(sample_data):
    X, y, num_cols, cat_cols = preprocess_data(sample_data)
    assert X.shape[0] == 5
    assert len(y) == 5
    assert 'Defas' not in X.columns
    assert 'INDE 22' in X.columns

# Testa a função de construção do pipeline de pré-processamento

# Testa a função de orquestração do treinamento simulando dependências pesadas
@patch('src.models.train_model.joblib.dump')
@patch('src.models.train_model.mlflow')
@patch('src.models.train_model.RandomizedSearchCV')
@patch('src.models.train_model.load_raw_data')

def test_model_training_runs(mock_load_data, mock_random_search, mock_mlflow, mock_joblib_dump, sample_data):
    """
    Testa a orquestração do modelo (train_model).
    Usamos mocks para evitar que o teste carregue os dados reais do CSV, 
    treine o RandomForest de verdade ou grave no disco.
    """
    # 1. Injeta nosso DataFrame pequeno (5 linhas) no lugar do dataset real
    mock_load_data.return_value = sample_data
    
    # 2. Configura o que o RandomizedSearchCV falso vai retornar
    mock_search_instance = mock_random_search.return_value
    mock_search_instance.best_params_ = {'n_estimators': 100, 'max_depth': 10}
    mock_search_instance.best_score_ = 0.95
    
    # Criamos um "melhor modelo" de mentira que sabe fazer predições
    class DummyBestEstimator:
        def predict(self, X):
            # Retorna uma array de '1's do mesmo tamanho dos dados de teste
            return np.ones(len(X)) 
            
    mock_search_instance.best_estimator_ = DummyBestEstimator()
    
    # 3. Chama a sua função original (que não sabe que está sendo "enganada")
    train_model()
    
    # 4. Validamos se o fluxo de engenharia passou pelos pontos vitais
    mock_load_data.assert_called_once()
    mock_random_search.assert_called_once()
    mock_search_instance.fit.assert_called_once()
    mock_joblib_dump.assert_called_once()

# Testa se o artifact do modelo existe
def test_model_artifact_exists():
    if os.path.exists(MODEL_PATH):
        assert True
    else:
        pytest.skip("Artifact do modelo não encontrado, pulando teste de artifact.")

# Testa a função de previsão do modelo usando um mock para o pipeline de pré-processamento e o modelo
def test_predict_model_function():
    fake_input_data = pd.DataFrame({
        'INDE 22': [7.5],
        'Idade 22': [11],
        'Matem': [6.0]
    })
    
    class MockPipeline:
        def predict(self, X):
            return np.array([1]) 
        
        def predict_proba(self, X):
            return np.array([[0.1, 0.9]])
            
    fake_model = MockPipeline()
    
    result = make_prediction(fake_model, fake_input_data)

    prediction, probability = make_prediction(fake_model, fake_input_data)
    
    assert result is not None
    assert prediction is not None
    assert probability is not None
    assert prediction[0] == 1 
    assert probability[0][1] == 0.9

@patch('src.models.monitor_drift.Report')
@patch('src.models.monitor_drift.load_raw_data')
def test_monitor_drift_runs(mock_load, mock_report, sample_data):
    
    # 1. Simulamos o carregamento dos dados
    mock_load.return_value = sample_data
    
    # 2. Configuramos o mock do Report e do objeto retornado pelo run
    mock_report_instance = mock_report.return_value
    mock_eval_mock = mock_report_instance.run.return_value
    
    # 3. Executamos a função
    generate_drift_dashboard()
    
    # 4. Verificamos se os métodos vitais foram chamados
    mock_report_instance.run.assert_called_once()
    mock_eval_mock.save_html.assert_called_once_with("drift_dashboard.html")