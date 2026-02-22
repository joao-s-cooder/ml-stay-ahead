import joblib
import os
import pandas as pd
from src.utils.paths import ARTIFACTS_DIR

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def make_prediction(model, data: pd.DataFrame):
    """
    Faz uma previsão para os dados de entrada.
    Os dados de entrada devem ser um DataFrame com as mesmas colunas dos dados de treinamento.
    """
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    return prediction, probability

if __name__ == "__main__":
    try:
        model = load_model()
        print("Model loaded.")
        pass
    except Exception as e:
        print(f"Error: {e}")
