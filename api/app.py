from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.utils.paths import ARTIFACTS_DIR

app = FastAPI(title="Passos Mágicos - School Lag Prediction API")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

# Carrega o modelo globalmente quando iniciar o carregamento apos cada request
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. API will not be able to predict.")

class StudentData(BaseModel):
    idade_22: float
    genero: str
    instituicao_ensino: str
    pedra_22: str
    inde_22: float
    iaa: float
    ieg: float
    ips: float
    ida: float
    matem: float
    portug: float
    ingles: float

    # Map to DataFrame columns expected by the model
    def to_dict(self):
        return {
            'Idade 22': [self.idade_22],
            'Gênero': [self.genero],
            'Instituição de ensino': [self.instituicao_ensino],
            'Pedra 22': [self.pedra_22],
            'INDE 22': [self.inde_22],
            'IAA': [self.iaa],
            'IEG': [self.ieg],
            'IPS': [self.ips],
            'IDA': [self.ida],
            'Matem': [self.matem],
            'Portug': [self.portug],
            'Inglês': [self.ingles]
        }

@app.get("/")
def read_root():
    return {"message": "Welcome to Passos Mágicos Lag Prediction API"}

@app.post("/predict")
def predict(student: StudentData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame(student.to_dict())
        
        # Predict
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Expected prediction is 0 (No Risk) or 1 (Risk)
        risk = bool(prediction[0])
        risk_probability = float(probability[0][1]) # Probability of class 1
        
        return {
            "risk_of_lag": risk,
            "risk_probability": risk_probability,
            "message": "High risk of lag" if risk else "Low risk of lag"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
