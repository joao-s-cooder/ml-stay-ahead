import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data, build_preprocessing_pipeline
from src.utils.paths import ARTIFACTS_DIR

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.joblib')

def train_model():
    """
    Treino de modelo com tuning de hiperparametros.
    """
    # 1. Carregar dados
    print("Carregando dados...")
    try:
        df = load_raw_data()
    except FileNotFoundError:
        print("Arquivo de dados nao encontrado. Por favor verifique o path 'data/raw/'")
        return

    # 2. Pre processamento
    print("Processando dados...")
    X, y, numeric_cols, categorical_cols = preprocess_data(df)
    
    # 3. Pipeline de modelagem
    # Processamento + Modelo
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # Modelo de base
    rf = RandomForestClassifier(random_state=42)
    
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    # 4. Separacao de dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Tuning de Hypertparametros
    print(" Realizando Tuning de hyperparametro (RandomizedSearchCV)...")
    
    # Define hyperparameter grid
    param_dist = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    }
    
    with mlflow.start_run():
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            clf, 
            param_distributions=param_dist, 
            n_iter=10, 
            cv=3, 
            verbose=1, 
            random_state=42, 
            n_jobs=-1,
            scoring='accuracy' 
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"Best Parameters: {random_search.best_params_}")
        print(f"Best CV Score: {random_search.best_score_:.4f}")
        
        # Log Melhores parametros
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("best_cv_score", random_search.best_score_)
        
        best_model = random_search.best_estimator_
        
        # 6. Evaluate
        print("Indicadores melhor modelo...")
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(classification_report(y_test, y_pred))
        print(f"Test Accuracy: {acc}")
        
        # Log Metrics
        mlflow.log_metric("test_accuracy", acc)
        
        # 7. Save
        print(f"Saving best model to {MODEL_PATH}...")
        joblib.dump(best_model, MODEL_PATH)
        print("Model saved.")
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "random_forest_model")

if __name__ == "__main__":
    train_model()
