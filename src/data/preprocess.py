import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Preprocessa os dados brutos.
    
    1. Trata valores ausentes.
    2. Cria variável alvo.
    3. Divide features e target.
    """
    
    # Target: 'Defas'
    # Verifica se 'Defas' existe
    if 'Defas' not in df.columns:
         # Tenta encontrar uma coluna similar se a correspondência exata falhar, ou gera erro
         # Baseado na inspeção, 'Defas' estava presente na saída do script personalizado.
         raise ValueError("Coluna 'Defas' nao encontrada no dataset.")

    # Drop rows quando target é missing
    df = df.dropna(subset=['Defas'])
    
    # Define Target
    # Queremos prever o RISCO de defasagem. 
    # Se Defas < 0, implica que o aluno está atrasado. 
    # Criei um target binário: 1 se Defas < 0 (Risco), 0 caso contrário.
    # A probabilidade da classe 1 (Defasagem) é um "Risk Score".
    
    y = (df['Defas'] < 0).astype(int)
    
    # Select Features
    
    # Features potenciais baseadas na análise exploratória:
    # 'Idade 22', 'Gênero', 'Instituição de ensino', 'Pedra 22', 'INDE 22', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês'
    
    feature_cols = [
        'Idade 22', 'Gênero', 'Instituição de ensino', 
        'Pedra 22', 'INDE 22', 
        'IAA', 'IEG', 'IPS', 'IDA', 
        'Matem', 'Portug', 'Inglês'
    ]
    
    # Filtra apenas colunas existentes
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    
    # Cleaning / Type conversion
    # Algumas colunas numéricas podem ser lidas como strings se tiverem lixo.
    numeric_features = ['Idade 22', 'INDE 22', 'IAA', 'IEG', 'IPS', 'IDA', 'Matem', 'Portug', 'Inglês']
    numeric_features = [c for c in numeric_features if c in X.columns]
    
    categorical_features = ['Gênero', 'Instituição de ensino', 'Pedra 22']
    categorical_features = [c for c in categorical_features if c in X.columns]
    
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    return X, y, numeric_features, categorical_features

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Cria um ColumnTransformer do Scikit-Learn para pré-processamento.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor
