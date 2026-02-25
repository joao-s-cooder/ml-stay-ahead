import pandas as pd
import os
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data

def generate_drift_dashboard():
    print("Carregando dados para análise de drift...")
    df = load_raw_data()
    
    # Processamos os dados para ter o formato final usado pelo modelo
    X, y, _, _ = preprocess_data(df)
    
    # Juntamos X e y para o Evidently poder analisar a base completa
    X['Defas'] = y
    
    # Dividimos os dados em referência (treino) e produção (teste) para simular o cenário real
    reference_data, current_data = train_test_split(X, test_size=0.3, random_state=42)
    
    print("Gerando painel de Drift (Evidently AI)...")
    
    # Inicializamos o relatório com os presets corretos
    drift_report = Report(metrics=[
        DataSummaryPreset(), # Gera um resumo estatístico das colunas
        DataDriftPreset()    # Calcula o desvio (drift) entre referência e produção
    ])
    
    # Executa a análise
    my_eval = drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Salva o dashboard interativo em HTML
    output_path = "drift_dashboard.html"
    my_eval.save_html(output_path)
    
    print(f"✅ Painel de monitoramento gerado com sucesso: {output_path}")

if __name__ == "__main__":
    generate_drift_dashboard()