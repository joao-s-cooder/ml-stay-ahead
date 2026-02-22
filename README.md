# Passos Mágicos - School Lag Prediction (MLOps)

Este projeto implementa o ciclo completo de **Machine Learning Operations (MLOps)** para predizer o risco de defasagem escolar de alunos da *Associação Passos Mágicos*.

## 🎯 Objetivo
Avaliar o risco de estudantes não acompanharem a fase ideal, permitindo intervenções assertivas. O modelo utiliza dados acadêmicos, de engajamento e psicossociais.

## 🗂️ Estrutura do Projeto
- `src/`: Código fonte para processamento de dados (`src/data`) e treinamento/predição de modelos (`src/models`).
- `api/`: Aplicação web usando **FastAPI**.
- `docker/`: Configurações de containerização.
- `tests/`: Testes automatizados (pytest).
- `.github/workflows/`: Pipeline de CI/CD (GitHub Actions).

## 🚀 Como Executar Localmente

### 1. Pré-Requisitos e Ambiente Virtual
É altamente recomendada a utilização de um ambiente virtual para isolar as dependências do projeto.
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pipeline de Treinamento e Otimização
O script de pipeline encadeia carregamento, pré-processamento, tuning de hiperparâmetros (RandomizedSearchCV) e salva o melhor modelo.
```bash
python3 run_pipeline.py
```
> O modelo resultante será salvo em `models_artifacts/model.joblib`.

### 3. Monitoramento de Experimentos (MLflow)
O projeto integra o **MLflow** para rastreabilidade de parâmetros (n_estimators, max_depth, etc.) e métricas (Acurácia, Precisão, F1-Score).
Para visualizar o dashboard:
```bash
mlflow ui
```
*Acesse em: `http://127.0.0.1:5000`*

### 4. Iniciando a API
Suba o servidor FastAPI:
```bash
uvicorn api.app:app --reload
```
Acesse a documentação interativa (Swagger) em: `http://127.0.0.1:8000/docs`

## 🐳 Como Executar com Docker
Você pode encapsular a aplicação completa num container estruturado.

1. **Build da Imagem**:
   ```bash
   docker build -t mlstayahead -f docker/Dockerfile .
   ```
2. **Executar o Container**:
   ```bash
   docker run -p 8000:8000 mlstayahead
   ```

## ☁️ Deploy para GCP (Google Cloud Run)
A aplicação está preparada para o Cloud Run. Com a [CLI gcloud instalada e configurada](https://cloud.google.com/sdk/docs/install):

1. **Autenticação e Build**:
   ```bash
   gcloud auth login
   gcloud builds submit --tag gcr.io/<SEU_PROJECT_ID>/mlstayahead-api
   ```
2. **Deploy Automático**:
   ```bash
   gcloud run deploy mlstayahead-api \
     --image gcr.io/<SEU_PROJECT_ID>/mlstayahead-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## 🧪 CI/CD e Qualidade
Toda contribuição ao ramo `main` passará automaticamente pela nossa pipeline de **Integração Contínua (GitHub Actions)**, que:
- Configura o Python
- Instala as dependências
- Treina o modelo via DVC/Scripts
- Valida o código com nossos testes em `tests/`
Para rodar localmente: `pytest tests/`
