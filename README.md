# Passos Mágicos - School Lag Prediction (MLOps)

Este projeto implementa o ciclo completo de **Machine Learning Operations (MLOps)** para predizer o risco de defasagem escolar de alunos da *Associação Passos Mágicos*.

## 🎯 Objetivo
Avaliar o risco de estudantes não acompanharem a fase ideal, permitindo intervenções assertivas. O modelo utiliza dados acadêmicos, de engajamento e psicossociais.

## 📊 Avaliação do Modelo e Confiabilidade para Produção
A principal métrica escolhida para avaliação e otimização do modelo foi o Recall da classe alvo (Alunos em Risco), complementado pelo **F1-Score**.

**Resultados obtidos no conjunto de testes:**
- Recall (Classe 1): 95.93%
- Precision (Classe 1): 92.00%
- F1-Score: 94.02%
- Acurácia Global: 91.28%

**Por que este modelo é confiável para produção?**
No contexto da Associação Passos Mágicos, o maior risco para o negócio (e para a sociedade) é o **Falso Negativo** ou seja, o modelo classificar um aluno como "Fora de Risco" quando ele, na verdade, precisa de intervenção pedagógica ou psicológica.

Com um **Recall de quase 96%**, o modelo demonstrou uma altíssima sensibilidade, garantindo que a imensa maioria dos estudantes em situação de vulnerabilidade educacional seja identificada preventivamente. Além disso, a **Precisão de 92%** assegura que os recursos limitados da ONG (tempo de psicólogos e pedagogos) sejam direcionados de forma assertiva, com baixíssimo índice de alarmes falsos. O equilíbrio refletido no F1-Score (94%) comprova a robustez e a maturidade do modelo para operar no mundo real.

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

### 5. Exemplos de Chamadas à API

A API foi desenvolvida em FastAPI e expõe um endpoint principal para receber os dados do aluno e retornar a probabilidade de defasagem escolar.

**Endpoint:** `POST /predict`
**Content-Type:** `application/json`

### Input Esperado (Payload)
O modelo espera receber um JSON contendo as features socioeconômicas e acadêmicas do aluno.
```json
{
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
```

### Output Gerado (Resposta)
A API processa os dados pelo pipeline de Machine Learning e retorna a classificação de risco (0 para "Sem Risco" e 1 para "Em Risco") junto com a probabilidade (confiança do modelo).
```json
{
  "risk_of_lag": 1,
  "risk_probability": 0.89
}
```

## Exemplo via cURL (Terminal)
Você pode testar a API localmente executando o comando abaixo no seu terminal:
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'
```

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

## 📊 Monitoramento Contínuo e Data Drift

Para garantir a confiabilidade do modelo em produção e mitigar a degradação da performance ao longo do tempo, o projeto conta com uma camada de monitoramento:

* **Logs de Experimentos e Treinamento:** Utilizado o **MLflow** para registrar todos os hiperparâmetros, artefatos do modelo e métricas de avaliação (Recall, F1-Score, Acurácia) a cada execução do pipeline de treino.
* **Painel de Acompanhamento de Drift:** Implementado a geração de relatórios com o **Evidently AI**. O script `src/models/monitor_drift.py` compara a distribuição dos dados de referência (treinamento) com os dados atuais (produção/inferência) e gera um dashboard interativo (`drift_dashboard.html`).
    * **Data Drift:** Avalia se as características socioeconômicas e acadêmicas dos alunos mudaram significativamente.
    * **Target Drift:** Monitora mudanças na proporção de alunos em risco de defasagem, gerando alertas visuais caso as premissas de negócio sofram alterações sistêmicas.
