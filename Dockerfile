FROM python:3.10-slim

WORKDIR /app

# Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codigo fonte
COPY src/ src/
COPY api/ api/
COPY models_artifacts/ models_artifacts/

# Define PYTHONPATH para incluir o diretorio atual para que os modulos src possam ser encontrados
ENV PYTHONPATH=/app

# Expõe a porta
EXPOSE 8000

# Roda a API
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}
