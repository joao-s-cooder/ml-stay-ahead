import os

# root do projeto 2 niveis acima deste arquivo (src/utils/paths.py -> src/ -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'models_artifacts')
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
