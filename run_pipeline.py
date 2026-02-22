from src.models.train_model import train_model

if __name__ == "__main__":
    print("Iniciando Pipeline de ML...")
    try:
        train_model()
        print("Pipeline concluído com sucesso.")
    except Exception as e:
        print(f"Pipeline falhou: {e}")
