import pandas as pd
from pathlib import Path

# Verificar el primer archivo
file_path = Path('~/gdrive/Educacion/Posgrado/Taller de Tesis 1/tesis-llm-lpo/data/snli/embeddings_snli_train_part0000.parquet')
try:
    df = pd.read_parquet(file_path)
    print(f"Archivo {file_path} es legible")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error al leer {file_path}: {str(e)}")