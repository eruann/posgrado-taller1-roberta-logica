import numpy as np
import pandas as pd

out_path = "/home/matias/gdrive/Educacion/Posgrado/Taller de Tesis 1/tesis-llm-lpo/data/snli_train_embeddings.parquet"
df = pd.read_parquet(out_path)
V = np.vstack(df["vector"].values)
print("Shape:", V.shape)            # debería ser (n_samples, 2304)
print("Mean per dim:", V.mean(0)[:5])   # ver los primeros 5 valores medios
print("Std per dim:",  V.std(0)[:5])    # ver los primeros 5 valores de desviación
