import pandas as pd

file = "data/snli_train_embeddings_part0003.parquet"   # ← cambia al que quieras
df = pd.read_parquet(file)

vec = df.loc[0, "vector"]        # primer registro
print("Etiqueta:", df.loc[0, "label"])
print("Longitud del vector:", len(vec))
print("Primeros 10 valores:", vec[:10])
