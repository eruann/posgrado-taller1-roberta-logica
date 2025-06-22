import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# Load data
parquet_path = Path("data/snli_embeddings.parquet")
df = pd.read_parquet(parquet_path)          # vectores 2304-d
X   = np.vstack(df.vector.to_numpy()).astype("float32")

# PCA a 3 componentes (GPU o CPU, da igual para ~100 k muestras)
pc = PCA(n_components=3, random_state=42).fit_transform(X)
pc1, pc2, pc3 = pc[:,0], pc[:,1], pc[:,2]
labels = df.label.to_numpy()

# coordenadas "cono 3D": eje = |PC1|, radio = sqrt(PC2² + PC3²)
radio  = np.sqrt(pc2**2 + pc3**2)
altura = np.abs(pc1)

# Get output directory and base name
out_dir = parquet_path.parent
base_name = parquet_path.stem

plt.figure(figsize=(6,4))
plt.scatter(altura, radio,
            c=np.array(["C0","C1","C2"])[labels],
            s=2, alpha=.4)
plt.xlabel("|PC1|  (altura)")
plt.ylabel("√(PC2²+PC3²)  (radio)")
plt.title("Único cono en coordenadas (altura, radio)")
plt.tight_layout()
plt.savefig(out_dir / f"{base_name}_cono3d.png", dpi=120)
plt.close()

print(f"✅ Gráfico guardado en {out_dir}/{base_name}_cono3d.png")
