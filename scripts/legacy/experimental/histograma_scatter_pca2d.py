import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Carga (ajusta la ruta si tu archivo está en otro lugar)
parquet_path = Path("data/snli/pca2d_results/snli_pca2d.parquet")
df = pd.read_parquet(parquet_path)

pc1 = df["pc1"].to_numpy()
pc2 = df["pc2"].to_numpy()
labels = df["label"].to_numpy()          # 0-Entailment, 1-Contradiction, 2-Neutral
color_map = np.array(["C0", "C1", "C2"]) # para scatter opcional

# Get output directory and base name
out_dir = parquet_path.parent
base_name = parquet_path.stem

# ------------------------------------------------------------------

# 1. Histograma de PC1  (muestra las dos "campanas" simétricas)
plt.figure(figsize=(6, 3))
plt.hist(pc1, bins=120, color="steelblue", alpha=0.8)
plt.axvline(0, color="k", lw=1)
plt.title("Distribución de PC1 (se ven dos lóbulos)")
plt.xlabel("PC1")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(out_dir / f"{base_name}_histogram.png", dpi=120)
plt.close()

# 2. Scatter colapsando el signo: |PC1| en X  (cone view 2D)
plt.figure(figsize=(6, 4))
plt.scatter(np.abs(pc1), pc2, c=color_map[labels], s=2, alpha=0.4)
plt.title(r"Visualización del cono — $|PC1|\,$ vs $PC2$")
plt.xlabel(r"|PC1|  (distancia sobre el eje principal)")
plt.ylabel("PC2  (apertura lateral)")
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(out_dir / f"{base_name}_cone.png", dpi=120)
plt.close()

print(f"✅ Gráficos guardados en {out_dir}:")
print(f"   - Histograma: {base_name}_histogram.png")
print(f"   - Vista cono: {base_name}_cone.png")
