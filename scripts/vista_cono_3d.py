import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# --- Try GPU PCA (cuML) and fall back to sklearn on CPU ---
try:
    import cupy as cp
    from cuml.decomposition import PCA as GPU_PCA
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.decomposition import PCA as CPU_PCA
    GPU_AVAILABLE = False
    print("cuML not found – falling back to sklearn PCA on CPU")

# -------------------------------------------------------------------
# 1. Load embeddings parquet  (vectors: 2304‑d  |  labels: 0/1/2)
# -------------------------------------------------------------------
parquet_path = Path("data/snli/embeddings_snli_train.parquet")  # adjust if needed
out_dir      = Path("data/snli")                     # save figures here
out_dir.mkdir(parents=True, exist_ok=True)

df      = pd.read_parquet(parquet_path)
X       = np.vstack(df.vector.to_numpy()).astype("float32")  # (N, 2304)
labels  = df.label.to_numpy()

# -------------------------------------------------------------------
# 2. Fit PCA to 3 components (GPU if available, else CPU)
# -------------------------------------------------------------------
if GPU_AVAILABLE:
    X_gpu = cp.asarray(X)
    pca   = GPU_PCA(n_components=3, random_state=42)
    pc    = cp.asnumpy(pca.fit_transform(X_gpu))  # back to NumPy
    del X_gpu; cp.get_default_memory_pool().free_all_blocks()
    backend = "cuML/GPU"
else:
    pca = CPU_PCA(n_components=3, random_state=42)
    pc  = pca.fit_transform(X)
    backend = "sklearn/CPU"

pc1, pc2, pc3 = pc[:, 0], pc[:, 1], pc[:, 2]

# -------------------------------------------------------------------
# 3. Convert to cone coordinates:  altura = |PC1|   ·   radio = √(PC2²+PC3²)
# -------------------------------------------------------------------
altura = np.abs(pc1)
radio  = np.sqrt(pc2 ** 2 + pc3 ** 2)

# -------------------------------------------------------------------
# 4. Plot single‑cone view |PC1| vs radius  (colored by label) & save
# -------------------------------------------------------------------
color_map = np.array(["C0", "C1", "C2"])  # entail, contr, neutral
plt.figure(figsize=(6, 4))
plt.scatter(altura, radio, c=color_map[labels], s=2, alpha=0.4)
plt.xlabel("|PC1|  (altura)")
plt.ylabel("√(PC2² + PC3²)  (radio)")
plt.title(f"Cone view – PCA3 using {backend}")
plt.tight_layout()

out_png = out_dir / "cone_view_pca3.png"
plt.savefig(out_png, dpi=120)
plt.close()
print(f"✅ Figura guardada en {out_png}")