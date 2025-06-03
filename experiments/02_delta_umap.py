#!/usr/bin/env python
"""
snli_delta_umap.py  –  Visualise entailment / contradiction / neutral lobes
==========================================================================

Input  : Parquet with columns  'vector' (2304-floats list)  and 'label' (0/1/2)
Output : • PNG scatter  |  • Parquet with UMAP2D  |  • purity score on stdout
Steps  :
    1.  build Δ = e_p – e_h   (768-D)
    2.  length-normalise each Δ
    3.  remove first principal component (CCR)
    4.  PCA → 50 dims
    5.  UMAP (cosine, n_neighbors=100) → 2D
    6.  plot & save
    7.  run K-Means(k=3) in PCA-50 space and report best purity
"""

# ------------------------------------------------------------------- stdlib
from pathlib import Path
import itertools
import argparse
# ---------------------------------------------------------------- 3rd-party
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster      import KMeans
from sklearn.metrics      import accuracy_score
import umap
import mlflow

# Force MLflow to use local mlruns directory
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

# ---------------------------- CLI -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--inp",  required=True, help="Parquet with vector,label")
parser.add_argument("--out_dir", default=None, help="Folder for outputs (default: same as inp)")
parser.add_argument("--experiment_name", default="delta-umap-snli",
                    help="Name of the MLflow experiment")
parser.add_argument("--dataset", default="snli",
                    help="Name of the dataset (e.g., snli, mnli, etc)")
parser.add_argument("--layer_num", type=int, default=12, help="Layer number to use (default: 12)")
args = parser.parse_args()

inp_path  = Path(args.inp)
out_dir   = Path(args.out_dir or inp_path.parent)
out_dir.mkdir(parents=True, exist_ok=True)

# Set up MLflow
mlflow.set_experiment(args.experiment_name)
run_name = f"delta_umap_{args.dataset}"
with mlflow.start_run(run_name=run_name) as run:
    # Log parameters
    for k, v in vars(args).items():
        mlflow.log_param(k, v)
    # Log layer_num as parameter (from args)
    mlflow.log_param("layer_num", args.layer_num)
    
    # Set dataset and experiment as tags
    mlflow.set_tag("dataset", args.dataset)
    mlflow.set_tag("experiment_name", args.experiment_name)
    mlflow.set_tag("model_type", "delta_umap")
    mlflow.set_tag("reduction_type", "umap")

    # ---------------------------- 1 · load ------------------------------------
    print("Reading parquet …")
    df = pd.read_parquet(inp_path)
    X  = np.vstack(df["vector"].to_numpy()).astype(np.float32)        # (N,2304)
    y  = df["label"].to_numpy()

    # Log dataset statistics
    n_samples, n_features = X.shape
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("n_features", n_features)

    # split premisa / conclusión / Δ
    e_p  = X[:,  :768]
    e_h  = X[:, 768:1536]
    Δ    = e_p - e_h                                                  # (N,768)

    # ---------------------------- 2 · L2 normalise ----------------------------
    Δ /= np.linalg.norm(Δ, axis=1, keepdims=True) + 1e-9

    # ---------------------------- 3 · CCR  (remove PC-1) ----------------------
    pc1_vec   = PCA(n_components=1, random_state=0).fit(Δ).components_[0]  # (768,)
    proj      = Δ @ pc1_vec
    Δ_ccr     = Δ - np.outer(proj, pc1_vec)

    # ---------------------------- 4 · PCA-50 ----------------------------------
    print("PCA to 50 dims …")
    X50 = PCA(n_components=50, random_state=0).fit_transform(Δ_ccr)

    # ---------------------------- 5 · UMAP 2-D --------------------------------
    print("UMAP 2-D …")
    u = umap.UMAP(
            n_neighbors = 100,
            n_components= 2,
            min_dist   = 0.15,
            metric     = "cosine",
            random_state=0
    ).fit_transform(X50)                                              # (N,2)

    # ---------------------------- 6 · Plot ------------------------------------
    colour = np.array(["C0","C1","C2"])[y]                            # 0/1/2
    plt.figure(figsize=(6,4))
    plt.scatter(u[:,0], u[:,1], c=colour, s=3, alpha=.35)
    plt.title("UMAP 2-D on Δ (CCR + PCA-50)")
    plt.xticks([]); plt.yticks([])
    
    # Generate base filename from input path
    base_name = inp_path.stem.replace("_embeddings", "")
    out_base = out_dir / f"{base_name}_delta_umap"
    
    # Save plot
    png_path = out_base.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(png_path), artifact_path="plots", copy=False)

    # save parquet with coordinates
    parquet_path = out_base.with_suffix(".parquet")
    pd.DataFrame({"umap1": u[:,0], "umap2": u[:,1], "label": y})\
      .to_parquet(parquet_path)
    mlflow.log_artifact(str(parquet_path), artifact_path="data", copy=False)

    # ---------------------------- 7 · K-Means purity --------------------------
    clusters = KMeans(n_clusters=3, random_state=0).fit_predict(X50)

    best = 0
    for perm in itertools.permutations([0,1,2]):
        best = max(best, accuracy_score(y, [perm[c] for c in clusters]))

    # Log metrics
    mlflow.log_metric("kmeans_purity", best)

    print(f"Best K-Means purity (k=3 in PCA-50 space):  {best:.3f}")
    print(f"✅ Archivos guardados en {out_dir}:")
    print(f"   - UMAP plot: {png_path}")
    print(f"   - UMAP data: {parquet_path}")
