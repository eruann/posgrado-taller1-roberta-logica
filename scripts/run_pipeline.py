#!/usr/bin/env python
"""
scripts/run_pipeline.py – Pipeline para ejecutar experimentos de PCA, ZCA, UMAP y KMeans
=====================================================================
Ejecuta un pipeline de experimentos para cada capa de embeddings del 9 al 12:
1. PCA y ZCA con diferentes dimensiones (1, 5, 50)
2. UMAP en salidas de PCA/ZCA:
   - Para PCA: métricas euclidean y L1 (manhattan)
   - Para ZCA: solo métrica euclidean
3. KMeans en salidas de UMAP

Cada paso se registra en MLflow con sus parámetros y artefactos.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline para ejecutar experimentos")
    parser.add_argument("--data_dir", default="data/snli/embeddings",
                      help="Directorio con archivos de embeddings")
    parser.add_argument("--output_dir", default="data/snli/experiments",
                      help="Directorio base para salidas de experimentos")
    parser.add_argument("--limpiar_directorios_salida",
                      action='store_true',  # Default to False if not specified
                      help="Si se establece, elimina los directorios de salida de experimentos (pca, umap, kmeans) antes de ejecutar.")
    return parser.parse_args()

def run_command(cmd):
    """Ejecuta un comando y retorna su salida"""
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error ejecutando comando: {result.stderr}")
        raise Exception(f"Comando falló: {' '.join(cmd)}")
    return result.stdout

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Crear directorios de salida base (si no existen)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pca_dir = output_dir / "pca"
    umap_dir = output_dir / "umap"
    kmeans_dir = output_dir / "kmeans"
    
    # Limpiar directorios existentes solo si se especifica
    if args.limpiar_directorios_salida:
        print("Opción --limpiar_directorios_salida activada. Limpiando directorios...")
        for d in [pca_dir, umap_dir, kmeans_dir]:
            if d.exists():
                print(f"Eliminando directorio existente: {d}")
                shutil.rmtree(d)
    
    # Asegurar que los directorios de experimentos existan (crearlos si no)
    # Esto es importante incluso si no se limpian, para la primera ejecución o si se borraron manualmente.
    for d in [pca_dir, umap_dir, kmeans_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada capa
    for layer in range(9, 13):
        embedding_file = data_dir / f"embeddings_snli_layer_{layer}.parquet"
        if not embedding_file.exists():
            print(f"Advertencia: {embedding_file} no encontrado, saltando...")
            continue
            
        print(f"\nProcesando capa {layer}...")
        
        # Ejecutar PCA/ZCA para diferentes dimensiones
        for n_components in [1, 5, 50]:
            # Ejecutar PCA (que ahora también calcula ZCA)
            pca_cmd = [
                "python", "experiments/01_pca.py",
                "--source_path", str(embedding_file),
                "--out", str(pca_dir / f"snli_{n_components}_layer{layer}.parquet"),
                "--n_components", str(n_components),
                "--experiment_name", "pca",
                "--dataset", "snli",
                "--layer_num", str(layer)
            ]
            run_command(pca_cmd)
            
            # Ejecutar UMAP en salidas de PCA y ZCA
            for reduction_type in ["pca", "zca"]:
                # Obtener el archivo de entrada apropiado
                input_file = pca_dir / f"{reduction_type}_snli_{n_components}_layer{layer}.parquet"
                
                # Definir métricas según el tipo de reducción
                metrics = ["euclidean", "manhattan"] if reduction_type == "pca" else ["euclidean"]
                
                # Ejecutar UMAP para cada métrica
                for n_neighbors in [15, 100]:
                    for metric in metrics:
                        umap_output = umap_dir / f"umap_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{reduction_type}.parquet"
                        umap_cmd = [
                            "python", "experiments/02_umap.py",
                            "--pca_path", str(input_file),
                            "--out_dir", str(umap_dir),
                            "--n_neighbors", str(n_neighbors),
                            "--min_dist", "0.1",
                            "--metric", metric,
                            "--dataset", "snli",
                            "--experiment_name", "umap",
                            "--reduction_type", reduction_type,
                            "--layer_num", str(layer),
                            "--input_n_components", str(n_components)
                        ]
                        run_command(umap_cmd)
                        
                        # Ejecutar KMeans en salida de UMAP
                        kmeans_output = kmeans_dir / f"kmeans_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{reduction_type}_k3.csv"
                        kmeans_cmd = [
                            "python", "experiments/03_kmeans.py",
                            "--input_path", str(umap_output),
                            "--out_dir", str(kmeans_dir),
                            "--k", "3",
                            "--dataset", "snli",
                            "--experiment_name", "kmeans",
                            "--reduction_type", "umap",
                            "--layer_num", str(layer),
                            "--input_n_components", str(n_components),
                            "--umap_n_neighbors", str(n_neighbors),
                            "--umap_metric", metric,
                            "--umap_source_reduction_type", reduction_type
                        ]
                        run_command(kmeans_cmd)

if __name__ == "__main__":
    main()