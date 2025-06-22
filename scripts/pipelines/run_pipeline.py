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
    
    # Directorios base para cada tipo de experimento
    base_pca_dir = output_dir / "pca"
    base_umap_dir = output_dir / "umap"
    base_kmeans_dir = output_dir / "kmeans"
    
    # Limpiar directorios existentes solo si se especifica
    if args.limpiar_directorios_salida:
        print("Opción --limpiar_directorios_salida activada. Limpiando directorios base...")
        for d_base in [base_pca_dir, base_umap_dir, base_kmeans_dir]:
            if d_base.exists():
                print(f"Eliminando directorio existente: {d_base}")
                shutil.rmtree(d_base)
    
    # Asegurar que los directorios base de experimentos existan (crearlos si no)
    for d_base in [base_pca_dir, base_umap_dir, base_kmeans_dir]:
        d_base.mkdir(parents=True, exist_ok=True)
    
    # Procesar cada capa
    for layer in range(9, 13):
        embedding_file = data_dir / f"embeddings_snli_layer_{layer}.parquet"
        if not embedding_file.exists():
            print(f"Advertencia: {embedding_file} no encontrado, saltando...")
            continue
            
        print(f"\nProcesando capa {layer}...")

        # Crear directorios específicos para esta capa dentro de los directorios base
        current_pca_layer_dir = base_pca_dir / f"layer_{layer}"
        current_umap_layer_dir = base_umap_dir / f"layer_{layer}"
        current_kmeans_layer_dir = base_kmeans_dir / f"layer_{layer}"

        for d_layer in [current_pca_layer_dir, current_umap_layer_dir, current_kmeans_layer_dir]:
            d_layer.mkdir(parents=True, exist_ok=True)
        
        # Ejecutar PCA/ZCA para diferentes dimensiones
        for n_components in [1, 5, 50]:
            # El nombre base del archivo que 01_pca.py usará (01_pca.py prefijará pca_ o zca_)
            # Mantenemos _layer{layer} en el nombre del archivo por claridad y consistencia con scripts anteriores
            pca_base_filename_component = f"snli_{n_components}_layer{layer}"
            # El argumento --out para 01_pca.py ahora apunta al directorio específico de la capa
            pca_out_template_for_script = current_pca_layer_dir / f"{pca_base_filename_component}.parquet"
            
            pca_cmd = [
                "python", "experiments/01_pca.py",
                "--source_path", str(embedding_file),
                "--out", str(pca_out_template_for_script),
                "--n_components", str(n_components),
                "--experiment_name", "pca",
                "--dataset", "snli",
                "--layer_num", str(layer)
            ]
            run_command(pca_cmd)
            
            # Ejecutar UMAP en salidas de PCA y ZCA
            for reduction_type in ["pca", "zca"]:
                # El archivo de entrada para UMAP ahora reside en el directorio específico de la capa de PCA
                input_file_for_umap = current_pca_layer_dir / f"{reduction_type}_{pca_base_filename_component}.parquet"
                
                metrics = ["euclidean", "manhattan"] if reduction_type == "pca" else ["euclidean"]
                
                for n_neighbors in [15, 100, 150, 200]:
                    for metric in metrics:
                        # El script 02_umap.py guardará sus salidas en current_umap_layer_dir.
                        # El nombre del archivo generado por 02_umap.py (asumiendo que sigue incluyendo _layer{layer})
                        umap_filename_by_script = f"umap_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{reduction_type}.parquet"
                        # Ruta completa del archivo UMAP que se usará como entrada para KMeans
                        umap_output_path_for_kmeans = current_umap_layer_dir / umap_filename_by_script
                        
                        umap_cmd = [
                            "python", "experiments/02_umap.py",
                            "--pca_path", str(input_file_for_umap),
                            "--out_dir", str(current_umap_layer_dir), # Se pasa el directorio específico de la capa
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
                        # El script 03_kmeans.py guardará sus salidas en current_kmeans_layer_dir
                        # El nombre del archivo generado por 03_kmeans.py
                        kmeans_filename_by_script = f"kmeans_snli_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{reduction_type}_k3.csv" # Asumiendo k=3
                        # Ruta completa del archivo final de KMeans (no se usa como entrada posterior en este script)
                        # kmeans_final_path = current_kmeans_layer_dir / kmeans_filename_by_script

                        kmeans_cmd = [
                            "python", "experiments/03_kmeans.py",
                            "--input_path", str(umap_output_path_for_kmeans), # Usar la ruta completa al archivo UMAP
                            "--out_dir", str(current_kmeans_layer_dir), # Se pasa el directorio específico de la capa
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