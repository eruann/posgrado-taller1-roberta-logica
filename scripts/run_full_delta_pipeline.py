#!/usr/bin/env python
"""
scripts/run_full_delta_pipeline.py – Pipeline completo para experimentos con vectores de diferencia
==================================================================================================
Ejecuta un pipeline completo sobre los vectores de diferencia ('delta') PRE-EXTRAÍDOS:
1. Ejecuta el pipeline estándar (PCA -> UMAP -> KMeans) sobre los vectores de diferencia.
2. Ejecuta el pipeline de slicing sobre las salidas PCA del paso anterior.

Este script asume que los datos de vectores de diferencia ya han sido generados por
`scripts/prepare_delta_data.py`.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
import glob
import mlflow

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Pipeline completo para experimentos con vectores de diferencia pre-extraídos.")
    parser.add_argument("--delta_data_dir", default="data/snli/difference_embeddings",
                      help="Directorio con archivos de embeddings de diferencia pre-extraídos.")
    parser.add_argument("--output_dir", default="data/snli/experiments_delta",
                      help="Directorio base para todas las salidas de este pipeline de análisis.")
    parser.add_argument("--skip_n_values", default="3,4",
                      help="Valores de N para el slicing (e.g., '3,4').")
    parser.add_argument("--limpiar_directorios_salida", action='store_true',
                      help="Si se establece, elimina el directorio de salida del ANÁLISIS antes de ejecutar.")
    parser.add_argument("--dataset_name", default="snli_delta", help="Nombre del dataset para logging (e.g., snli_delta)")
    return parser.parse_args()

def run_command(cmd):
    """Ejecuta un comando y maneja errores."""
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Error ejecutando comando: {result.stderr}")
        # Continue if possible, some steps might be non-critical
    return result

def get_vector_dimension(parquet_path):
    """Lee un Parquet y retorna el número de dimensiones (columnas de features) usando GPU."""
    try:
        import cudf
        gdf_sample = cudf.read_parquet(parquet_path)
        if len(gdf_sample) == 0: return 0
        # Count non-label columns as feature dimensions
        feature_cols = [col for col in gdf_sample.columns if col != 'label']
        return len(feature_cols)
    except Exception as e:
        print(f"Error leyendo las dimensiones de {parquet_path}: {e}")
        return 0

def main():
    """Función principal del pipeline."""
    args = parse_args()
    base_data_dir = Path(args.delta_data_dir)
    base_output_dir = Path(args.output_dir)
    dataset_name_delta = args.dataset_name

    if args.limpiar_directorios_salida and base_output_dir.exists():
        print(f"Limpiando directorio de salida de análisis: {base_output_dir}")
        shutil.rmtree(base_output_dir)

    # Definir y crear subdirectorios de análisis
    regular_pipeline_dir = base_output_dir / "regular_pipeline_on_delta"
    slicing_pipeline_dir = base_output_dir / "slicing_pipeline_on_delta"
    
    # --- PASO 1: Ejecutar pipeline regular sobre vectores de diferencia ---
    print("\n--- PASO 1: Ejecutando pipeline regular sobre vectores de diferencia ---")
    base_pca_dir = regular_pipeline_dir / "pca"
    base_umap_dir = regular_pipeline_dir / "umap"
    base_kmeans_dir = regular_pipeline_dir / "kmeans"
    for d in [base_pca_dir, base_umap_dir, base_kmeans_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for layer in range(9, 13):
        embedding_file = base_data_dir / f"embeddings_{dataset_name_delta}_layer_{layer}.parquet"
        if not embedding_file.exists():
            print(f"Advertencia: {embedding_file} no encontrado, saltando capa {layer} en pipeline regular.")
            continue
            
        print(f"\nProcesando capa {layer} (Pipeline Regular)...")
        current_pca_layer_dir = base_pca_dir / f"layer_{layer}"
        current_umap_layer_dir = base_umap_dir / f"layer_{layer}"
        current_kmeans_layer_dir = base_kmeans_dir / f"layer_{layer}"
        for d_layer in [current_pca_layer_dir, current_umap_layer_dir, current_kmeans_layer_dir]:
            d_layer.mkdir(parents=True, exist_ok=True)
        
        for n_components in [1, 5, 50]:
            pca_base_filename = f"{dataset_name_delta}_{n_components}_layer{layer}.parquet"
            pca_out_path = current_pca_layer_dir / pca_base_filename
            
            pca_cmd = ["python", "experiments/01_pca.py", "--source_path", str(embedding_file), "--out", str(pca_out_path), "--n_components", str(n_components), "--experiment_name", f"pca_{dataset_name_delta}", "--dataset", dataset_name_delta, "--layer_num", str(layer)]
            run_command(pca_cmd)
            
            for reduction_type in ["pca", "zca"]:
                input_file_for_umap = current_pca_layer_dir / f"{reduction_type}_{pca_base_filename}"
                if not input_file_for_umap.exists(): continue
                
                metrics = ["euclidean", "manhattan"] if reduction_type == "pca" else ["euclidean"]
                for n_neighbors in [15, 100]:
                    for metric in metrics:
                        # Predict UMAP output path
                        umap_filename = f"umap_{dataset_name_delta}_{n_components}_layer{layer}_n{n_neighbors}_{metric}_{reduction_type}.parquet"
                        umap_output_path = current_umap_layer_dir / umap_filename
                        
                        umap_cmd = ["python", "experiments/02_umap.py", "--pca_path", str(input_file_for_umap), "--out_dir", str(current_umap_layer_dir), "--n_neighbors", str(n_neighbors), "--metric", metric, "--dataset", dataset_name_delta, "--experiment_name", f"umap_{dataset_name_delta}", "--reduction_type", reduction_type, "--layer_num", str(layer), "--input_n_components", str(n_components)]
                        run_command(umap_cmd)
                        
                        if not umap_output_path.exists(): continue

                        kmeans_cmd = ["python", "experiments/03_kmeans.py", "--input_path", str(umap_output_path), "--out_dir", str(current_kmeans_layer_dir), "--k", "3", "--dataset", dataset_name_delta, "--experiment_name", f"kmeans_{dataset_name_delta}", "--reduction_type", "umap", "--layer_num", str(layer), "--input_n_components", str(n_components), "--umap_n_neighbors", str(n_neighbors), "--umap_metric", metric, "--umap_source_reduction_type", reduction_type]
                        run_command(kmeans_cmd)

    # --- PASO 2: Ejecutando pipeline de slicing sobre resultados de PCA ---
    print("\n--- PASO 2: Ejecutando pipeline de slicing sobre PCA de vectores de diferencia ---")
    skip_n_values_list = [int(x.strip()) for x in args.skip_n_values.split(',') if x.strip() and int(x.strip()) > 0]
    if not skip_n_values_list:
        print("Slicing desactivado.")
    else:
        slicing_pipeline_dir.mkdir(parents=True, exist_ok=True)
        base_sliced_pca_output_dir = slicing_pipeline_dir / "sliced_pca_files"
        
        for layer in range(9, 13):
            current_pca_input_layer_dir = base_pca_dir / f"layer_{layer}"
            if not current_pca_input_layer_dir.is_dir(): continue

            print(f"\nProcesando capa {layer} (Pipeline de Slicing)...")
            
            pca_files_pattern = str(current_pca_input_layer_dir / '*.parquet')
            for original_pca_file_path in [Path(f) for f in glob.glob(pca_files_pattern)]:
                filename_parts = original_pca_file_path.name.replace('.parquet', '').split('_')
                reduction_type = filename_parts[0]
                
                original_pca_n_components = None
                for part in filename_parts:
                    if part.isdigit():
                        original_pca_n_components = int(part)
                        break
                if original_pca_n_components is None: continue

                for skip_n in skip_n_values_list:
                    dim_original = get_vector_dimension(original_pca_file_path)
                    if dim_original <= skip_n: continue
                    
                    n_comps_after_slicing = dim_original - skip_n
                    
                    # Define directories for sliced outputs
                    current_sliced_pca_layer_dir = base_sliced_pca_output_dir / f"layer_{layer}"
                    umap_on_sliced_dir = slicing_pipeline_dir / f"umap_on_sliced_skip{skip_n}/layer_{layer}"
                    kmeans_on_sliced_dir = slicing_pipeline_dir / f"kmeans_on_umap_from_sliced_skip{skip_n}/layer_{layer}"
                    for d in [current_sliced_pca_layer_dir, umap_on_sliced_dir, kmeans_on_sliced_dir]:
                        d.mkdir(parents=True, exist_ok=True)

                    sliced_filename = f"{original_pca_file_path.stem}_first_{skip_n}_skipped.parquet"
                    path_to_sliced_parquet = current_sliced_pca_layer_dir / sliced_filename

                    slice_cmd = ["python", "scripts/utilities/slice_parquet_vectors.py", "--input_parquet", str(original_pca_file_path), "--output_parquet", str(path_to_sliced_parquet), "--skip_first_n", str(skip_n)]
                    run_command(slice_cmd)

                    if not path_to_sliced_parquet.exists(): continue
                    
                    metrics_for_umap = ["euclidean", "manhattan"] if reduction_type == "pca" else ["euclidean"]
                    for n_neighbors in [15, 100]:
                        for metric in metrics_for_umap:
                            umap_cmd = ["python", "experiments/02_umap.py", "--pca_path", str(path_to_sliced_parquet), "--out_dir", str(umap_on_sliced_dir), "--n_neighbors", str(n_neighbors), "--metric", metric, "--dataset", dataset_name_delta, "--experiment_name", f"umap_on_sliced_s{skip_n}_{dataset_name_delta}", "--reduction_type", reduction_type, "--layer_num", str(layer), "--input_n_components", str(n_comps_after_slicing), "--original_pca_n_components_before_slice", str(original_pca_n_components), "--skipped_n_components", str(skip_n)]
                            run_command(umap_cmd)
                            
                            # Find the created UMAP file by constructing a specific glob pattern.
                            # The UMAP script (02_umap.py) creates filenames like:
                            # umap_{dataset}_{id_part}_layer{...}_n{...}_{metric}_{reduction_type}-skipped{...}.parquet
                            # where id_part is like 'pca50' and reduction_type-skipped is 'pca-skipped3'
                            
                            id_component_part = f"pca{original_pca_n_components}"
                            reduction_suffix = f"{reduction_type}-skipped{skip_n}"
                            
                            umap_glob_pattern = f"umap_*_{id_component_part}_layer{layer}_n{n_neighbors}_{metric}_{reduction_suffix}.parquet"
                            found_files = list(umap_on_sliced_dir.glob(umap_glob_pattern))

                            if not found_files:
                                print(f"Advertencia: No se encontró el archivo de salida de UMAP con el patrón '{umap_glob_pattern}' en {umap_on_sliced_dir}, saltando KMeans.")
                                continue
                            
                            # In case multiple matches are found (e.g. from different original_pca_n_components),
                            # we warn the user and proceed with the first one found.
                            if len(found_files) > 1:
                                print(f"Advertencia: Múltiples archivos UMAP encontrados para el patrón '{umap_glob_pattern}'. Usando el primero: {found_files[0].name}")

                            umap_output_path_for_kmeans = found_files[0]
                            
                            kmeans_cmd = ["python", "experiments/03_kmeans.py", "--input_path", str(umap_output_path_for_kmeans), "--out_dir", str(kmeans_on_sliced_dir), "--k", "3", "--dataset", dataset_name_delta, "--experiment_name", f"kmeans_on_umap_from_sliced_s{skip_n}_{dataset_name_delta}", "--reduction_type", "umap", "--layer_num", str(layer), "--input_n_components", str(n_comps_after_slicing), "--umap_n_neighbors", str(n_neighbors), "--umap_metric", metric, "--umap_source_reduction_type", reduction_type, "--original_pca_n_components_before_slice", str(original_pca_n_components), "--skipped_n_components", str(skip_n)]
                            run_command(kmeans_cmd)
    
    print("\nPipeline de vectores de diferencia completado.")

if __name__ == "__main__":
    main() 