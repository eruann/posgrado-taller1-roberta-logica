#!/usr/bin/env python
"""
scripts/run_slicing_extended_pipeline.py – Pipeline para Slicing de PCA existente, UMAP y KMeans
==============================================================================================
Utiliza salidas de PCA/ZCA preexistentes, las corta (slice) y luego ejecuta UMAP y KMeans.

1. Identifica archivos PCA/ZCA en un directorio de entrada estructurado por capas.
2. Slicing: Para cada archivo PCA/ZCA, elimina las primeras N componentes (e.g., N=3, N=4).
3. UMAP en salidas de PCA/ZCA originales Y en salidas sliceadas.
4. KMeans en salidas de UMAP.

Cada paso (post-PCA) se registra en MLflow.
"""

import argparse
import subprocess
import shutil
import glob # For finding PCA files
from pathlib import Path
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline para Slicing de PCA existente, UMAP y KMeans")
    # No data_dir for embeddings, as PCA is pre-computed
    parser.add_argument("--pca_input_dir", default="data/snli/experiments/pca",
                      help="Directorio base con salidas de PCA pre-computadas (espera subdirectorios layer_X)")
    parser.add_argument("--output_dir", default="data/snli/experiments_on_sliced_pca",
                      help="Directorio base para salidas de este pipeline (slicing, umap, kmeans)")
    parser.add_argument("--skip_n_values", default="3,4",
                      help="Comma-separated list of n values for skipping first n components (e.g., '3,4'). Set to empty string or '0' to disable slicing.")
    parser.add_argument("--limpiar_directorios_salida",
                      action='store_true',
                      help="Si se establece, elimina el directorio de salida principal de este pipeline antes de ejecutar.")
    # Retain dataset and layer_num for consistency if needed by downstream, though layer is looped
    parser.add_argument("--dataset_name", default="snli", help="Nombre del dataset (e.g., snli)")
    return parser.parse_args()

def run_command(cmd):
    """Ejecuta un comando y retorna su resultado"""
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Error ejecutando comando: {result.stderr}")
        print(f"Comando falló y continuará si es posible: {' '.join(cmd)}")
    return result

def get_vector_dimension(parquet_path):
    """Lee un Parquet y retorna el número de dimensiones (columnas de features) usando GPU."""
    try:
        import cudf
        gdf_sample = cudf.read_parquet(parquet_path)
        if len(gdf_sample) == 0:
            print(f"Advertencia: El archivo Parquet {parquet_path} está vacío.")
            return 0
        # Count non-label columns as feature dimensions
        feature_cols = [col for col in gdf_sample.columns if col != 'label']
        return len(feature_cols)
    except Exception as e:
        print(f"Error leyendo las dimensiones de {parquet_path}: {e}")
        return 0

def main():
    args = parse_args()
    pca_input_base_dir = Path(args.pca_input_dir)
    pipeline_output_base_dir = Path(args.output_dir) # Main output for this script
    dataset_name = args.dataset_name

    skip_n_values_list = []
    if args.skip_n_values:
        try:
            skip_n_values_list = [int(x.strip()) for x in args.skip_n_values.split(',') if x.strip()]
            skip_n_values_list = sorted(list(set(n for n in skip_n_values_list if n > 0)))
        except ValueError:
            print(f"Advertencia: --skip_n_values ('{args.skip_n_values}') no es una lista válida. Slicing desactivado.")
            skip_n_values_list = []
    
    if not skip_n_values_list:
        print("Info: No se especificaron valores válidos para --skip_n_values. Slicing desactivado.")
    else:
        print(f"Valores para skip_first_n que se procesarán: {skip_n_values_list}")

    if args.limpiar_directorios_salida and pipeline_output_base_dir.exists():
        print(f"Opción --limpiar_directorios_salida activada. Eliminando directorio: {pipeline_output_base_dir}")
        shutil.rmtree(pipeline_output_base_dir)
    
    pipeline_output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directories for this pipeline's artifacts
    # Sliced files will go into a subfolder of the original PCA layer dir for clarity, or a dedicated one here.
    # Let's create a dedicated structure within pipeline_output_base_dir for all outputs of this script.
    base_sliced_pca_output_dir = pipeline_output_base_dir / "sliced_pca_files"
    
    # Remove creation of directories for original/unsliced files since we don't process them
    base_umap_on_sliced_output_dirs = {n: pipeline_output_base_dir / f"umap_on_sliced_skip{n}" for n in skip_n_values_list}
    base_kmeans_on_umap_from_sliced_output_dirs = {n: pipeline_output_base_dir / f"kmeans_on_umap_from_sliced_skip{n}" for n in skip_n_values_list}

    dirs_to_create = [base_sliced_pca_output_dir]
    for n_skip_val in skip_n_values_list:
        dirs_to_create.append(base_umap_on_sliced_output_dirs[n_skip_val])
        dirs_to_create.append(base_kmeans_on_umap_from_sliced_output_dirs[n_skip_val])

    for d_create in dirs_to_create:
        d_create.mkdir(parents=True, exist_ok=True)
    
    for layer in range(9, 13):
        current_pca_input_layer_dir = pca_input_base_dir / f"layer_{layer}"
        if not current_pca_input_layer_dir.is_dir():
            print(f"Advertencia: Directorio de entrada PCA {current_pca_input_layer_dir} no encontrado. Saltando capa {layer}.")
            continue
        
        print(f"\nProcesando capa {layer} desde {current_pca_input_layer_dir}...")

        # Layer-specific output dirs for this pipeline
        current_sliced_pca_layer_dir = base_sliced_pca_output_dir / f"layer_{layer}"
        current_sliced_pca_layer_dir.mkdir(parents=True, exist_ok=True)

        current_umap_on_sliced_layer_output_dirs = {}
        current_kmeans_on_umap_from_sliced_layer_output_dirs = {}
        for n_skip in skip_n_values_list:
            dir_umap_sliced = base_umap_on_sliced_output_dirs[n_skip] / f"layer_{layer}"
            dir_kmeans_sliced = base_kmeans_on_umap_from_sliced_output_dirs[n_skip] / f"layer_{layer}"
            dir_umap_sliced.mkdir(parents=True, exist_ok=True)
            dir_kmeans_sliced.mkdir(parents=True, exist_ok=True)
            current_umap_on_sliced_layer_output_dirs[n_skip] = dir_umap_sliced
            current_kmeans_on_umap_from_sliced_layer_output_dirs[n_skip] = dir_kmeans_sliced

        # Find existing PCA/ZCA files
        # Example filename: pca_snli_50_layer9.parquet or zca_snli_1_layer9.parquet
        # We need to infer pca_n_components from the filename if possible, or process all found.
        
        # Let's iterate over *found* pca/zca files to determine original_pca_n_components from filename
        # This is a bit fragile. A more robust way would be if 01_pca.py stored n_components in metadata or a manifest.
        pca_files_pattern = str(current_pca_input_layer_dir / '*.parquet')
        existing_pca_files = [Path(f) for f in glob.glob(pca_files_pattern)]

        if not existing_pca_files:
            print(f"Advertencia: No se encontraron archivos .parquet en {current_pca_input_layer_dir}. Saltando capa {layer}.")
            continue

        for original_pca_file_path in existing_pca_files:
            print(f"  Procesando archivo PCA/ZCA existente: {original_pca_file_path.name}")
            
            # Try to infer original_pca_n_components and reduction_type from filename
            filename_parts = original_pca_file_path.name.split('_')
            if len(filename_parts) < 3:
                print(f"    Advertencia: Nombre de archivo {original_pca_file_path.name} no sigue el formato esperado. Saltando.")
                continue
            
            reduction_type = filename_parts[0] # Should be 'pca' or 'zca'
            if reduction_type not in ["pca", "zca"]:
                print(f"    Advertencia: Tipo de reducción desconocido '{reduction_type}' en {original_pca_file_path.name}. Saltando.")
                continue
            
            try:
                original_pca_n_components_from_filename = int(filename_parts[2])
            except (ValueError, IndexError):
                print(f"    Advertencia: No se pudo extraer n_components del nombre de archivo {original_pca_file_path.name}. Saltando.")
                continue

            tasks_for_umap_kmeans = []

            # Only process sliced versions (do not process original unsliced file)
            if skip_n_values_list:
                for skip_n in skip_n_values_list:
                    dim_to_slice_from = get_vector_dimension(original_pca_file_path)
                    if dim_to_slice_from == 0:
                        print(f"    Advertencia: No se pudo determinar dimensión de {original_pca_file_path} para slicing (skip={skip_n}). Saltando.")
                        continue
                    if dim_to_slice_from <= skip_n:
                        print(f"    Advertencia: Dim original ({dim_to_slice_from}) en {original_pca_file_path.name} es <= skip_n ({skip_n}). Saltando slicing.")
                        continue
                    n_comps_after_slicing = dim_to_slice_from - skip_n
                    if n_comps_after_slicing <= 0: continue

                    base_name_no_ext = original_pca_file_path.stem
                    sliced_filename = f"{base_name_no_ext}_first_{skip_n}_skipped.parquet"
                    path_to_sliced_parquet = current_sliced_pca_layer_dir / sliced_filename

                    print(f"      Generando archivo sliceado: {path_to_sliced_parquet} de {original_pca_file_path.name}")
                    slice_cmd = [
                        "python", "scripts/utilities/slice_parquet_vectors.py",
                        "--input_parquet", str(original_pca_file_path),
                        "--output_parquet", str(path_to_sliced_parquet),
                        "--skip_first_n", str(skip_n)
                    ]
                    slice_run_result = run_command(slice_cmd)

                    if slice_run_result.returncode != 0 or not path_to_sliced_parquet.exists():
                        print(f"Advertencia: Archivo sliceado {path_to_sliced_parquet} no fue creado o slicing falló.")
                        continue
                    
                    tasks_for_umap_kmeans.append({
                        "input_file": path_to_sliced_parquet,
                        "umap_out_dir": current_umap_on_sliced_layer_output_dirs[skip_n],
                        "kmeans_out_dir": current_kmeans_on_umap_from_sliced_layer_output_dirs[skip_n],
                        "n_comps_for_umap_script": n_comps_after_slicing,
                        "umap_exp_name": f"umap_on_sliced_s{skip_n}",
                        "kmeans_exp_name": f"kmeans_on_umap_from_sliced_s{skip_n}",
                        "reduction_tag_for_umap": f"{reduction_type}_skipped{skip_n}",
                        "reduction_tag_for_kmeans": f"{reduction_type}_skipped{skip_n}",
                        "is_sliced": True, "skip_n_val": skip_n,
                        "original_pca_n_components_inferred": original_pca_n_components_from_filename
                    })
            
            # Execute UMAP and KMeans for all scheduled tasks for this original_pca_file_path
            for task_idx, task in enumerate(tasks_for_umap_kmeans):
                input_file_name_for_print = Path(task['input_file']).name
                print(f"    Procesando Tarea UMAP/KMeans {task_idx+1}/{len(tasks_for_umap_kmeans)}: input {input_file_name_for_print}")
                
                input_file_for_umap = task["input_file"]
                actual_dim_for_umap_script = task["n_comps_for_umap_script"]
                inferred_pca_n_comp = task["original_pca_n_components_inferred"]
                
                # Skip UMAP/KMeans if input dimension is less than 2
                if actual_dim_for_umap_script < 2:
                    print(f"    Advertencia: Dimensión de entrada para UMAP ({actual_dim_for_umap_script}) es < 2. Saltando UMAP y KMeans para {input_file_name_for_print}.")
                    continue
                
                metrics_for_umap = ["euclidean", "manhattan"] if task["reduction_tag_for_umap"].startswith("pca") else ["euclidean"]
                
                for n_neighbors in [15, 100]:
                    for metric in metrics_for_umap:
                        umap_suffix = task["reduction_tag_for_umap"].replace('_','-') # e.g. pca or zca or pca-skipped3
                        # Filename construction for UMAP output parquet
                        umap_output_filename = f"umap_{dataset_name}_pca{inferred_pca_n_comp}_layer{layer}_n{n_neighbors}_{metric}_{umap_suffix}.parquet"
                        umap_output_path_for_kmeans = task["umap_out_dir"] / umap_output_filename
                        
                        umap_cmd = [
                            "python", "experiments/02_umap.py",
                            "--pca_path", str(input_file_for_umap), # This is the actual input (original or sliced)
                            "--out_dir", str(task["umap_out_dir"]),
                            "--n_neighbors", str(n_neighbors),
                            "--metric", metric,
                            "--dataset", dataset_name,
                            "--experiment_name", task["umap_exp_name"],
                            "--reduction_type", reduction_type, # Changed from task["reduction_tag_for_umap"] to use original reduction_type (pca or zca)
                            "--layer_num", str(layer),
                            "--input_n_components", str(actual_dim_for_umap_script) # Actual dim of input_file_for_umap
                        ]
                        if task["is_sliced"]:
                             umap_cmd.extend(["--original_pca_n_components_before_slice", str(inferred_pca_n_comp)])
                             umap_cmd.extend(["--skipped_n_components", str(task["skip_n_val"])])

                        umap_run_result = run_command(umap_cmd)

                        if umap_run_result.returncode != 0 or not umap_output_path_for_kmeans.exists():
                            print(f"Advertencia: UMAP falló o archivo UMAP no encontrado {umap_output_path_for_kmeans}, saltando KMeans.")
                            continue
                        
                        kmeans_cmd = [
                            "python", "experiments/03_kmeans.py",
                            "--input_path", str(umap_output_path_for_kmeans),
                            "--out_dir", str(task["kmeans_out_dir"]),
                            "--k", "3",
                            "--dataset", dataset_name,
                            "--experiment_name", task["kmeans_exp_name"],
                            "--reduction_type", "umap", 
                            "--layer_num", str(layer),
                            "--input_n_components", str(actual_dim_for_umap_script), 
                            "--umap_n_neighbors", str(n_neighbors),
                            "--umap_metric", metric,
                            "--umap_source_reduction_type", reduction_type
                        ]
                        if task["is_sliced"]:
                            kmeans_cmd.extend(["--original_pca_n_components_before_slice", str(inferred_pca_n_comp)])
                            kmeans_cmd.extend(["--skipped_n_components", str(task["skip_n_val"])])
                        run_command(kmeans_cmd)
            
            tasks_for_umap_kmeans.clear() # Ensure list is cleared for next original_pca_file_path

    print("\nPipeline completado.")

if __name__ == "__main__":
    main() 