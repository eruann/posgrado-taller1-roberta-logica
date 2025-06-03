#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import os

def merge_parquet_files(input_dir: str, output_file: str, batch_size: int = 10):
    """
    Une archivos parquet en grupos para evitar problemas de memoria
    
    Args:
        input_dir: Directorio con los archivos parciales
        output_file: Archivo de salida
        batch_size: Número de archivos a procesar por vez
    """
    print(f"Directorio actual: {os.getcwd()}")
    print(f"Verificando si existe el directorio: {input_dir}")
    print(f"El directorio existe: {os.path.exists(input_dir)}")
    
    # Obtener lista de archivos y ordenarlos
    pattern = f"{input_dir}/embeddings_snli_train_part*.parquet"
    print(f"Buscando archivos con el patrón: {pattern}")
    files = sorted(glob.glob(pattern))
    print(f"Encontrados {len(files)} archivos para unir")
    
    if not files:
        print("Listando archivos en el directorio:")
        print(os.listdir(input_dir))
        raise Exception(f"No se encontraron archivos en {input_dir}")
    
    # Procesar en grupos
    all_dfs = []
    for i in tqdm(range(0, len(files), batch_size), desc="Procesando grupos"):
        batch_files = files[i:i + batch_size]
        print(f"\nProcesando archivos: {batch_files}")
        try:
            # Leer y unir el grupo actual
            print("Intentando leer archivos...")
            batch_dfs = []
            for f in batch_files:
                print(f"Leyendo {f}...")
                try:
                    df = pd.read_parquet(f)
                    print(f"Archivo {f} leído correctamente. Shape: {df.shape}")
                    batch_dfs.append(df)
                except Exception as e:
                    print(f"Error leyendo {f}: {str(e)}")
                    raise
            
            print("Concatenando DataFrames...")
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            all_dfs.append(batch_df)
            print(f"Grupo {i//batch_size + 1}: {len(batch_df)} filas")
        except Exception as e:
            print(f"Error procesando grupo {i//batch_size + 1}: {str(e)}")
            continue
    
    if not all_dfs:
        raise Exception("No se pudo procesar ningún grupo de archivos")
    
    # Unir todos los grupos
    print("Uniendo todos los grupos...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Guardar resultado
    print(f"Guardando {len(final_df)} filas en {output_file}")
    final_df.to_parquet(output_file)
    print("✅ Proceso completado")

if __name__ == "__main__":
    # Usar rutas desde el directorio raíz
    root_dir = Path("~/gdrive/Educacion/Posgrado/Taller de Tesis 1/tesis-llm-lpo").expanduser()
    input_dir = str(root_dir / "data/snli")
    output_file = str(root_dir / "data/snli/embeddings_snli_train.parquet")
    
    print(f"Buscando archivos en: {input_dir}")
    merge_parquet_files(input_dir, output_file) 