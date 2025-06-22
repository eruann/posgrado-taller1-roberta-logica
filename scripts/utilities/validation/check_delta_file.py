#!/usr/bin/env python
"""
scripts/check_delta_file.py – Verifica el contenido de un archivo Parquet de "delta"
=====================================================================================
Carga un archivo parquet y muestra información básica para verificar su contenido,
incluyendo:
- Columnas presentes
- Número de filas y columnas
- Dimensión del primer vector
- Primeras 5 filas del DataFrame
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    """Función principal para verificar el archivo."""
    parser = argparse.ArgumentParser(description="Check the contents of a delta vector parquet file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the .parquet file to check.")
    args = parser.parse_args()
    
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"Error: El archivo no existe en la ruta especificada: {file_path}")
        return

    print(f"--- Verificando archivo: {file_path.name} ---")

    try:
        df = pd.read_parquet(file_path)

        print(f"\n[INFO] Columnas encontradas: {df.columns.tolist()}")
        print(f"[INFO] Dimensiones del DataFrame (filas, columnas): {df.shape}")

        if 'vector' in df.columns and 'label' in df.columns:
            print("[SUCCESS] Se encontraron las columnas 'vector' y 'label'.")
            
            # Verificar la dimensión del vector
            if not df.empty:
                first_vector = df['vector'].iloc[0]
                vector_dim = len(first_vector) if hasattr(first_vector, '__len__') else 'N/A'
                print(f"[INFO] Dimensión del primer vector: {vector_dim}")
            else:
                print("[WARNING] El DataFrame está vacío, no se puede verificar la dimensión del vector.")

            # Mostrar las primeras filas
            print("\n[DATA] Primeras 5 filas del archivo:")
            print(df.head())
        else:
            print("[ERROR] No se encontraron las columnas esperadas ('vector', 'label').")

    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error al leer o procesar el archivo: {e}")

if __name__ == "__main__":
    main() 