#!/usr/bin/env python
"""
scripts/extract_delta_from_composite.py – Extrae el vector de diferencia de un vector compuesto
================================================================================================
Carga un archivo parquet que contiene un 'vector' compuesto y una 'label'.
El vector compuesto se espera que sea la concatenación de [premise, conclusion, difference].
Este script extrae la tercera parte (difference) y la guarda en un nuevo archivo parquet
con la 'label', nombrando la columna del vector de diferencia como 'vector' para
compatibilidad con el pipeline.
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Extracts the difference vector from a composite vector parquet file.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input composite vector parquet file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output difference vector parquet file.")
    return parser.parse_args()

def main():
    """Función principal."""
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print(f"Leyendo vectores compuestos desde: {input_path}")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error: No se pudo leer el archivo {input_path}. {e}")
        exit(1)

    if 'vector' not in df.columns or 'label' not in df.columns:
        print(f"Error: Se requieren las columnas 'vector' y 'label'. Columnas encontradas: {df.columns.tolist()}")
        exit(1)

    # Convertir la columna de vectores a un array de numpy para un slicing eficiente
    composite_vectors = np.array(df['vector'].tolist())
    
    # El vector original fue concatenado como [premise, conclusion, difference].
    # Cada uno de estos tiene la misma dimensionalidad.
    vector_dim = composite_vectors.shape[1]
    if vector_dim % 3 != 0:
        print(f"Error: La dimensión del vector ({vector_dim}) no es divisible por 3. No se puede dividir en premisa, conclusión y diferencia.")
        exit(1)
    
    single_embedding_dim = vector_dim // 3
    
    # Extraer el vector de diferencia (la tercera parte)
    # El inicio es en 2 * single_embedding_dim, el final es el final del array
    difference_vectors = composite_vectors[:, 2 * single_embedding_dim:]

    # Crear el nuevo DataFrame
    df_diff = pd.DataFrame({
        'vector': list(difference_vectors),
        'label': df['label']
    })

    # Asegurar que el directorio de salida exista
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Guardando los vectores de diferencia extraídos en: {output_path}")
    df_diff.to_parquet(output_path)

    print("Extracción del vector de diferencia completada.")

if __name__ == "__main__":
    main() 