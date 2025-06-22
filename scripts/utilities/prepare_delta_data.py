#!/usr/bin/env python
"""
scripts/prepare_delta_data.py – Prepara los datos de vectores de diferencia
=============================================================================
Script de un solo uso para extraer los vectores de diferencia ('delta') de los
archivos de embeddings compuestos.

Toma los archivos de embeddings originales, que contienen un vector compuesto
[premise, conclusion, difference], y crea nuevos archivos .parquet que
contienen solo el vector de diferencia y la etiqueta.
"""

import argparse
import subprocess
from pathlib import Path

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Prepara los datos extrayendo los vectores de diferencia.")
    parser.add_argument("--source_embeddings_dir", default="data/snli/embeddings",
                      help="Directorio con los archivos de embeddings originales y compuestos.")
    parser.add_argument("--delta_output_dir", default="data/snli/difference_embeddings",
                      help="Directorio donde se guardarán los vectores de diferencia extraídos.")
    parser.add_argument("--original_dataset_name", default="snli", help="Nombre del dataset original (e.g., snli)")
    return parser.parse_args()

def run_command(cmd):
    """Ejecuta un comando y maneja errores."""
    print(f"Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Error ejecutando comando: {result.stderr}")
        raise RuntimeError(f"Comando falló: {' '.join(cmd)}")
    return result

def main():
    """Función principal."""
    args = parse_args()
    source_dir = Path(args.source_embeddings_dir)
    output_dir = Path(args.delta_output_dir)
    dataset_name_original = args.original_dataset_name
    dataset_name_delta = f"{dataset_name_original}_delta"

    print(f"Directorio de salida para vectores de diferencia: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Extrayendo vectores de diferencia para cada capa ---")
    for layer in range(9, 13):
        original_embedding_file = source_dir / f"embeddings_{dataset_name_original}_layer_{layer}.parquet"
        if not original_embedding_file.exists():
            print(f"Advertencia: {original_embedding_file} no encontrado, saltando capa {layer}.")
            continue
        
        delta_embedding_file = output_dir / f"embeddings_{dataset_name_delta}_layer_{layer}.parquet"
        
        print(f"  Procesando capa {layer}...")
        extract_cmd = [
            "python", "scripts/utilities/extract_delta_from_composite.py",
            "--input_path", str(original_embedding_file),
            "--output_path", str(delta_embedding_file)
        ]
        run_command(extract_cmd)
    
    print("\nPreparación de datos de vectores de diferencia completada.")
    print(f"Los archivos están guardados en: {output_dir}")

if __name__ == "__main__":
    main() 