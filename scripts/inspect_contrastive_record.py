#!/usr/bin/env python
"""
Inspecciona y compara los registros de un único hash de premisa
a través de los diferentes archivos de salida del análisis contrastivo.

Este script ayuda a visualizar la estructura de los datos generados por:
- arithmetic_mean
- geometric_median
- cross_differences

Genera un informe en: reports/contrastive_inspection/

Uso:
    python scripts/inspect_contrastive_record.py --layer 12
"""

import argparse
from pathlib import Path

try:
    import cudf
    # Import pandas para la conversión y visualización
    import pandas as pd
    print("Usando cuDF para la carga de datos.")
    USING_CUDF = True
except ImportError:
    print("cuDF no encontrado. Usando pandas. El rendimiento puede ser menor.")
    import pandas as cudf
    USING_CUDF = False


def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Capa del modelo cuyos resultados se van a inspeccionar (ej. 9, 10, 11, 12)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/snli/contrastive_analysis",
        help="Directorio base donde se encuentran los archivos de análisis contrastivo."
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default="ec",
        help="Tipo de comparación a inspeccionar ('ec' o 'en')."
    )
    return parser.parse_args()

def format_vector(row, num_features=5):
    """Formatea un vector para mostrarlo de forma truncada."""
    # En pandas, las features están en el mismo `row`. En cuDF, pueden estar en un objeto diferente.
    # Esta función asume que se le pasa una fila de pandas.
    feature_cols = [c for c in row.index if str(c).startswith('feature_')]
    if not feature_cols:
        return "No features found"
    
    vector = row[feature_cols].values
    
    if len(vector) <= 2 * num_features:
        return f"[{', '.join(f'{v:.4f}' for v in vector)}]"

    start = ", ".join(f'{v:.4f}' for v in vector[:num_features])
    end = ", ".join(f'{v:.4f}' for v in vector[-num_features:])
    
    return f"[{start}, ..., {end}] (dims: {len(vector)})"

def main():
    """Función principal del script."""
    args = parse_args()
    base_path = Path(args.base_dir)
    layer = args.layer
    comparison = args.comparison
    
    # --- Configuración del archivo de salida ---
    report_dir = Path("reports/contrastive_inspection")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file_path = report_dir / f"inspection_layer_{layer}_{comparison}.txt"

    with open(report_file_path, "w", encoding="utf-8") as report_file:

        def log(message):
            """Función anidada para escribir en consola y archivo."""
            print(message)
            report_file.write(message + "\n")

        methods = {
            "arithmetic_mean": f"contrastive_arithmetic_mean_{comparison}_layer_{layer}.parquet",
            "geometric_median": f"contrastive_geometric_median_{comparison}_layer_{layer}.parquet",
            "cross_differences": f"contrastive_cross_differences_{comparison}_layer_{layer}.parquet",
        }

        dataframes = {}
        log(f"Cargando datos para la capa {layer}...\n")
        for method, filename in methods.items():
            file_path = base_path / filename
            if not file_path.exists():
                log(f"ADVERTENCIA: No se encontró el archivo para '{method}': {file_path}")
                continue
            try:
                dataframes[method] = cudf.read_parquet(file_path)
            except Exception as e:
                log(f"ERROR: No se pudo cargar el archivo {file_path}. Error: {e}")
                return

        if not dataframes:
            log("No se cargó ningún archivo. Saliendo.")
            return

        # --- Obtención del hash de forma robusta ---
        first_method = list(dataframes.keys())[0]
        first_df = dataframes[first_method]

        if first_df.empty:
            log(f"El archivo para '{first_method}' está vacío. No se puede continuar.")
            return
        
        log(f"Columnas encontradas en '{methods[first_method]}': {first_df.columns.to_list()}")
        
        try:
            # Manera robusta de acceder a la columna y luego al primer elemento
            target_hash = first_df['premise_hash'].iloc[0]
            # Si es un objeto de cupy/cudf, convertir a tipo nativo de Python
            if hasattr(target_hash, 'item'):
                target_hash = target_hash.item()
        except (KeyError, IndexError):
            log("\nERROR: No se pudo encontrar la columna 'premise_hash' o el dataframe está vacío.")
            return

        log("\n" + "=" * 80)
        log(f"Inspeccionando todos los registros para el premise_hash: {target_hash}")
        log("=" * 80)

        for method, df in dataframes.items():
            log(f"\n--- MÉTODO: {method} (Archivo: {methods[method]}) ---")
            
            records_for_hash = df[df['premise_hash'] == target_hash]

            if records_for_hash.empty:
                log("No se encontraron registros para este hash con este método.")
                continue
            
            label_col = 'gold_label' if 'gold_label' in records_for_hash.columns else 'new_label'

            # Convertir a pandas para iterar y mostrar fácilmente
            records_pd = records_for_hash.to_pandas() if USING_CUDF else records_for_hash
            
            for _, row in records_pd.iterrows():
                log(f"  Registro:")
                log(f"    -> hash:  {row['premise_hash']}")
                log(f"    -> label: {row[label_col]}")
                log(f"    -> vector: {format_vector(row)}")
    
    print(f"\nInforme guardado en: {report_file_path}")

if __name__ == "__main__":
    main() 