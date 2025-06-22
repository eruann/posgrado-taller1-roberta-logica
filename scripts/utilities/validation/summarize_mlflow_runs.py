#!/usr/bin/env python
"""
scripts/summarize_mlflow_runs.py - Genera un resumen de todas las corridas de MLflow.
=====================================================================================
Este script lee el repositorio de MLflow en el directorio 'mlruns', extrae los
parámetros y métricas de cada corrida, y presenta un resumen en formato tabular
utilizando pandas.

El resumen incluye información clave como el nombre del experimento, el ID de la corrida,
y parámetros relevantes como el tipo de reducción, la capa del modelo, el número de
componentes, etc.
"""

import mlflow
import pandas as pd
import argparse
from pathlib import Path

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description="Genera un resumen de todas las corridas de MLflow y lo guarda en un CSV.")
    parser.add_argument("--output_csv", 
                        type=str, 
                        default="reports/mlflow_summary.csv",
                        help="Ruta del archivo CSV para guardar el resumen. Default: reports/mlflow_summary.csv")
    return parser.parse_args()

def summarize_runs(output_csv):
    """
    Carga todas las corridas de MLflow del repositorio local, las resume y
    guarda el resultado en un archivo CSV.
    """
    # Conectar al repositorio de MLflow local
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    # Obtener todos los experimentos
    experiments = client.search_experiments()

    all_runs_data = []

    for exp in experiments:
        exp_id = exp.experiment_id
        exp_name = exp.name
        
        # Buscar todas las corridas para el experimento actual
        runs = client.search_runs(experiment_ids=[exp_id], max_results=5000) # Aumentar si hay más de 1000 corridas por experimento
        
        for run in runs:
            run_data = {
                "experiment_name": exp_name,
                "run_id": run.info.run_id,
                "start_time": run.info.start_time,
            }
            # Añadir todos los parámetros
            run_data.update(run.data.params)
            # Añadir todas las métricas
            run_data.update(run.data.metrics)
            
            all_runs_data.append(run_data)

    if not all_runs_data:
        print("No se encontraron corridas en MLflow.")
        return

    # Convertir a DataFrame de pandas para una mejor visualización
    df_runs = pd.DataFrame(all_runs_data)
    
    # Convertir start_time a formato legible y ordenar
    df_runs['start_time'] = pd.to_datetime(df_runs['start_time'], unit='ms')
    df_runs = df_runs.sort_values(by="start_time", ascending=False)

    # Llenar valores NaN con un string vacío o un valor por defecto para evitar problemas en el guardado
    df_runs = df_runs.fillna('')

    # Seleccionar y reordenar columnas para una mejor legibilidad
    core_cols = ["experiment_name", "start_time"]
    metric_param_cols = sorted([col for col in df_runs.columns if col not in core_cols and col != "run_id"])
    
    all_cols = core_cols + metric_param_cols
    # Reordenar el dataframe
    df_runs = df_runs[[col for col in all_cols if col in df_runs.columns]]
    
    # Guardar en CSV
    try:
        output_path = Path(output_csv)
        # Asegurarse de que el directorio de salida exista
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_runs.to_csv(output_path, index=False)
        print(f"Resumen guardado exitosamente en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")

if __name__ == "__main__":
    args = parse_args()
    summarize_runs(args.output_csv) 