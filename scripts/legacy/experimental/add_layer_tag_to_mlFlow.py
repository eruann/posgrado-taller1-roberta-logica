import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Force MLflow to use local mlruns directory
tracking_dir = Path.cwd().joinpath("mlruns")
mlflow.set_tracking_uri(tracking_dir.as_uri())

client = MlflowClient()

# 1) Recorremos todos los experimentos
experiments = client.search_experiments()  # This is the current way to list experiments

for exp in experiments:
    print(f"Processing experiment: {exp.name}")
    
    # 2) Obtenemos todos los runs del experimento
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    
    for run in runs:
        run_id = run.info.run_id
        print(f"  Processing run: {run_id}")

        # --- LÓGICA PARA DETERMINAR LA CAPA ---
        # Si ya sabes el nº de capa de antemano:
        layer = 12
        # O, por ejemplo, lo extraes de un param existente
        #   layer = int(run.data.params["model_name"].split("-")[1])

        # 3) Añadimos / sobre-escribimos el parámetro
        client.log_param(run_id, "layer_num", layer)
        print(f"    Added parameter layer_num={layer}")

print("\n✅ ¡Listo! Todos los runs ahora tienen el parámetro layer_num.")
