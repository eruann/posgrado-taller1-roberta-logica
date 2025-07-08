#!/usr/bin/env python
"""
Script para verificar la consistencia de los embeddings generados.

Comprueba que para una misma premisa de texto, el embedding de la premisa
sea idéntico en todos los registros, independientemente de la conclusión o la etiqueta.

Guarda un reporte detallado en la carpeta 'reports/'.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk

# --- Parámetros ---
EMBEDDINGS_PATH = "data/snli/embeddings/filtered/embeddings_snli_layer_9.parquet"
DATASET_PATH = "data/snli/dataset/snli_filtered" 
NUM_SAMPLES = 1000 
PREMISE_EMBEDDING_DIMS = 768
REPORTS_DIR = Path("reports")

def setup_logging():
    """Configura el logging para que escriba a consola y a un archivo."""
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "verify_embeddings_report.txt"
    
    # Configuración básica del logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(report_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Reporte guardado en: {report_path.resolve()}")

def main():
    setup_logging()

    logging.info(f"Cargando embeddings desde: {EMBEDDINGS_PATH}")
    df_embeds = pd.read_parquet(EMBEDDINGS_PATH)
    
    logging.info(f"Cargando dataset original desde: {DATASET_PATH}")
    ds = load_from_disk(DATASET_PATH)

    logging.info(f"Tomando las primeras {NUM_SAMPLES} muestras para la verificación...")
    df_sample = df_embeds.head(NUM_SAMPLES)
    
    # Extraer el texto de la premisa del dataset original
    premises_text = ds['premise'][:NUM_SAMPLES]
    
    # Extraer el embedding de la premisa (primeras 768 columnas)
    premise_embeddings = df_sample.iloc[:, :PREMISE_EMBEDDING_DIMS].values

    # Diccionario para almacenar el primer embedding encontrado para cada texto de premisa
    seen_premises = {}
    
    logging.info("\n--- Iniciando Verificación ---")
    inconsistencies = 0
    
    for i, text in enumerate(premises_text):
        current_embedding = premise_embeddings[i]
        
        if text not in seen_premises:
            # Es la primera vez que vemos esta premisa, la guardamos
            seen_premises[text] = current_embedding
            logging.info(f"Premisa nueva encontrada (idx {i}): '{text[:50]}...'")
        else:
            # Ya hemos visto esta premisa, comparamos los embeddings
            logging.info(f"Premisa repetida encontrada (idx {i}): '{text[:50]}...'")
            stored_embedding = seen_premises[text]
            
            # Usamos np.allclose para comparar arrays de punto flotante
            if not np.allclose(current_embedding, stored_embedding):
                inconsistencies += 1
                logging.warning(f"  [!] INCONSISTENCIA DETECTADA en el índice {i}!")
                # Opcional: calcular y mostrar la diferencia
                diff = np.linalg.norm(current_embedding - stored_embedding)
                logging.warning(f"      - Norma de la diferencia: {diff:.6f}")
            else:
                logging.info("  [✓] Consistente: El embedding es idéntico al anterior.")

    logging.info("\n--- Resultados de la Verificación ---")
    if inconsistencies == 0:
        logging.info("✅ Éxito: Todas las premisas idénticas tienen embeddings idénticos.")
    else:
        logging.error(f"❌ Fallo: Se encontraron {inconsistencies} inconsistencias.")

if __name__ == "__main__":
    main() 