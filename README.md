# Observabilidad de la Inferencia Lógica en Modelos de Lenguaje Transformer

Este repositorio contiene la implementación de un pipeline completo para analizar la **observabilidad de la inferencia lógica** en modelos de lenguaje tipo transformer. El trabajo se enfoca en entender cómo los modelos de tipo transformer, específicamente RoBERTa procesan y representan información lógica a través de diferentes capas del modelo.

## Objetivo del Trabajo Práctico

El objetivo principal es investigar la **capacidad de inferencia lógica** de modelos transformer mediante:

- **Extracción de embeddings** de diferentes capas del modelo
- **Análisis de anisotropía** para entender la distribución de representaciones
- **Reducción dimensional** (PCA + UMAP) para visualización
- **Clustering** para identificar patrones en las representaciones
- **Análisis contrastivo** para comparar diferentes tipos de inferencia
- **Probing** con árboles de decisión para evaluar capacidad predictiva
- **Validación estadística** de los resultados

## Arquitectura del Pipeline

El pipeline está diseñado como una serie de experimentos modulares que procesan datos de inferencia lógica:

```
Pipeline Principal
├── 1. Extracción de Embeddings (10_embeddings.py)
├── 2. Separación de Configuraciones (15_separate_configurations.py)
├── 3. Normalización (20_normalization.py)
├── 4. Análisis de Anisotropía (70_anisotropy_analysis.py)
├── 5. Reducción Dimensional (31_pca.py, 32_umap.py)
├── 6. Clustering (40_clustering_analysis.py)
├── 7. Análisis Contrastivo (50_contrastive_analysis.py)
├── 8. Probing (60_decision_tree_probe.py)
└── 9. Validación Estadística (90_statistical_validation.py)
```

## Estructura del Repositorio

```
tesis-llm-lpo/
├── data/                          # Datasets (ignorado por git)
│   ├── folio/
│   │   └── dataset/                  # Dataset FOLIO en formato Arrow
│   └── snli/
│       └── dataset/                  # Dataset SNLI en formato Arrow
├── experiments/                   # Scripts individuales de experimentos
│   ├── 10_embeddings.py             # Extracción de embeddings
│   ├── 15_separate_configurations.py # Separación EC/ECN
│   ├── 20_normalization.py          # Normalización de embeddings
│   ├── 31_pca.py                    # Reducción dimensional PCA
│   ├── 32_umap.py                   # Reducción dimensional UMAP
│   ├── 40_clustering_analysis.py    # Análisis de clustering
│   ├── 50_contrastive_analysis.py   # Análisis contrastivo
│   ├── 60_decision_tree_probe.py    # Probing con árboles
│   ├── 70_anisotropy_analysis.py    # Análisis de anisotropía
│   └── 90_statistical_validation.py # Validación estadística
├── scripts/                      # Scripts de utilidad y pipelines
│   ├── pipelines/
│   │   └── 10_run_pipeline.py      # Pipeline principal
│   └── utilities/                   # Scripts de utilidad
├── output/                      # Resultados (creado en ejeucucion)
├── mlruns/                       # Logs de MLflow (creado en ejecucion)
├── reports/                      # Reportes y documentación
├── reports_src/                  # Fuentes de reportes
├── requirements.txt                  # Dependencias Python
├── run_pipeline.sh                  # Script de ejecución
└── README.md                        # Este archivo
```

## Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd tesis-llm-lpo
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Preparar Datasets
Los datasets deben estar en formato **Apache Arrow** (.arrow) en la siguiente estructura:

```
data/
├── folio/
│   └── dataset/
│       ├── data-00000-of-00001.arrow
│       ├── dataset_info.json
│       └── state.json
└── snli/
    └── dataset/
        ├── data-00000-of-00001.arrow
        ├── dataset_info.json
        └── state.json
```


## Uso del Pipeline

### Ejecución Completa
```bash
python3 scripts/pipelines/10_run_pipeline.py \
    --dataset folio \
    --layers 9 10 11 12 \
    --output_dir ./output \
    --configurations EC ECN \
    --normalization_methods none all_but_mean per_type standard \
    --pca_components 50 \
    --umap_neighbors 15 \
    --umap_min_dist 0.1 \
    --umap_metrics_pca euclidean \
    --reduction_types pca \
    --slice_n_values 0 \
    --kmeans_k_values 2 3 \
    --probe_max_depth 4 \
    --experiment_name "observabilidad_inferencia_logica"
```

### Ejecución Rápida (Test)
```bash
python3 scripts/pipelines/10_run_pipeline.py \
    --dataset folio \
    --layers 9 \
    --output_dir ./output_test \
    --configurations EC \
    --normalization_methods none \
    --pca_components 50 \
    --umap_neighbors 15 \
    --umap_min_dist 0.1 \
    --umap_metrics_pca euclidean \
    --reduction_types pca \
    --slice_n_values 0 \
    --kmeans_k_values 2 \
    --probe_max_depth 4 \
    --experiment_name "test_rapido"
```

### Parámetros Principales

| Parámetro | Descripción | Valores por Defecto |
|-----------|-------------|-------------------|
| `--dataset` | Dataset a procesar | `folio`, `snli`, `both` |
| `--layers` | Capas del modelo a analizar | `[9, 10, 11, 12]` |
| `--configurations` | Configuraciones lógicas | `EC` (Entailment-Contradiction), `ECN` (Entailment-Contradiction-Neutral) |
| `--normalization_methods` | Métodos de normalización | `none`, `all_but_mean`, `per_type`, `standard` |
| `--pca_components` | Componentes PCA | `[1, 5, 50]` |
| `--umap_neighbors` | Vecinos UMAP | `[15, 100, 200]` |
| `--kmeans_k_values` | Valores k para clustering | `[2, 3]` |

## Configuraciones de Análisis

### Configuraciones Lógicas
- **EC**: Entailment-Contradiction (solo dos clases)
- **ECN**: Entailment-Contradiction-Neutral (tres clases)

### Métodos de Normalización
- **none**: Sin normalización (baseline)
- **all_but_mean**: Normalización global
- **per_type**: Normalización por tipo de inferencia
- **standard**: Estandarización Z-score

## Seguimiento de Experimentos

El pipeline utiliza **MLflow** para el seguimiento de experimentos:

```bash
# Ver interfaz de MLflow
mlflow ui

# Acceder desde navegador
# http://localhost:5000
```

### Convención de Nombres de Runs
Los runs siguen el patrón: `{run_id}_{config}_layer_{layer}_{step}_{params}`

Ejemplo: `id4ccde200_EC_layer_9_32_umap_pca50_none_n15_d0.1_meuclidean`

## Resultados

Los resultados se guardan en:
- **Embeddings**: `output/embeddings/`
- **Normalizados**: `output/normalized/{config}/{method}/`
- **Anisotropía**: `output/anisotropy/{method}/`
- **Reducción Dimensional**: `output/dimensionality/{config}/{method}/`
- **Clustering**: `output/clustering/{config}/{method}/k{k}/`
- **Contrastivo**: `output/contrastive/{config}/{method}/`
- **Probing**: `output/probes/{config}/{method}/`
- **Estadístico**: `output/statistical/`

## Pipeline Self-Healing

El pipeline incluye un sistema de **checkpointing** que permite:
- **Reanudar** ejecuciones interrumpidas
- **Saltar** pasos ya completados
- **Mantener** consistencia de run IDs

## Scripts de Utilidad

- `scripts/get_experiment_info.py`: Información de experimentos MLflow
- `scripts/export_mlflow_data.py`: Exportar datos de MLflow
- `scripts/consolidate_probe_results.py`: Consolidar resultados de probing
- `scripts/verify_embeddings.py`: Verificar embeddings generados

## Contexto Teórico

Este trabajo se enmarca en la investigación de **interpretabilidad de modelos de lenguaje**, específicamente:

1. **Observabilidad**: Capacidad de observar y medir el comportamiento interno del modelo
2. **Inferencia Lógica**: Procesamiento de relaciones lógicas (entailment, contradicción, neutralidad)
3. **Representaciones**: Análisis de cómo se codifica la información lógica en las capas del transformer
4. **Análisis Dimensional**: Reducción y visualización de espacios de alta dimensionalidad

## Contribuciones

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar cambios
4. Ejecutar tests
5. Crear Pull Request

## Licencia

Este proyecto es parte de un trabajo académico sobre observabilidad de inferencia lógica en modelos transformer.

---