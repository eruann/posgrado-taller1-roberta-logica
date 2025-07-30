#!/bin/bash
"""
run_pipeline.sh - Ejemplo de uso del Pipeline Principal
=============================================================
Script para ejecutar el pipeline principal con diferentes configuraciones.

Uso:
    ./run_pipeline.sh <dataset> [opciones]

Ejemplos:
    ./run_pipeline.sh snli
    ./run_pipeline.sh folio
    ./run_pipeline.sh both
    ./run_pipeline.sh snli --layers 10 11 --pca_components 25 50
    ./run_pipeline.sh snli --umap_neighbors 15 100 --slice_n_values 0 3
"""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

if [ $# -eq 0 ]; then
    print_error "Debe especificar un dataset: snli, folio, o both"
    echo "Uso: $0 <dataset> [opciones]"
    echo ""
    echo "Ejemplos:"
    echo "  $0 snli"
    echo "  $0 folio"
    echo "  $0 both"
    echo "  $0 snli --layers 10 11 --pca_components 25 50"
    echo "  $0 snli --umap_neighbors 15 100 --slice_n_values 0 3"
    echo "  $0 snli --umap_metrics_pca euclidean manhattan --kmeans_k_values 2 3"
    echo ""
    echo "Parámetros disponibles:"
    echo "  --layers <num1> <num2> ...     Capas a procesar (default: 9 10 11 12)"
    echo "  --pca_components <num1> <num2> Componentes PCA (default: 1 5 50)"
    echo "  --umap_neighbors <num1> <num2> Vecinos UMAP (default: 15 100 200)"
    echo "  --umap_min_dist <num1> <num2> Min dist UMAP (default: 0.1 0.5)"
    echo "  --umap_metrics_pca <m1> <m2>  Métricas UMAP para PCA (default: euclidean manhattan)"
    echo "  --umap_metrics_zca <m1> <m2>  Métricas UMAP para ZCA (default: euclidean)"
    echo "  --slice_n_values <num1> <num2> Valores de slicing (default: 0 3 5)"
    echo "  --kmeans_k_values <num1> <num2> Valores k para K-means (default: 2 3)"
    echo "  --reduction_types <t1> <t2>    Tipos de reducción (default: pca zca)"
    echo "  --probe_max_depth <num>        Profundidad máxima para probes (default: 4)"
    echo "  --skip_embeddings              Saltar extracción de embeddings"
    echo "  --skip_normalization           Saltar normalización"
    echo "  --skip_anisotropy              Saltar medición de anisotropía"
    exit 1
fi

DATASET=$1
shift

if [[ ! "$DATASET" =~ ^(snli|folio|both)$ ]]; then
    print_error "Dataset inválido: $DATASET"
    echo "Opciones válidas: snli, folio, both"
    exit 1
fi

print_info "Iniciando Pipeline Principal para dataset: $DATASET"
OUTPUT_DIR="data/${DATASET}/unified_pipeline"
mkdir -p "$OUTPUT_DIR"
print_info "Directorio de salida: $OUTPUT_DIR"

# Verificar que el script del pipeline existe
if [ ! -f "scripts/pipelines/10_run_pipeline.py" ]; then
    print_error "No se encontró el script del pipeline: scripts/pipelines/10_run_pipeline.py"
    exit 1
fi

print_info "Ejecutando pipeline principal..."
print_info "Parámetros adicionales: $@"

python scripts/pipelines/10_run_pipeline.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "unified_pipeline_${DATASET}" \
    "$@"

if [ $? -eq 0 ]; then
    print_success "Pipeline completado exitosamente!"
    print_info "Resultados guardados en: $OUTPUT_DIR"
    print_info "MLflow tracking disponible en: .mlflow/"
    echo ""
    print_info "Estructura de resultados:"
    tree "$OUTPUT_DIR" -L 3 2>/dev/null || ls -la "$OUTPUT_DIR"
    
    echo ""
    print_info "Configuraciones ejecutadas:"
    echo "  • Normalizaciones: none, all_but_mean, per_type, standard, cross_differences, arithmetic_mean, geometric_median"
    echo "  • Configuraciones: EC (Entailment-Contradiction), ECN (Entailment-Contradiction-Neutral)"
    echo "  • Capas: ${LAYERS:-9 10 11 12}"
    echo "  • Componentes PCA: ${PCA_COMPONENTS:-1 5 50}"
    echo "  • Vecinos UMAP: ${UMAP_NEIGHBORS:-15 100 200}"
    echo "  • Métricas UMAP PCA: ${UMAP_METRICS_PCA:-euclidean manhattan}"
    echo "  • Métricas UMAP ZCA: ${UMAP_METRICS_ZCA:-euclidean}"
    echo "  • Slicing: ${SLICE_N_VALUES:-0 3 5}"
    echo "  • K-means k: ${KMEANS_K_VALUES:-2 3}"
    echo "  • Tipos de reducción: ${REDUCTION_TYPES:-pca zca}"
    
else
    print_error "Pipeline falló. Revisar logs para más detalles."
    exit 1
fi

print_success "¡Pipeline principal completado!" 