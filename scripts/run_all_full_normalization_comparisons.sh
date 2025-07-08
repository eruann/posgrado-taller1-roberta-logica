#!/bin/bash
#
# scripts/run_all_normalization_comparisons.sh
# ============================================
#
# This script automates the execution of the normalization comparison pipeline
# across multiple RoBERTa layers for a specific embedding type (e.g., 'full' or 'delta').
# It iterates through predefined layers, preparing the environment and
# executing the main comparison script for each.
#

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
EMBEDDING_MODE="full" # Can be 'full' or 'delta'
LAYERS=(9 10 11 12)
DATASET_NAME="snli"

# Construct paths based on the configuration
BASE_INPUT_DIR="data/${DATASET_NAME}/embeddings/filtered/embeddings"
BASE_OUTPUT_DIR="data/${DATASET_NAME}/norm_comp_${EMBEDDING_MODE}_ec_only"


# --- Main Execution ---
echo "🚀 Starting batch execution of Normalization Comparison for EMBEDDING_MODE=${EMBEDDING_MODE}..."
echo "Layers: ${LAYERS[@]}"
echo "Input Dir: ${BASE_INPUT_DIR}"
echo "Output Dir: ${BASE_OUTPUT_DIR}"
echo "----------------------------------------------------"

# Create the main output directory. The Python script handles subdirectories.
mkdir -p "$BASE_OUTPUT_DIR"

for layer in "${LAYERS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "▶️  RUNNING NORMALIZATION: LAYER=${layer}, MODE=${EMBEDDING_MODE}"
    echo "========================================================================"

    # Execute the main normalization comparison script
    python scripts/run_normalization_comparison_fixed.py \
        --dataset_name "${DATASET_NAME}" \
        --embedding_type "${EMBEDDING_MODE}" \
        --full_embeddings_dir "${BASE_INPUT_DIR}" \
        --output_dir "${BASE_OUTPUT_DIR}" \
        --layer_num "${layer}" \
        --filter_to_ec

    echo "✓ Layer ${layer} processing complete."
done

echo ""
echo "🎉 Normalization comparison pipeline for EMBEDDING_MODE=${EMBEDDING_MODE} finished successfully." 