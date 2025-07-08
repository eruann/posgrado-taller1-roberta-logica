#!/bin/bash
#
# scripts/run_all_probes.sh
# =========================
#
# This script automates the execution of the Decision Tree Probe across
# multiple RoBERTa layers and normalization strategies.
# It iterates through predefined layers and normalization types, constructing
# the necessary paths and executing the probe for each combination.
#

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
EMBEDDING_MODE="full" # Can be 'full' or 'delta'
LAYERS=(9 10 11 12)
NORMALIZATION_TYPES=("none" "per_type" "all_but_mean")
BASE_INPUT_DIR="data/snli/norm_comp_${EMBEDDING_MODE}_ec_only"
BASE_OUTPUT_DIR="data/snli/probes"
EXPERIMENT_NAME="DecisionTree Probe - SNLI EC (PCA ${EMBEDDING_MODE})"
MAX_DEPTH=3

echo "🚀 Starting batch execution of Decision Tree Probes for EMBEDDING_MODE=${EMBEDDING_MODE}..."
echo "Layers: ${LAYERS[@]}"
echo "Normalization Types: ${NORMALIZATION_TYPES[@]}"
echo "----------------------------------------------------"

# --- Main Loop ---
for norm_type in "${NORMALIZATION_TYPES[@]}"; do
    for layer in "${LAYERS[@]}"; do
        echo
        echo "========================================================================"
        echo "▶️  RUNNING PROBE: LAYER=${layer}, NORMALIZATION=${norm_type}, MODE=${EMBEDDING_MODE}"
        echo "========================================================================"

        # Construct paths and arguments dynamically
        INPUT_DIR="${BASE_INPUT_DIR}/02_pca_${EMBEDDING_MODE}_${norm_type}/50_components"
        # The PCA script names the output file based on the normalization type
        INPUT_FILE="${INPUT_DIR}/pca_${EMBEDDING_MODE}_snli_layer${layer}_${norm_type}.parquet"
        
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/dt_probe_pca_${EMBEDDING_MODE}_${norm_type}_layer_${layer}"
        
        EMBEDDING_TYPE="pca_${EMBEDDING_MODE}_${norm_type}"

        # Check if the required input file exists before running
        if [ ! -f "$INPUT_FILE" ]; then
            echo "⚠️  WARNING: Input file not found, skipping."
            echo "   (Searched for: ${INPUT_FILE})"
            continue
        fi

        # Execute the main probe script
        python -m scripts.run_decision_tree_probe \
            --input_path "${INPUT_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --experiment_name "${EXPERIMENT_NAME}" \
            --embedding_type "${EMBEDDING_TYPE}" \
            --layer_num "${layer}" \
            --max_depth "${MAX_DEPTH}"

        echo "✅ DONE: LAYER=${layer}, NORMALIZATION=${norm_type}, MODE=${EMBEDDING_MODE}"
    done
done

echo
echo "🎉 All probe experiments launched successfully for EMBEDDING_MODE=${EMBEDDING_MODE}!"
echo "Check MLflow UI for results."
echo "----------------------------------------------------" 