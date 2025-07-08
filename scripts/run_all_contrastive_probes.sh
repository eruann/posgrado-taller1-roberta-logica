#!/bin/bash
#
# scripts/run_all_contrastive_probes.sh
# =====================================
# This script automates running the decision tree probe on the outputs
# of the contrastive analysis (experiments/02_contrastive_analysis.py).
#
# It iterates through all contrastive embedding files, identifies the
# method and layer, and runs the probe, saving the results in a
# structured directory tree organized by the contrastive method.
#

set -e

# --- Configuration ---
INPUT_DIR="data/snli/embeddings/contrastive"
BASE_OUTPUT_DIR="experiments/probes/from_contrastive"
MAX_DEPTH=3
EXPERIMENT_NAME="Probe_on_Contrastive_Embeddings"
LAYERS=(9 10 11 12)
METHODS=("arithmetic_mean" "geometric_median" "cross_differences")

echo "🚀 Starting batch probe execution on contrastive embeddings..."
echo "=============================================================="

for method in "${METHODS[@]}"; do
    echo -e "\nProcessing Method: $method"
    echo "--------------------------"

    for layer in "${LAYERS[@]}"; do
        # We only probe on 'ec' (Entailment/Contradiction) files because the probe is binary.
        input_file="${INPUT_DIR}/contrastive_${method}_ec_layer_${layer}.parquet"
        
        if [ ! -f "$input_file" ]; then
            echo "  - Layer ${layer}: SKIPPED (Input file not found: ${input_file})"
            continue
        fi

        echo "  - Layer ${layer}: Found input, starting probe..."

        # Define a structured output path
        output_dir="${BASE_OUTPUT_DIR}/${method}/layer_${layer}"
        mkdir -p "$output_dir"

        # Define a descriptive embedding type for MLflow logging
        embedding_type="contrastive_${method}"

        # Run the probe using 'python -m' to handle module paths correctly
        python -m scripts.run_decision_tree_probe \
            --input_path "$input_file" \
            --output_dir "$output_dir" \
            --experiment_name "$EXPERIMENT_NAME" \
            --embedding_type "$embedding_type" \
            --layer_num "$layer" \
            --max_depth "$MAX_DEPTH"
        
        echo "  - Layer ${layer}: Probe finished. Results in ${output_dir}"
    done
done

echo -e "\n🎉 All contrastive probe runs completed!"
echo "==============================================================" 