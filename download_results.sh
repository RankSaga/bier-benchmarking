#!/bin/bash
# Download results from Modal volumes

echo "Downloading results from Modal..."

# Create directories
mkdir -p results/baseline results/optimized

# Download baseline results (if not already present)
if [ ! -f "results/baseline/e5_base_v2_results.json" ]; then
    echo "Downloading baseline results..."
    modal volume get ranksaga-benchmark-results /results/baseline/e5_base_v2_results.json ./results/baseline/e5_base_v2_results.json 2>&1 || echo "Baseline results may already be in Modal volume"
fi

# Download optimized results
echo "Downloading optimized results..."
modal volume get ranksaga-benchmark-results /results/optimized/ranksaga_optimized_e5_v1_results.json ./results/optimized/ranksaga_optimized_e5_v1_results.json 2>&1 || echo "Optimized results may not be ready yet"

echo ""
echo "Results downloaded. Check:"
echo "  - results/baseline/"
echo "  - results/optimized/"

