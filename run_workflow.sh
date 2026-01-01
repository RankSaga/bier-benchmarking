#!/bin/bash
# Complete workflow script for RankSaga benchmarking

set -e

echo "============================================================"
echo "RankSaga BEIR Benchmarking Workflow"
echo "============================================================"

# Step 1: Get baseline results (if not already downloaded)
echo ""
echo "Step 1: Checking baseline results..."
if [ ! -f "results/baseline/e5_base_v2_results.json" ] || [ ! -s "results/baseline/e5_base_v2_results.json" ]; then
    echo "⚠️  Baseline results not found or empty."
    echo "💡 Access via Modal dashboard: https://modal.com/apps -> ranksaga-beir-benchmark"
    echo "   Or the results should already be in the Modal volume from the previous run."
else
    echo "✅ Baseline results found"
fi

# Step 2: Fine-tune model
echo ""
echo "Step 2: Fine-tuning model on Modal..."
echo "Running: modal run modal_fine_tune.py"
modal run modal_fine_tune.py

# Step 3: Benchmark optimized model
echo ""
echo "Step 3: Benchmarking optimized model on Modal..."
echo "Running: modal run modal_app.py::benchmark_optimized"
modal run modal_app.py::benchmark_optimized

# Step 4: Compare results
echo ""
echo "Step 4: Comparing results locally..."
echo "Running: python compare_results.py"
python compare_results.py

echo ""
echo "============================================================"
echo "✅ Workflow complete!"
echo "============================================================"
echo ""
echo "Results available in:"
echo "  - results/baseline/"
echo "  - results/optimized/"
echo "  - results/comparison_table.csv"
echo "  - results/comparison_visualization.png"
echo "  - results/comparison_report.md"

