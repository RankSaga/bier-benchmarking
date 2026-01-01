#!/bin/bash

# Run the complete benchmarking workflow from scratch
# This script runs baseline, advanced fine-tuning, and optimized benchmarking

set -e  # Exit on error

echo "============================================================"
echo "RankSaga Advanced Benchmarking Workflow - Fresh Start"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Run baseline benchmark"
echo "  2. Run advanced fine-tuning (ALL datasets, better hyperparameters)"
echo "  3. Run optimized benchmark"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

cd "$(dirname "$0")"

# Step 1: Deploy apps
echo ""
echo "============================================================"
echo "Step 1: Deploying Modal apps..."
echo "============================================================"
modal deploy modal_app.py
modal deploy modal_fine_tune_advanced.py

# Step 2: Run baseline benchmark
echo ""
echo "============================================================"
echo "Step 2: Running baseline benchmark..."
echo "============================================================"
echo "Running baseline benchmark..."
echo "Monitor at: https://modal.com/apps -> ranksaga-beir-benchmark"
echo ""
echo "⚠️  This will run in the foreground. It may take 15-30 minutes."
echo "   You can monitor progress at: https://modal.com/apps"
echo ""
modal run modal_app.py::run_benchmark

# Step 3: Run advanced fine-tuning
echo ""
echo "============================================================"
echo "Step 3: Running advanced fine-tuning..."
echo "============================================================"
echo "This uses:"
echo "  - ALL datasets (scifact, nfcorpus, scidocs, quora)"
echo "  - ALL available splits (train/dev/test)"
echo "  - 5 epochs with optimized hyperparameters"
echo "  - Latest stable packages (2026)"
echo ""
echo "⚠️  This will run in the foreground. It may take 3-5 hours."
echo "   You can monitor progress at: https://modal.com/apps"
echo "   Note: You can safely close your terminal - the job runs on Modal's servers"
echo ""
echo "Starting fine-tuning..."
modal run modal_fine_tune_advanced.py

# Step 4: Run optimized benchmark
echo ""
echo "============================================================"
echo "Step 4: Running optimized benchmark..."
echo "============================================================"
echo "Running optimized benchmark..."
echo "Monitor at: https://modal.com/apps -> ranksaga-beir-benchmark"
echo ""
echo "⚠️  This will run in the foreground. It may take 15-30 minutes."
echo "   You can monitor progress at: https://modal.com/apps"
echo ""
modal run modal_app.py::benchmark_optimized --model-name ranksaga-optimized-e5-v2

# Step 5: Download results and compare
echo ""
echo "============================================================"
echo "Step 5: Downloading results and comparing..."
echo "============================================================"

# Download results from Modal volumes
echo "Downloading results from Modal..."
modal volume get ranksaga-benchmark-results /results/baseline/e5_base_v2_results.json ./results/baseline/ 2>/dev/null || echo "⚠️  Baseline results not found (may still be running)"
modal volume get ranksaga-benchmark-results /results/optimized/ranksaga_optimized_e5_v2_results.json ./results/optimized/ 2>/dev/null || echo "⚠️  Optimized results not found (may still be running)"

# Compare results
if [ -f "./results/baseline/e5_base_v2_results.json" ] && [ -f "./results/optimized/ranksaga_optimized_e5_v2_results.json" ]; then
    echo ""
    echo "Comparing results..."
    python compare_results.py
    echo ""
    echo "============================================================"
    echo "✅ Workflow Complete!"
    echo "============================================================"
    echo "Results saved to: ./results/"
    echo "  - comparison_table.csv"
    echo "  - comparison_visualization.png"
    echo "  - comparison_report.md"
else
    echo ""
    echo "⚠️  Results not yet available. Please wait for benchmarks to complete, then run:"
    echo "   modal volume get ranksaga-benchmark-results /results/baseline/e5_base_v2_results.json ./results/baseline/"
    echo "   modal volume get ranksaga-benchmark-results /results/optimized/ranksaga_optimized_e5_v2_results.json ./results/optimized/"
    echo "   python compare_results.py"
fi

echo ""
echo "Done!"

