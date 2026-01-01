#!/bin/bash

# Run domain-specific fine-tuning and benchmarking workflow
# This trains separate models for scientific and general domains

set -e  # Exit on error

echo "============================================================"
echo "RankSaga Domain-Specific Fine-Tuning Workflow"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Train scientific domain model (scifact, nfcorpus, scidocs)"
echo "  2. Train general domain model (quora)"
echo "  3. Benchmark both models on their respective domains"
echo "  4. Compare with baseline"
echo ""
echo "Starting in 5 seconds... (Ctrl+C to cancel)"
sleep 5

cd "$(dirname "$0")"

# Step 1: Deploy apps
echo ""
echo "============================================================"
echo "Step 1: Deploying Modal apps..."
echo "============================================================"
modal deploy modal_fine_tune_domain_specific.py
modal deploy modal_app_domain_specific.py

# Step 2: Train all domain models
echo ""
echo "============================================================"
echo "Step 2: Training domain-specific models..."
echo "============================================================"
echo "This will train:"
echo "  - Scientific domain model (scifact, nfcorpus, scidocs)"
echo "  - General domain model (quora)"
echo ""
echo "⚠️  This will run in the foreground. It may take 3-5 hours per domain."
echo "   You can monitor progress at: https://modal.com/apps"
echo ""
modal run modal_fine_tune_domain_specific.py --all-domains

# Step 3: Benchmark domain models
echo ""
echo "============================================================"
echo "Step 3: Benchmarking domain-specific models..."
echo "============================================================"
echo "Benchmarking models on their respective domains..."
echo ""
modal run modal_app_domain_specific.py::benchmark_all_domains

# Step 4: Download results
echo ""
echo "============================================================"
echo "Step 4: Downloading results..."
echo "============================================================"

# Create directories
mkdir -p results/domain_specific/scientific
mkdir -p results/domain_specific/general

# Download results
echo "Downloading scientific domain results..."
modal volume get ranksaga-benchmark-results /results/domain_specific/scientific/ranksaga_optimized_e5_v2_scientific_results.json ./results/domain_specific/scientific/ 2>/dev/null || echo "⚠️  Scientific results not found"

echo "Downloading general domain results..."
modal volume get ranksaga-benchmark-results /results/domain_specific/general/ranksaga_optimized_e5_v2_general_results.json ./results/domain_specific/general/ 2>/dev/null || echo "⚠️  General results not found"

echo ""
echo "============================================================"
echo "✅ Domain-Specific Workflow Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/domain_specific/scientific/"
echo "  - results/domain_specific/general/"
echo ""
echo "Next: Compare domain-specific results with baseline using compare_results.py"
echo ""

