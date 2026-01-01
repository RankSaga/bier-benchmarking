#!/bin/bash
# Run optimized benchmark in detached mode (keeps running even if terminal closes)

echo "Starting optimized benchmark in detached mode..."
echo "This will continue running even if you close this terminal."
echo ""
echo "Monitor progress at: https://modal.com/apps -> ranksaga-beir-benchmark"
echo ""

modal run --detach modal_app.py::benchmark_optimized

echo ""
echo "✅ Benchmark started in detached mode!"
echo "💡 Check Modal dashboard to monitor progress"
echo "💡 Results will be saved to Modal volume when complete"

