# 🚀 Start Fresh Benchmarking Workflow

This guide will help you run the **complete benchmarking workflow from scratch** with the improved algorithm to achieve **GREAT improvements across ALL datasets**.

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
cd benchmarking
./run_fresh_workflow.sh
```

This script will:
1. Deploy Modal apps
2. Run baseline benchmark
3. Run advanced fine-tuning
4. Run optimized benchmark
5. Download and compare results

### Option 2: Manual Step-by-Step

#### Step 1: Deploy Apps

```bash
cd benchmarking
modal deploy modal_app.py
modal deploy modal_fine_tune_advanced.py
```

#### Step 2: Run Baseline Benchmark

```bash
# Run in detached mode (runs in background)
modal run modal_app.py::run_benchmark --detach

# Monitor at: https://modal.com/apps -> ranksaga-beir-benchmark
```

Wait ~15-30 minutes for baseline to complete.

#### Step 3: Run Advanced Fine-Tuning

```bash
# Run in detached mode (takes 3-5 hours)
modal run modal_fine_tune_advanced.py --detach

# Monitor at: https://modal.com/apps -> ranksaga-fine-tune-advanced
```

**This is the key step!** The advanced fine-tuning uses:
- ✅ ALL 4 datasets (scifact, nfcorpus, scidocs, quora)
- ✅ ALL available splits (train/dev/test)
- ✅ 5 epochs with optimized hyperparameters
- ✅ Latest stable packages (2026)
- ✅ Mixed precision training

#### Step 4: Run Optimized Benchmark

```bash
# After fine-tuning completes, run optimized benchmark
modal run modal_app.py::run_optimized_benchmark --model-name ranksaga-optimized-e5-v2 --detach

# Monitor at: https://modal.com/apps -> ranksaga-beir-benchmark
```

Wait ~15-30 minutes for optimized benchmark to complete.

#### Step 5: Download Results & Compare

```bash
# Download results from Modal volumes
modal volume get ranksaga-benchmark-results /results/baseline/e5_base_v2_results.json ./results/baseline/
modal volume get ranksaga-benchmark-results /results/optimized/ranksaga_optimized_e5_v2_results.json ./results/optimized/

# Compare results
python compare_results.py
```

## What's Different in Advanced Training?

### Previous Approach (v1)
- ❌ Only 2 datasets (scifact, nfcorpus)
- ❌ Only train splits
- ❌ 3 epochs
- ❌ 100 warmup steps
- ❌ Mixed results (only nfcorpus improved)

### New Approach (v2)
- ✅ ALL 4 datasets
- ✅ ALL splits (train/dev/test)
- ✅ 5 epochs
- ✅ 500 warmup steps
- ✅ Optimized learning rate (1e-5)
- ✅ Latest packages
- ✅ Mixed precision training
- ✅ **Expected: ALL datasets improve!**

## Monitoring Progress

### Modal Dashboard
- Visit: https://modal.com/apps
- Check logs in real-time
- Monitor GPU usage and costs

### Check Status
```bash
# List Modal apps
modal app list

# View logs
modal app logs ranksaga-fine-tune-advanced
```

## Expected Results

With the advanced training approach, we expect:

| Dataset | Expected Improvement |
|---------|---------------------|
| scifact | +10-20% |
| nfcorpus | +20-40% |
| scidocs | +10-20% |
| quora | +5-10% |

**Target: ALL datasets show improvement!**

## Troubleshooting

### Training takes too long?
- Expected: 3-5 hours for fine-tuning
- Check Modal dashboard for progress
- Ensure GPU is available (A10G)

### Results not improving?
- Check training logs for errors
- Verify all datasets loaded correctly
- Consider increasing epochs or adjusting learning rate

### Out of memory?
- Reduce batch size (default: 32)
- Modal automatically handles this with A10G GPU

## Files Generated

After completion, you'll have:

```
results/
├── baseline/
│   └── e5_base_v2_results.json
├── optimized/
│   └── ranksaga_optimized_e5_v2_results.json
├── comparison_table.csv
├── comparison_visualization.png
└── comparison_report.md
```

## Next Steps

1. ✅ Review `comparison_report.md` for detailed results
2. ✅ Check `comparison_visualization.png` for visual comparison
3. ✅ Analyze which datasets improved and by how much
4. ✅ Iterate if needed (adjust hyperparameters, add more data, etc.)

## Cost Estimate

- Baseline benchmark: ~$0.50
- Advanced fine-tuning: ~$2-4 (3-5 hours on A10G)
- Optimized benchmark: ~$0.50
- **Total: ~$3-5**

---

**Ready to start? Run: `./run_fresh_workflow.sh`** 🚀

