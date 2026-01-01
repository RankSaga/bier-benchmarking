# Running BEIR Benchmarking on Modal.com

Modal.com provides fast GPU access for running the BEIR benchmarks. This guide will help you set it up and run the benchmarks.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) (free tier available)
2. **Modal CLI**: Install the Modal Python package

## Quick Setup

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate

```bash
modal token new
```

This will open a browser to authenticate with Modal.

### 3. Deploy the App

```bash
cd benchmarking
modal deploy modal_app.py
```

### 4. Run the Benchmark

```bash
# Run all datasets (recommended)
modal run modal_app.py

# Or run a specific dataset
modal run modal_app.py --dataset-name scifact
```

## GPU Configuration

The app is configured to use **A10G GPU** for maximum speed. You can modify the GPU type in `modal_app.py`:

```python
# Current (fastest): A10G
gpu=modal.gpu.A10G()

# Alternatives:
gpu=modal.gpu.T4()      # Cheaper, slower
gpu=modal.gpu.A100()    # Most powerful, expensive
```

## Expected Performance

- **A10G GPU**: ~20-40 minutes for all 4 datasets
- **T4 GPU**: ~40-60 minutes for all 4 datasets
- **CPU**: ~2-4 hours (not recommended)

## Cost Estimate

- **A10G**: ~$1.10/hour → ~$0.40-0.70 per full run
- **T4**: ~$0.40/hour → ~$0.30-0.40 per full run

## Retrieving Results

Results are automatically saved to a Modal volume. To download them:

```bash
# List volumes
modal volume list

# Download results (after run completes)
modal volume download ranksaga-benchmark-results /results/baseline/e5_base_v2_results.json ./results/baseline/
```

Or access via Modal dashboard:
1. Go to [modal.com/apps](https://modal.com/apps)
2. Find your app: `ranksaga-beir-benchmark`
3. View logs and results

## Monitoring

Watch the run in real-time:

```bash
modal app logs ranksaga-beir-benchmark
```

Or view in the Modal dashboard at [modal.com](https://modal.com)

## Troubleshooting

### Out of Memory
- Reduce batch_size in `manual_retrieve()` function
- Use T4 GPU instead of A10G
- Process datasets one at a time

### Timeout Issues
- Increase timeout in `modal_app.py` (currently 7200 seconds = 2 hours)
- Run datasets individually

### GPU Not Available
- Check Modal status: `modal gpu list`
- Try a different GPU type
- Check your Modal account limits

## Next Steps

After running the benchmark:

1. **Download results** to your local machine
2. **Place in** `benchmarking/results/baseline/`
3. **Run fine-tuning** (when ready): `python fine_tune_model.py`
4. **Run optimized benchmark**: `python run_optimized.py`
5. **Compare results**: `python compare_results.py`

## Advanced Usage

### Run Specific Datasets

```bash
modal run modal_app.py --dataset-name scifact
modal run modal_app.py --dataset-name nfcorpus
```

### Check Volume Contents

```bash
modal volume ls ranksaga-benchmark-results
```

### Delete Volume (if needed)

```bash
modal volume delete ranksaga-benchmark-results
```

## Benefits of Modal

- ✅ **Fast GPU access** - A10G/T4 available immediately
- ✅ **Pay per use** - Only charged while running
- ✅ **Persistent storage** - Results saved automatically
- ✅ **Scalable** - Handle large datasets easily
- ✅ **No laptop heating** - All computation in cloud

