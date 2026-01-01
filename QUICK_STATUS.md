# Quick Status Check

## If Modal Commands Are Stuck

The Modal CLI can sometimes hang. Here's how to check status:

### Option 1: Modal Dashboard (Recommended)
1. Go to: https://modal.com/apps
2. Find: `ranksaga-fine-tune`
3. Click on it to see:
   - Current runs
   - Logs
   - Status (running/completed/failed)

### Option 2: Check Recent Runs
```bash
# This should work even if logs are stuck
modal app list | grep ranksaga-fine-tune
```

### Option 3: Run Fine-Tuning Again
If previous runs failed, just run again:
```bash
cd benchmarking
modal run modal_fine_tune.py
```

## Common Issues

1. **No training data**: Some BEIR datasets might not have train splits
2. **Out of memory**: Try reducing batch_size
3. **Timeout**: Fine-tuning can take 1-2 hours

## Quick Fix: Run with Smaller Config

If stuck, try running with smaller settings:
```bash
modal run modal_fine_tune.py --epochs 1 --batch-size 16
```

This will be faster and use less memory.

