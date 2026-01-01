# Running BEIR Benchmarking in the Cloud

Your laptop is overheating? No problem! Here are several cloud options to run the benchmarking:

## Option 1: Modal.com (Recommended - Fast GPU)

**Best for**: Fast execution, pay-per-use, professional setup

See [MODAL_SETUP.md](./MODAL_SETUP.md) for detailed instructions.

### Quick Start:
```bash
pip install modal
modal token new
modal deploy modal_app.py
modal run modal_app.py
```

### Advantages:
- ✅ Fast GPU access (A10G/T4)
- ✅ Pay per use (~$0.40-0.70 per run)
- ✅ Persistent storage
- ✅ Professional infrastructure

---

## Option 2: Google Colab (Free GPU)

**Best for**: Quick setup, free GPU access, no credit card needed

### Steps:

1. **Open the Colab notebook**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click "File" → "Upload notebook"
   - Upload `run_benchmark_colab.ipynb` from this directory
   - OR copy the notebook content into a new Colab notebook

2. **Enable GPU**:
   - Click "Runtime" → "Change runtime type"
   - Set "Hardware accelerator" to "GPU" (T4)
   - Click "Save"

3. **Run all cells**:
   - Click "Runtime" → "Run all"
   - The notebook will:
     - Install dependencies
     - Download datasets
     - Run benchmarks on GPU
     - Display results
     - Allow you to download results

4. **Download results**:
   - The last cell will download the JSON results file
   - Or use: `Files` sidebar → right-click `results/baseline/e5_base_v2_results.json` → Download

### Advantages:
- ✅ Free GPU (T4) for ~12 hours
- ✅ No setup required
- ✅ Fast downloads
- ✅ Easy to share results

### Limitations:
- ⚠️ Session timeout after ~12 hours of inactivity
- ⚠️ GPU access may be limited during peak times

---

## Option 2: Kaggle Notebooks (Free GPU)

**Best for**: Alternative to Colab, also free

### Steps:

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create new notebook
3. Copy the Colab notebook code
4. Enable GPU: Settings → Accelerator → GPU T4 x2
5. Run all cells

### Advantages:
- ✅ Free GPU access
- ✅ 30 hours/week GPU time
- ✅ Persistent storage

---

## Option 3: RunPod / Vast.ai (Cheap GPU Rental)

**Best for**: Longer runs, more powerful GPUs, dedicated resources

### RunPod Setup:

1. **Sign up**: [RunPod.io](https://www.runpod.io/)
2. **Create Pod**:
   - Choose GPU (RTX 3090, A100, etc.)
   - Select PyTorch template
   - Start pod

3. **Upload code**:
   ```bash
   # In RunPod terminal
   git clone <your-repo>  # or upload files
   cd benchmarking
   pip install -r requirements.txt
   python run_baseline.py
   ```

4. **Download results**:
   - Use RunPod's file browser
   - Or SCP/SFTP to download files

### Vast.ai Setup:

Similar to RunPod, but often cheaper:
- [Vast.ai](https://vast.ai/)
- Search for GPU instances
- Connect via SSH
- Run benchmarks

### Cost:
- ~$0.20-0.50/hour for RTX 3090
- ~$1-2/hour for A100

---

## Option 4: AWS / GCP / Azure (Enterprise)

**Best for**: Production workloads, longer-term projects

### AWS SageMaker / GCP Colab Pro / Azure ML

These offer managed ML environments but require:
- Credit card
- More setup
- Higher costs for extended use

---

## Quick Start: Google Colab

**Fastest way to get started:**

1. Open: https://colab.research.google.com/
2. Upload `run_benchmark_colab.ipynb`
3. Enable GPU (Runtime → Change runtime type → GPU)
4. Run all cells
5. Download results when done

**Expected runtime**: 
- ~30-60 minutes for all 4 datasets on GPU
- Much faster than CPU on your laptop!

---

## Tips for Cloud Execution

1. **Batch processing**: The Colab notebook processes all queries at once (faster)
2. **GPU utilization**: Use batch_size=64 or higher for GPU
3. **Save progress**: Results are saved after each dataset
4. **Monitor costs**: Free tiers are generous, but watch usage on paid services

---

## Troubleshooting

### Colab GPU not available?
- Try again later (peak hours can be busy)
- Use CPU (slower but works)

### Out of memory?
- Reduce batch_size in the notebook
- Process datasets one at a time
- Use smaller datasets first

### Session disconnected?
- Results are saved after each dataset
- Re-run from the last completed dataset
- Use Colab Pro for longer sessions

---

## Next Steps After Cloud Run

Once you have results from the cloud:

1. **Download the JSON results file**
2. **Place it in**: `benchmarking/results/baseline/`
3. **Run locally** (lightweight):
   ```bash
   python compare_results.py  # When you have optimized results
   ```

The heavy computation is done in the cloud, local analysis is fast!

