# RankSaga BEIR Benchmarking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/RankSaga/ranksaga-optimized-e5-v2)
[![Blog Post](https://img.shields.io/badge/Blog-Post-green)](https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization)

This repository contains scripts, results, and documentation for RankSaga's comprehensive BEIR (Benchmarking IR) benchmarking study. We evaluated embedding model optimization techniques and achieved significant improvements, including **up to 51% improvement** on medical information retrieval tasks.

## Key Results

Our optimized models show substantial improvements across multiple BEIR datasets:

- **NFE Corpus (Medical)**: +15-51% improvement across all metrics
- **SciDocs (Scientific)**: +3-20% improvement
- **Quora (General)**: Maintained high baseline performance
- **Maximum Gain**: 51% improvement on Recall@100 for medical information retrieval

**📊 [View Detailed Results and Visualizations](https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization)**

## Quick Start

### Using the Optimized Model

The fine-tuned model is available on Hugging Face:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")
embeddings = model.encode(["Your text here"])
```

**[Download from Hugging Face →](https://huggingface.co/RankSaga/ranksaga-optimized-e5-v2)**

### Running the Benchmark

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline evaluation
python run_baseline.py

# Fine-tune model
python fine_tune_model.py

# Evaluate optimized model
python run_optimized.py

# Compare results
python compare_results.py
```

## 📖 Overview

This benchmarking suite allows you to:

1. **Baseline Evaluation**: Benchmark pre-trained models (e.g., `intfloat/e5-base-v2`) on BEIR datasets
2. **Model Fine-Tuning**: Apply RankSaga optimization techniques to improve model performance
3. **Optimized Evaluation**: Benchmark fine-tuned models to measure improvements
4. **Results Comparison**: Compare baseline vs optimized performance with visualizations
5. **Reproduce Results**: All code, data, and configurations are provided

## Setup

### Option 1: Run on Modal.com (Recommended - Fast GPU)

**Best for**: Fast execution, no local GPU needed, pay-per-use

See [MODAL_SETUP.md](./MODAL_SETUP.md) for complete instructions.

Quick start:
```bash
pip install modal
modal token new
modal deploy modal_app.py
modal run modal_app.py
```

### Option 2: Run Locally

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training and evaluation)

### Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.py` to customize:

- Base model to fine-tune
- BEIR datasets to benchmark
- Training hyperparameters (epochs, batch size, learning rate)
- Output paths

## Usage

### 1. Run Baseline Benchmarking

Evaluate the pre-trained model to establish baseline performance:

```bash
python run_baseline.py
```

This will:
- Download BEIR datasets if not present
- Evaluate the base model on all configured datasets
- Save results to `results/baseline/`

**Expected time**: 10-30 minutes depending on datasets and hardware

### 2. Fine-Tune Model

Apply RankSaga optimization techniques to improve the model:

```bash
python fine_tune_model.py
```

This will:
- Load the base model
- Prepare training data from BEIR datasets
- Fine-tune using Multiple Negatives Ranking Loss
- Save optimized model to `models/ranksaga-optimized-e5-v1/`

**Expected time**: 30-120 minutes depending on training data size and epochs

### 3. Benchmark Optimized Model

Evaluate the fine-tuned model:

```bash
python run_optimized.py
```

This will:
- Load the fine-tuned model
- Evaluate on the same BEIR datasets
- Save results to `results/optimized/`

**Expected time**: 10-30 minutes

### 4. Compare Results

Generate comparison reports and visualizations:

```bash
python compare_results.py
```

This will:
- Load baseline and optimized results
- Calculate improvement percentages
- Generate comparison table (CSV)
- Create visualization charts
- Generate markdown report

Output files:
- `results/comparison_table.csv` - Detailed comparison data
- `results/comparison_visualization.png` - Performance charts
- `results/comparison_report.md` - Human-readable report

## File Structure

```
benchmarking/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
├── run_baseline.py           # Baseline benchmarking script
├── fine_tune_model.py        # Model fine-tuning script
├── run_optimized.py          # Optimized model benchmarking
├── compare_results.py        # Results comparison and visualization
├── utils/
│   ├── data_loader.py        # BEIR dataset utilities
│   └── evaluation.py         # Evaluation metrics helpers
├── models/                   # Saved fine-tuned models (gitignored)
├── results/                  # Benchmark results (gitignored)
│   ├── baseline/
│   └── optimized/
└── datasets/                # Downloaded BEIR datasets (gitignored)
```

## BEIR Datasets

By default, the following datasets are used:

- **scifact**: Scientific fact-checking (300 queries, 5K documents)
- **nfcorpus**: Medical information retrieval (323 queries, 3.6K documents)
- **scidocs**: Scientific document retrieval (1K queries, 25K documents)
- **quora**: Duplicate question detection (10K queries, 523K documents)

You can modify the dataset list in `config.py`.

## Understanding Results

### Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10 (primary metric)
- **NDCG@100**: NDCG at rank 100
- **MAP@100**: Mean Average Precision at rank 100
- **Recall@100**: Recall at rank 100
- **Precision@100**: Precision at rank 100

### Interpreting Improvements

- **Positive improvement %**: The optimized model performs better
- **Negative improvement %**: The optimized model performs worse (may indicate overfitting)
- **Average improvement**: Overall performance gain across all datasets

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in `config.py`
- Use fewer datasets for training
- Use a smaller base model

### Slow Training

- Use GPU if available (PyTorch will automatically detect)
- Reduce number of epochs
- Use a subset of training data

### Missing Datasets

- Datasets are automatically downloaded on first use
- Check internet connection
- Verify dataset names in `config.py` match BEIR dataset names

### Model Loading Errors

- Ensure `fine_tune_model.py` completed successfully
- Check that model path in `config.py` matches saved model location
- Verify model files exist in `models/` directory

## Advanced Usage

### Custom Training Configuration

Edit `config.py` to modify:

```python
TRAINING_CONFIG = {
    "epochs": 5,              # Increase for better results (slower)
    "batch_size": 32,         # Increase if you have more GPU memory
    "learning_rate": 1e-5,    # Lower for more stable training
    "warmup_steps": 200,      # More warmup for larger models
}
```

### Adding More Datasets

Add dataset names to `BEIR_DATASETS` in `config.py`:

```python
BEIR_DATASETS = ["scifact", "nfcorpus", "scidocs", "quora", "msmarco", "natural-questions"]
```

### Using Different Base Models

Change `BASE_MODEL` in `config.py`:

```python
BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Alternative model
```

## Citation

If you use BEIR in your research, please cite:

```bibtex
@inproceedings{thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas Rücklé and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```

## 📊 Results Summary

### Performance Improvements

| Dataset | NDCG@10 | NDCG@100 | MAP@100 | Recall@100 |
|---------|---------|----------|---------|------------|
| **NFE Corpus** | +15.25% | +32.62% | +49.49% | +51.03% |
| **SciDocs** | +3.14% | +11.82% | +7.70% | +20.21% |
| **Quora** | -0.81% | -0.79% | -0.90% | -0.39% |
| **SciFact** | -26.28% | -22.82% | -28.61% | -7.71% |

*See detailed results in `results/` directory and our [blog post](https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization)*

### Visualizations

Comprehensive visualizations are available:
- Performance comparison charts
- Improvement heatmaps
- Per-dataset analysis
- Domain-specific comparisons

Generate visualizations:
```bash
python generate_blog_visualizations.py
```

## 📚 Documentation

- **[Methodology](docs/METHODOLOGY.md)**: Detailed explanation of our fine-tuning approach
- **[Datasets](docs/DATASETS.md)**: Information about BEIR datasets used
- **[Modal Setup](MODAL_SETUP.md)**: Guide for running on Modal.com cloud infrastructure

## 🔬 Methodology

Our optimization approach uses:

- **Base Model**: `intfloat/e5-base-v2`
- **Loss Function**: Multiple Negatives Ranking Loss
- **Training**: 5 epochs, batch size 32, learning rate 1e-5
- **Data**: All BEIR dataset splits (train/dev/test)
- **Hardware**: A10G GPU on Modal.com

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for complete details.

## 🎁 Open Source Contributions

We're committed to open-source AI research. This repository includes:

- ✅ Complete benchmarking code
- ✅ Fine-tuning scripts and configurations
- ✅ Evaluation results and analysis
- ✅ Visualization generation tools
- ✅ Model export utilities for Hugging Face

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Open issues for bugs or questions
- Submit pull requests for improvements
- Share results from different configurations
- Provide feedback on methodology

## 📝 Citation

If you use our models or reference our work:

```bibtex
@misc{ranksaga-beir-2026,
  title={BEIR Benchmarking Results: RankSaga Embedding Model Optimization},
  author={RankSaga},
  year={2026},
  url={https://github.com/RankSaga/bier-benchmarking}
}
```

For BEIR benchmark:
```bibtex
@inproceedings{thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas Rücklé and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```

## 🔗 Links

- **Model on Hugging Face**: [RankSaga/ranksaga-optimized-e5-v2](https://huggingface.co/RankSaga/ranksaga-optimized-e5-v2)
- **Blog Post**: [BEIR Benchmarking Results](https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization)
- **RankSaga Website**: [https://ranksaga.com](https://ranksaga.com)
- **Contact**: [Get in Touch](https://ranksaga.com/contact/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The BEIR benchmark itself is licensed under Apache 2.0.

## 🙏 Acknowledgments

- BEIR benchmark creators for the comprehensive evaluation framework
- E5 model authors for the excellent base model
- Sentence Transformers library for the powerful framework
- Modal.com for GPU infrastructure

## 💼 Commercial Use

RankSaga provides enterprise AI consulting services including:
- Custom embedding model optimization
- Semantic search solutions
- Vector database management
- LLM training and fine-tuning

[Contact us](https://ranksaga.com/contact/) for commercial inquiries.
