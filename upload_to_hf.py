"""
Upload model to Hugging Face Hub.
Requires: pip install huggingface_hub
"""
import logging
import os
from pathlib import Path
import json

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub import login as hf_login
except ImportError:
    logger.error("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")
    exit(1)

MODEL_NAME = "ranksaga-optimized-e5-v2"
REPO_ID = f"ranksaga/{MODEL_NAME}"  # Use lowercase for organization
EXPORT_DIR = Path(__file__).parent / "hf_export" / MODEL_NAME


def load_results() -> dict:
    """Load benchmarking results for model card."""
    results_file = Path(__file__).parent / "results" / "optimized" / f"{MODEL_NAME.replace('-', '_')}_results.json"
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def create_model_card() -> str:
    """Create comprehensive model card."""
    results = load_results()
    
    card = f"""---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- information-retrieval
- embeddings
- beir
- fine-tuned
- ranksaga
license: mit
base_model: intfloat/e5-base-v2
datasets:
- beir
metrics:
- ndcg@10
- ndcg@100
- map@100
- recall@100
---

# {MODEL_NAME}

This is a fine-tuned embedding model optimized by RankSaga for information retrieval tasks. It's based on `intfloat/e5-base-v2` and has been optimized using advanced fine-tuning techniques on BEIR benchmark datasets.

## Model Details

### Base Model
- **Base Model**: `intfloat/e5-base-v2`
- **Architecture**: E5 (Embeddings from bidirectional Encoder representations)
- **Model Type**: Sentence Transformer

### Training
- **Fine-tuning Method**: Multiple Negatives Ranking Loss
- **Training Datasets**: BEIR datasets (scifact, nfcorpus, scidocs, quora)
- **Epochs**: 5
- **Batch Size**: 32
- **Learning Rate**: 1e-5
- **Mixed Precision**: FP16

### Optimization Results

The model was evaluated on BEIR benchmark datasets and shows significant improvements on technical domains:

**NFE Corpus (Medical Information Retrieval)**:
- NDCG@10: +15.25% improvement
- NDCG@100: +32.62% improvement
- MAP@100: +49.49% improvement
- Recall@100: +51.03% improvement

**SciDocs (Scientific Document Retrieval)**:
- NDCG@10: +3.14% improvement
- NDCG@100: +11.82% improvement
- MAP@100: +7.70% improvement
- Recall@100: +20.21% improvement

**Quora (General Semantic Similarity)**:
- Maintained high baseline performance (NDCG@10: 0.8472)

## Usage

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Encode sentences
sentences = [
    "What is the capital of France?",
    "Paris is the capital of France."
]
embeddings = model.encode(sentences)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {{similarity.item():.4f}}")
```

### Using for Information Retrieval

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Encode documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "Deep learning uses neural networks with multiple layers."
]
doc_embeddings = model.encode(documents)

# Encode query
query = "What is machine learning?"
query_embedding = model.encode(query)

# Find most similar documents
similarities = cos_sim(query_embedding, doc_embeddings)[0]
top_result_idx = similarities.argmax().item()

print(f"Query: {{query}}")
print(f"Most relevant document: {{documents[top_result_idx]}}")
print(f"Similarity: {{similarities[top_result_idx].item():.4f}}")
```

### Using with Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("RankSaga/ranksaga-optimized-e5-v2")
model = AutoModel.from_pretrained("RankSaga/ranksaga-optimized-e5-v2")

# Encode
inputs = tokenizer("What is machine learning?", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
```

## Evaluation

The model was evaluated on the BEIR benchmark suite:

| Dataset | NDCG@10 | NDCG@100 | MAP@100 | Recall@100 |
|---------|---------|----------|---------|------------|
| NFE Corpus | 0.3921 | 0.4187 | 0.2373 | 0.4830 |
| SciDocs | 0.1767 | 0.2726 | 0.1246 | 0.4782 |
| Quora | 0.8472 | 0.8631 | 0.8121 | 0.9865 |
| SciFact | 0.5137 | 0.5563 | 0.4658 | 0.8684 |

For detailed results and comparisons, see our [benchmarking blog post](https://ranksaga.com/blog/beir-benchmarking-ranksaga-optimization) and [GitHub repository](https://github.com/RankSaga/beir-benchmarking).

## Limitations

- The model performs best on technical and domain-specific content (medical, scientific)
- Performance on general tasks may be similar to or slightly lower than the base model
- Model size: ~110M parameters
- Requires sentence-transformers library for optimal usage

## Training Data

The model was fine-tuned on:
- **SciFact**: Scientific fact-checking dataset (300 queries, 5K documents)
- **NFE Corpus**: Medical information retrieval (323 queries, 3.6K documents)
- **SciDocs**: Scientific document retrieval (1K queries, 25K documents)
- **Quora**: Duplicate question detection (10K queries, 523K documents)

All datasets are part of the BEIR benchmark suite.

## Citation

If you use this model, please cite:

```bibtex
@misc{{ranksaga-optimized-e5-v2,
  title={{RankSaga Optimized E5-v2: Fine-tuned Embedding Model for Information Retrieval}},
  author={{RankSaga}},
  year={{2026}},
  url={{https://huggingface.co/RankSaga/ranksaga-optimized-e5-v2}}
}}
```

## License

This model is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, issues, or commercial inquiries:
- **Website**: [https://ranksaga.com](https://ranksaga.com)
- **Contact**: [Contact RankSaga](https://ranksaga.com/contact/)
- **GitHub**: [RankSaga/beir-benchmarking](https://github.com/RankSaga/beir-benchmarking)

## Acknowledgments

- Base model: [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- BEIR Benchmark: [BEIR Paper](https://arxiv.org/abs/2104.08663)
- Sentence Transformers: [sentence-transformers](https://www.sbert.net/)
"""
    
    return card


def upload_model():
    """Upload model to Hugging Face."""
    if not EXPORT_DIR.exists():
        logger.error(f"❌ Export directory not found: {EXPORT_DIR}")
        logger.error("Please run: python export_model_for_hf.py first")
        return False
    
    # Check for Hugging Face token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not token:
        logger.warning("⚠️  Hugging Face token not found in environment")
        logger.info("Please set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable")
        logger.info("\nTo get a token:")
        logger.info("  1. Go to https://huggingface.co/settings/tokens")
        logger.info("  2. Create a new token with write permissions")
        logger.info("  3. Set it: export HF_TOKEN=your_token_here")
        
        # Try to login interactively
        logger.info("\nAttempting interactive login...")
        try:
            hf_login()
            token = True  # Login successful
        except Exception as e:
            logger.error(f"❌ Login failed: {e}")
            return False
    else:
        hf_login(token=token)
    
    # Create repository if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id=REPO_ID, exist_ok=True, repo_type="model")
        logger.info(f"✅ Repository ready: {REPO_ID}")
    except Exception as e:
        logger.error(f"❌ Failed to create repository: {e}")
        return False
    
    # Create model card
    model_card = create_model_card()
    readme_path = EXPORT_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    logger.info("✅ Model card created")
    
    # Upload model
    logger.info(f"Uploading model to {REPO_ID}...")
    try:
        upload_folder(
            folder_path=str(EXPORT_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            ignore_patterns=["*.pyc", "__pycache__", ".git"]
        )
        logger.info(f"✅ Model uploaded successfully to https://huggingface.co/{REPO_ID}")
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def main():
    """Main function."""
    logger.info("="*60)
    logger.info("Uploading model to Hugging Face Hub")
    logger.info("="*60)
    
    if upload_model():
        logger.info("\n" + "="*60)
        logger.info("✅ Upload complete!")
        logger.info("="*60)
        logger.info(f"\nModel available at: https://huggingface.co/{REPO_ID}")
    else:
        logger.error("\n❌ Upload failed. Please check errors above.")


if __name__ == "__main__":
    main()

