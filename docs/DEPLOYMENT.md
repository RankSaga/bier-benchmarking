# Deployment Guide

This guide covers deploying and using RankSaga's optimized embedding models in production environments.

## Using the Model from Hugging Face

### Installation

```bash
pip install sentence-transformers
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")
embeddings = model.encode("Your text here")
```

### Production Considerations

#### GPU Acceleration

For production deployments with high throughput:

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2", device=device)
```

#### Batch Processing

For efficient batch processing:

```python
sentences = ["Text 1", "Text 2", "Text 3", ...]
embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)
```

#### Caching

For frequently used queries, consider caching:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding_cached(text):
    return model.encode(text)[0]

# Use cached embeddings for repeated queries
```

## Local Model Deployment

If you've fine-tuned a model locally or downloaded from Modal:

```python
from sentence_transformers import SentenceTransformer

# Load from local path
model = SentenceTransformer("./models/ranksaga-optimized-e5-v2")
```

## Using with Vector Databases

### Pinecone

```python
import pinecone
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")
pinecone.init(api_key="your-api-key", environment="your-environment")

# Create index
index = pinecone.Index("your-index-name")

# Encode and upsert
documents = ["Doc 1", "Doc 2", ...]
embeddings = model.encode(documents)

for i, embedding in enumerate(embeddings):
    index.upsert([(f"doc_{i}", embedding.tolist())])
```

### Weaviate

```python
import weaviate
from sentence_transformers import SentenceTransformer

client = weaviate.Client("http://localhost:8080")
model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Encode and add to Weaviate
documents = ["Doc 1", "Doc 2", ...]
embeddings = model.encode(documents)

for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    client.data_object.create(
        data_object={"text": doc},
        class_name="Document",
        vector=emb.tolist()
    )
```

### Qdrant

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Encode and add to Qdrant
documents = ["Doc 1", "Doc 2", ...]
embeddings = model.encode(documents)

points = [
    {"id": i, "vector": emb.tolist(), "payload": {"text": doc}}
    for i, (doc, emb) in enumerate(zip(documents, embeddings))
]

client.upsert(collection_name="documents", points=points)
```

## Docker Deployment

### Dockerfile Example

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir sentence-transformers torch

# Download model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('RankSaga/ranksaga-optimized-e5-v2')"

# Copy application code
COPY app.py .

# Run application
CMD ["python", "app.py"]
```

### FastAPI Service Example

```python
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

@app.post("/encode")
async def encode_text(text: str):
    embedding = model.encode(text)
    return {"embedding": embedding.tolist()}

@app.post("/encode_batch")
async def encode_batch(texts: list[str]):
    embeddings = model.encode(texts)
    return {"embeddings": [emb.tolist() for emb in embeddings]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Performance Optimization

### Model Quantization

For faster inference with minimal quality loss:

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Quantize model (PyTorch 2.0+)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### ONNX Export

For optimized inference:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

# Export to ONNX (requires onnxruntime)
model.save("path/to/onnx/model", format="onnx")
```

## Monitoring and Observability

### Logging

```python
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")
logger.info("Model loaded successfully")
```

### Performance Metrics

```python
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("RankSaga/ranksaga-optimized-e5-v2")

def encode_with_timing(text):
    start = time.time()
    embedding = model.encode(text)
    elapsed = time.time() - start
    return embedding, elapsed

embedding, elapsed = encode_with_timing("Your text")
print(f"Encoding took {elapsed:.4f} seconds")
```

## Scaling Considerations

### Horizontal Scaling

For high-throughput scenarios:
- Use load balancer (nginx, AWS ALB)
- Deploy multiple model instances
- Consider model serving frameworks (TensorFlow Serving, TorchServe, Triton)

### Caching Strategy

- Cache frequently accessed embeddings
- Use Redis for distributed caching
- Implement TTL-based cache invalidation

### Resource Requirements

**Minimum**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 2GB (for model)

**Recommended**:
- CPU: 4+ cores or GPU
- RAM: 8GB+
- Storage: 5GB+

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use CPU instead of GPU
- Enable gradient checkpointing

### Slow Inference

- Use GPU if available
- Increase batch size
- Consider model quantization
- Use ONNX runtime

### Model Loading Errors

- Check internet connection (for Hugging Face download)
- Verify model name spelling
- Ensure sufficient disk space

## Best Practices

1. **Warm-up**: Load model before handling requests
2. **Batching**: Process multiple texts together when possible
3. **Error Handling**: Implement proper error handling and fallbacks
4. **Monitoring**: Track latency, throughput, and error rates
5. **Versioning**: Pin model versions for reproducibility
6. **Security**: Validate input text, prevent injection attacks

## Support

For deployment questions or issues:
- GitHub Issues: [RankSaga/beir-benchmarking](https://github.com/RankSaga/beir-benchmarking)
- Contact: [RankSaga Support](https://ranksaga.com/contact/)

