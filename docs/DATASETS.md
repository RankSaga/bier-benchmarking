# BEIR Datasets Documentation

This document describes the BEIR datasets used in RankSaga's benchmarking study.

## BEIR Benchmark Overview

BEIR (Benchmarking IR) is a comprehensive benchmark suite for evaluating information retrieval models. It was introduced in the paper "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models" (Thakur et al., 2021).

**Key features**:
- Diverse domains and tasks
- Standardized evaluation protocols
- Zero-shot evaluation (no task-specific training)
- Reproducible results

**Paper**: [arXiv:2104.08663](https://arxiv.org/abs/2104.08663)

## Datasets in Our Study

We selected four datasets representing different domains and challenges:

### 1. SciFact

**Domain**: Scientific fact-checking

**Size**:
- Queries: 300
- Documents: ~5,000

**Task**: Given a scientific claim, retrieve supporting or refuting evidence from scientific literature.

**Characteristics**:
- High precision requirements
- Scientific terminology and concepts
- Citation-based evidence

**Use Cases**:
- Academic research verification
- Scientific literature search
- Evidence-based fact-checking systems

**Challenges**:
- Technical vocabulary
- Precise matching requirements
- Long-form scientific documents

### 2. NFE Corpus (NFCorpus)

**Domain**: Medical information retrieval

**Size**:
- Queries: 323
- Documents: ~3,600

**Task**: Retrieve relevant medical information for clinical queries.

**Characteristics**:
- Medical terminology
- Domain-specific knowledge required
- Clinical relevance important

**Use Cases**:
- Healthcare information systems
- Clinical decision support
- Medical literature search
- Patient information retrieval

**Challenges**:
- Specialized medical vocabulary
- Technical concepts
- Need for domain expertise

### 3. SciDocs

**Domain**: Scientific document retrieval

**Size**:
- Queries: 1,000
- Documents: ~25,000

**Task**: Retrieve relevant scientific documents for research queries.

**Characteristics**:
- Large document collection
- Scientific publications
- Abstract and title matching

**Use Cases**:
- Academic search engines
- Research paper discovery
- Literature review tools
- Scientific knowledge bases

**Challenges**:
- Large-scale retrieval
- Abstract concepts
- Citation relationships
- Technical terminology

### 4. Quora

**Domain**: General knowledge, duplicate question detection

**Size**:
- Queries: 10,000
- Documents: ~523,000

**Task**: Identify duplicate or similar questions from Quora.

**Characteristics**:
- General domain knowledge
- Semantic similarity detection
- Large-scale dataset
- Natural language queries

**Use Cases**:
- Question-answering systems
- Community forums
- FAQ matching
- Content deduplication

**Challenges**:
- Semantic equivalence (different wording, same meaning)
- Large scale
- General language understanding
- Paraphrase detection

## Dataset Statistics

| Dataset | Queries | Documents | Domain | Size |
|---------|---------|-----------|--------|------|
| SciFact | 300 | ~5,000 | Scientific | Small |
| NFE Corpus | 323 | ~3,600 | Medical | Small |
| SciDocs | 1,000 | ~25,000 | Scientific | Medium |
| Quora | 10,000 | ~523,000 | General | Large |

## Downloading Datasets

BEIR datasets are automatically downloaded when running our benchmarking scripts. They are downloaded from:

```
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip
```

### Manual Download

If you need to download datasets manually:

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
data_path = util.download_and_unzip(url, "datasets/")

# Load dataset
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
```

## Evaluation Protocol

All datasets follow BEIR's standardized evaluation protocol:

1. **Zero-shot evaluation**: No task-specific training
2. **Test split usage**: Evaluate on held-out test data
3. **Standard metrics**: NDCG@10, NDCG@100, MAP@100, Recall@100
4. **Reproducible**: Same evaluation code across all datasets

## Dataset Characteristics

### Technical vs General

- **Technical**: SciFact, NFE Corpus, SciDocs
  - Require domain knowledge
  - Technical terminology
  - Specialized content

- **General**: Quora
  - General language
  - Common knowledge
  - Natural queries

### Size Variations

- **Small** (< 10K docs): SciFact, NFE Corpus
  - Faster evaluation
  - Easier to analyze

- **Medium** (10K-100K docs): SciDocs
  - Moderate scale
  - Realistic workload

- **Large** (> 100K docs): Quora
  - Large-scale challenges
  - Production-like scenarios

## Performance Considerations

Different datasets present different optimization opportunities:

- **Domain-specific datasets** (medical, scientific) show larger improvements with specialized training
- **General datasets** (Quora) may already perform well with base models
- **Scale matters**: Larger datasets provide more training data but require more computation

## Citation

If you use these datasets, please cite the BEIR paper:

```bibtex
@inproceedings{thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas Rücklé and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
```

## Additional Resources

- **BEIR GitHub**: [beir-datasets](https://github.com/beir-cellar/beir)
- **BEIR Paper**: [arXiv](https://arxiv.org/abs/2104.08663)
- **Dataset Details**: See individual dataset papers for more information

