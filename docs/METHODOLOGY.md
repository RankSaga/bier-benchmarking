# Methodology: RankSaga BEIR Benchmarking

This document provides a detailed explanation of the methodology used in RankSaga's BEIR benchmarking study.

## Overview

Our benchmarking approach evaluates embedding models on the BEIR (Benchmarking IR) benchmark suite, which provides standardized datasets and evaluation protocols for information retrieval tasks. We compare baseline pre-trained models against RankSaga-optimized fine-tuned models.

## Base Model

We selected `intfloat/e5-base-v2` as our base model for the following reasons:

- **Strong baseline performance**: E5 models achieve state-of-the-art results on retrieval tasks
- **E5 architecture**: Designed specifically for retrieval tasks with bidirectional encoder representations
- **Model size**: Base variant provides good balance between performance and efficiency (~110M parameters)
- **Wide adoption**: Commonly used in research and production systems

## Fine-Tuning Approach

### Loss Function: Multiple Negatives Ranking Loss

We use Multiple Negatives Ranking Loss (MNR), which is specifically designed for retrieval optimization:

**Key advantages**:
- Automatically uses in-batch negatives, eliminating need for explicit negative sampling
- Optimizes for ranking quality rather than absolute similarity scores
- Proven effective in retrieval research literature
- Efficient training with large batch sizes

**How it works**:
1. Given query-document pairs in a batch
2. For each query, all other documents in the batch serve as negatives
3. Optimizes query to retrieve its corresponding document over all negatives
4. Learns ranking relationships rather than absolute similarity

### Training Data Strategy

We employ a comprehensive data strategy:

#### Multi-Dataset Training

Training on multiple BEIR datasets provides:
- **Diversity**: Different domains and retrieval patterns
- **Generalization**: Better performance across varied use cases
- **Robustness**: Less prone to overfitting to a single dataset

#### All Available Splits

We use all available data splits:
- **Train split**: Primary training data
- **Dev split**: When train not available, use dev
- **Test split**: When train/dev not available, use test

This maximizes training data while respecting dataset availability.

#### Dataset Selection

We selected four diverse datasets:

1. **SciFact**: Scientific fact-checking (300 queries, 5K documents)
2. **NFE Corpus**: Medical information retrieval (323 queries, 3.6K documents)
3. **SciDocs**: Scientific document retrieval (1K queries, 25K documents)
4. **Quora**: Duplicate question detection (10K queries, 523K documents)

These datasets represent different domains, sizes, and retrieval challenges.

### Training Configuration

#### Hyperparameters

- **Epochs**: 5
  - Provides sufficient learning without overfitting
  - Validated through experiments on validation data

- **Batch Size**: 32
  - Balances GPU memory usage and training efficiency
  - Larger batches provide more in-batch negatives for MNR loss

- **Learning Rate**: 1e-5
  - Conservative rate for stable fine-tuning
  - Prevents destructive updates to pre-trained weights

- **Warmup Steps**: 500
  - Gradual learning rate increase prevents early instability
  - Allows model to adapt smoothly to fine-tuning

- **Mixed Precision**: FP16
  - Reduces memory usage and speeds training
  - Minimal impact on final model quality

#### Training Process

1. **Data Preparation**: Load and combine all datasets
2. **Model Loading**: Initialize from pre-trained e5-base-v2
3. **Training**: Fine-tune with MNR loss for 5 epochs
4. **Checkpointing**: Save checkpoints after each epoch
5. **Validation**: Monitor performance on validation sets

### Domain-Specific Models

We also trained specialized models:

#### Scientific Domain Model

**Training Data**: Scifact, nfcorpus, scidocs

**Purpose**: Optimized for scientific and medical information retrieval

**Expected Benefits**: Better performance on technical, domain-specific content

#### General Domain Model

**Training Data**: Quora

**Purpose**: Optimized for general semantic similarity tasks

**Expected Benefits**: Better performance on general knowledge and question-answering

## Evaluation Methodology

### Evaluation Datasets

We evaluate on the same datasets used for training (standard BEIR evaluation protocol):

- SciFact
- NFE Corpus
- SciDocs
- Quora

### Evaluation Metrics

We report four key metrics:

1. **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10
   - Primary metric for ranking quality
   - Weights top results more heavily

2. **NDCG@100**: NDCG at rank 100
   - Evaluates deeper ranking quality
   - Important for systems returning many results

3. **MAP@100**: Mean Average Precision at rank 100
   - Average precision across all queries
   - Provides comprehensive accuracy measure

4. **Recall@100**: Recall at rank 100
   - Proportion of relevant documents found
   - Measures coverage and completeness

### Evaluation Process

1. **Model Loading**: Load fine-tuned model
2. **Dataset Loading**: Load test splits of BEIR datasets
3. **Embedding Generation**: Generate embeddings for queries and documents
4. **Retrieval**: Compute similarity and retrieve top-k documents
5. **Evaluation**: Calculate metrics using BEIR evaluation framework
6. **Results Storage**: Save results in JSON format

### Hardware and Infrastructure

- **Training**: Modal.com with A10G GPU (24GB VRAM), 32GB RAM
- **Evaluation**: Same infrastructure for consistency
- **Training Time**: ~3-5 hours for full fine-tuning
- **Evaluation Time**: ~15-30 minutes per model

## Limitations and Considerations

### Overfitting

Multi-dataset training can lead to:
- Domain conflicts when datasets have competing objectives
- Overfitting to certain patterns that don't generalize
- Performance regressions on some datasets

We address this by:
- Careful hyperparameter tuning
- Monitoring validation performance
- Creating domain-specific models

### Evaluation on Training Data

Following BEIR protocol, we evaluate on test splits. However, some test data may overlap with training data in multi-dataset scenarios. This is a known limitation of the BEIR evaluation framework.

### Generalization

Results on BEIR datasets may not perfectly generalize to:
- Different domains not in training data
- Different retrieval tasks
- Production systems with different data distributions

## Reproducibility

To reproduce our results:

1. **Install dependencies**: See `requirements.txt`
2. **Download datasets**: BEIR datasets download automatically
3. **Run fine-tuning**: Execute `modal_fine_tune_advanced.py`
4. **Run evaluation**: Execute benchmarking scripts
5. **Compare results**: Use `compare_results.py`

All code, configurations, and results are available in this repository.

## References

- BEIR Benchmark: [Paper](https://arxiv.org/abs/2104.08663)
- E5 Model: [Paper](https://arxiv.org/abs/2212.03533)
- Sentence Transformers: [Documentation](https://www.sbert.net/)
- Multiple Negatives Ranking: Research literature on retrieval optimization

