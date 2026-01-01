# Domain-Specific Model Benchmark Results

## Summary

This report shows the performance of domain-specific fine-tuned models.

## Scientific Domain Model

**Model**: ranksaga-optimized-e5-v2-scientific

**Datasets**: scifact, nfcorpus, scidocs

### Performance Metrics

| Dataset | NDCG@10 | NDCG@100 | MAP@100 | Recall@100 |
|---------|---------|----------|---------|------------|
| scifact | 0.5324 | 0.5703 | 0.4869 | 0.8574 |
  - Baseline (approx): NDCG@10 = 0.6650
  - Improvement: -19.94%

| nfcorpus | 0.3941 | 0.4177 | 0.2385 | 0.4784 |
  - Baseline (approx): NDCG@10 = 0.3250
  - Improvement: +21.27%

| scidocs | 0.1732 | 0.2688 | 0.1230 | 0.4725 |
  - Baseline (approx): NDCG@10 = 0.1580
  - Improvement: +9.61%


## General Domain Model

**Model**: ranksaga-optimized-e5-v2-general

**Datasets**: quora

### Performance Metrics

| Dataset | NDCG@10 | NDCG@100 | MAP@100 | Recall@100 |
|---------|---------|----------|---------|------------|
| quora | 0.8826 | 0.8953 | 0.8526 | 0.9956 |
  - Baseline (approx): NDCG@10 = 0.7890
  - Improvement: +11.87%


## Overall Summary

- **Average Improvement (NDCG@10)**: +5.70%
- **Datasets Evaluated**: 4

> **Note**: Baseline values are approximate. Final comparison will be available once baseline benchmark completes.

