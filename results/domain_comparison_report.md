# Domain-Specific Model Benchmark Results

## Summary

- **Average Improvement**: 3.57%
- **Maximum Improvement**: 21.27%
- **Minimum Improvement**: -19.94%
- **Datasets Improved**: 3/4

## Domain Models

### Scientific Domain
- **Model**: ranksaga-optimized-e5-v2-scientific
- **Datasets**: scifact, nfcorpus, scidocs

### General Domain
- **Model**: ranksaga-optimized-e5-v2-general
- **Datasets**: quora

## Detailed Results

### NDCG@10 Comparison

| Domain     | Dataset   |   Baseline |   Domain_Model |   Improvement_% |
|:-----------|:----------|-----------:|---------------:|----------------:|
| scientific | scifact   |     0.665  |         0.5324 |        -19.9368 |
| scientific | nfcorpus  |     0.325  |         0.3941 |         21.2708 |
| scientific | scidocs   |     0.158  |         0.1732 |          9.6076 |
| general    | quora     |     0.8541 |         0.8826 |          3.3415 |

### All Metrics Comparison

| Domain     | Dataset   | Metric     |   Baseline |   Domain_Model |   Improvement_% |   Absolute_Improvement |
|:-----------|:----------|:-----------|-----------:|---------------:|----------------:|-----------------------:|
| scientific | scifact   | NDCG@10    |     0.665  |         0.5324 |        -19.9368 |                -0.1326 |
| scientific | scifact   | NDCG@100   |     0.695  |         0.5703 |        -17.9367 |                -0.1247 |
| scientific | scifact   | MAP@100    |     0.62   |         0.4869 |        -21.4726 |                -0.1331 |
| scientific | scifact   | Recall@100 |     0.92   |         0.8574 |         -6.8    |                -0.0626 |
| scientific | nfcorpus  | NDCG@10    |     0.325  |         0.3941 |         21.2708 |                 0.0691 |
| scientific | nfcorpus  | NDCG@100   |     0.35   |         0.4177 |         19.3371 |                 0.0677 |
| scientific | nfcorpus  | MAP@100    |     0.2    |         0.2385 |         19.24   |                 0.0385 |
| scientific | nfcorpus  | Recall@100 |     0.4    |         0.4784 |         19.6    |                 0.0784 |
| scientific | scidocs   | NDCG@10    |     0.158  |         0.1732 |          9.6076 |                 0.0152 |
| scientific | scidocs   | NDCG@100   |     0.24   |         0.2688 |         12.0083 |                 0.0288 |
| scientific | scidocs   | MAP@100    |     0.11   |         0.123  |         11.8    |                 0.013  |
| scientific | scidocs   | Recall@100 |     0.42   |         0.4725 |         12.5    |                 0.0525 |
| general    | quora     | NDCG@10    |     0.8541 |         0.8826 |          3.3415 |                 0.0285 |
| general    | quora     | NDCG@100   |     0.8699 |         0.8953 |          2.9245 |                 0.0254 |
| general    | quora     | MAP@100    |     0.8194 |         0.8526 |          4.0456 |                 0.0332 |
| general    | quora     | Recall@100 |     0.9903 |         0.9956 |          0.5342 |                 0.0053 |

## Visualizations

![Comparison Plot](domain_comparison_visualization.png)
