# RankSaga Model Optimization Benchmark Results

## Summary

- **Average Improvement**: -2.17%
- **Maximum Improvement**: 15.25%
- **Minimum Improvement**: -26.28%
- **Datasets Improved**: 2/4

## Detailed Results

### NDCG@10 Comparison

| Dataset | Baseline | Optimized | Improvement % |
|---------|----------|-----------|---------------|
| scifact | 0.6967 | 0.5137 | -26.28% |
| nfcorpus | 0.3402 | 0.3921 | 15.25% |
| scidocs | 0.1713 | 0.1767 | 3.14% |
| quora | 0.8541 | 0.8472 | -0.81% |

### All Metrics Comparison

| Dataset   | Metric     |   Baseline |   Optimized |   Improvement_% |   Absolute_Improvement |
|:----------|:-----------|-----------:|------------:|----------------:|-----------------------:|
| scifact   | NDCG@10    |    0.69675 |     0.51366 |      -26.2777   |               -0.18309 |
| scifact   | NDCG@100   |    0.72074 |     0.5563  |      -22.8154   |               -0.16444 |
| scifact   | MAP@100    |    0.65238 |     0.46575 |      -28.6076   |               -0.18663 |
| scifact   | Recall@100 |    0.941   |     0.86844 |       -7.71095  |               -0.07256 |
| nfcorpus  | NDCG@10    |    0.34021 |     0.3921  |       15.2523   |                0.05189 |
| nfcorpus  | NDCG@100   |    0.31569 |     0.41868 |       32.6238   |                0.10299 |
| nfcorpus  | MAP@100    |    0.15875 |     0.23732 |       49.4929   |                0.07857 |
| nfcorpus  | Recall@100 |    0.31982 |     0.48301 |       51.0256   |                0.16319 |
| scidocs   | NDCG@10    |    0.17134 |     0.17672 |        3.13996  |                0.00538 |
| scidocs   | NDCG@100   |    0.24381 |     0.27263 |       11.8207   |                0.02882 |
| scidocs   | MAP@100    |    0.11564 |     0.12455 |        7.70495  |                0.00891 |
| scidocs   | Recall@100 |    0.39783 |     0.47823 |       20.2096   |                0.0804  |
| quora     | NDCG@10    |    0.85412 |     0.84719 |       -0.811361 |               -0.00693 |
| quora     | NDCG@100   |    0.86995 |     0.86306 |       -0.792    |               -0.00689 |
| quora     | MAP@100    |    0.81943 |     0.81206 |       -0.899406 |               -0.00737 |
| quora     | Recall@100 |    0.99033 |     0.98648 |       -0.388759 |               -0.00385 |