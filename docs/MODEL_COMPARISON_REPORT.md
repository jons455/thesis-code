# SNN Model Comparison Report

**Date:** 2026-01-30 14:26

**Best Overall SNN:** SNN_learned_linear

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |
|---|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0145 | 0.0 | 100.0% |
| SNN_learned_linear | 3.2504 | 1.1645 | 618.4 | 76.5% |
| SNN_ttfs | 6.0201 | 1.5678 | 6916.2 | 74.0% |
| SNN_population | 14.6645 | 4.2630 | 2753.8 | 62.1% |
| SNN_membrane | 123.9429 | 37.9254 | 1853.8 | 73.1% |
| SNN_delta | 252.3592 | 76.4323 | 2003.3 | 55.3% |
| SNN_recurrent | 750.6934 | 204.4680 | 4693.0 | 66.9% |

## 2. Detailed Results by Scenario

### Scenario: A_Nominal (Baseline (1000 rpm, 2A))

             Model       RMSE       ITAE       TV     SyOps  Sparsity        LAC
       PI_Baseline   0.014488   0.000002 0.004965    0.0000  1.000000   0.000000
SNN_learned_linear   1.164478   0.580838 0.516690  618.3744  0.764837   3.250352
          SNN_ttfs   1.567789   0.659002 7.336170 6916.1566  0.739669   6.020097
    SNN_population   4.263029   2.133323 0.694606 2753.7840  0.620878  14.664521
      SNN_membrane  37.925357  19.201450 0.185622 1853.8472  0.730552 123.942871
         SNN_delta  76.432253  39.379550 0.194937 2003.2576  0.552982 252.359182
     SNN_recurrent 204.468037 103.032658 0.205878 4692.9520  0.669417 750.693378

