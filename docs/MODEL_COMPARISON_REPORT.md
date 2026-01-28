# SNN Model Comparison Report

**Date:** 2026-01-28 22:39

**Best Overall SNN:** SNN_recurrent

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |
|---|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0145 | 0.0 | 100.0% |
| SNN_recurrent | 2.4495 | 0.7442 | 1956.2 | 86.6% |
| SNN_population | 22.7381 | 6.3586 | 3766.8 | 55.7% |
| SNN_learned_linear | 24.9077 | 8.7931 | 680.2 | 67.5% |
| SNN_ttfs | 31.6662 | 8.3520 | 6186.5 | 52.4% |
| SNN_membrane | 68.5429 | 20.6548 | 2082.1 | 63.6% |

## 2. Detailed Results by Scenario

### Scenario: A_Nominal (Baseline (1000 rpm, 2A))

             Model      RMSE     ITAE       TV     SyOps  Sparsity       LAC
       PI_Baseline  0.014488 0.000002 0.004965    0.0000  1.000000  0.000000
     SNN_recurrent  0.744196 0.283332 0.210722 1956.2068  0.865910  2.449457
    SNN_population  6.358576 3.179057 0.165828 3766.8192  0.557490 22.738109
          SNN_ttfs  8.352010 4.180903 0.006020 6186.5348  0.524034 31.666207
SNN_learned_linear  8.793145 4.410272 0.034762  680.1856  0.674586 24.907704
      SNN_membrane 20.654765 9.059677 0.319786 2082.1056  0.635884 68.542896

