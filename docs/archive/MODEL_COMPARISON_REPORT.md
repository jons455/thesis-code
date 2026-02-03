# SNN Model Comparison Report

**Date:** 2026-01-29 14:39

**Best Overall SNN:** SNN_recurrent

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |
|---|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0462 | 0.0 | 100.0% |
| SNN_recurrent | 3.3283 | 0.9879 | 2220.5 | 84.9% |
| SNN_population | 20.9068 | 5.8643 | 3658.4 | 56.1% |
| SNN_learned_linear | 21.0655 | 7.4958 | 634.9 | 71.3% |
| SNN_ttfs | 30.0891 | 7.9148 | 6356.4 | 52.1% |
| SNN_membrane | 173.8289 | 51.8582 | 2188.8 | 61.6% |

## 2. Detailed Results by Scenario

### Scenario: A_Nominal (Baseline (1000 rpm, 2A))

             Model      RMSE     ITAE       TV     SyOps  Sparsity       LAC
       PI_Baseline  0.014488 0.000002 0.004965    0.0000  1.000000  0.000000
     SNN_recurrent  0.744196 0.283332 0.210722 1956.2068  0.865910  2.449457
    SNN_population  6.358576 3.179057 0.165828 3766.8192  0.557490 22.738109
          SNN_ttfs  8.352010 4.180903 0.006020 6186.5348  0.524034 31.666207
SNN_learned_linear  8.858740 4.443582 0.044991  669.9584  0.677051 25.035222
      SNN_membrane 20.654765 9.059677 0.319786 2082.1056  0.635884 68.542896

### Scenario: B_HighSpeed (High Speed (3000 rpm))

             Model       RMSE      ITAE       TV     SyOps  Sparsity        LAC
       PI_Baseline   0.025938  0.000005 0.003655    0.0000  1.000000   0.000000
     SNN_recurrent   1.557781  0.752037 0.228175 2748.0458  0.816638   5.357245
SNN_learned_linear   4.769902  2.380883 0.193479  564.5824  0.785295  13.125468
    SNN_population   4.879786  2.437333 0.995512 3441.8064  0.567710  17.258761
          SNN_ttfs   7.034186  3.514326 0.004983 6696.9256  0.515428  26.911918
      SNN_membrane 105.935422 53.899736 0.033830 2302.0774  0.588286 356.167591

### Scenario: C_Robustness (Noisy (σ=0.05A))

             Model      RMSE      ITAE       TV     SyOps  Sparsity       LAC
       PI_Baseline  0.098080  0.038323 1.544460    0.0000  1.000000  0.000000
     SNN_recurrent  0.661773  0.240502 0.208317 1957.1046  0.865840  2.178302
    SNN_population  6.354566  3.175825 0.163124 3766.5756  0.557515 22.723589
          SNN_ttfs  8.358165  4.180953 0.004877 6185.6804  0.524017 31.689044
SNN_learned_linear  8.858771  4.443643 0.047100  670.0224  0.677128 25.035677
      SNN_membrane 28.984289 14.030175 0.316324 2182.3156  0.623437 96.776152
