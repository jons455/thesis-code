# SNN Model Comparison Report

**Date:** 2026-01-28 23:08

**Best Overall SNN:** SNN_recurrent

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |
|---|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0470 | 0.0 | 100.0% |
| SNN_recurrent | 3.4107 | 1.0129 | 2220.2 | 84.9% |
| SNN_population | 20.9080 | 5.8647 | 3658.4 | 56.1% |
| SNN_learned_linear | 21.0658 | 7.4959 | 634.9 | 71.3% |
| SNN_ttfs | 30.0896 | 7.9150 | 6356.2 | 52.1% |
| SNN_membrane | 167.9872 | 50.1408 | 2166.5 | 61.9% |

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

             Model      RMSE     ITAE       TV     SyOps  Sparsity       LAC
       PI_Baseline  0.100583 0.040045 1.570332    0.0000  1.000000  0.000000
     SNN_recurrent  0.736837 0.279132 0.208427 1956.3112  0.865900  2.425252
    SNN_population  6.355591 3.176431 0.164135 3766.5132  0.557514 22.727209
          SNN_ttfs  8.358674 4.180951 0.004351 6185.0070  0.524026 31.690576
SNN_learned_linear  8.859020 4.443684 0.046781  670.0736  0.677097 25.036675
      SNN_membrane 23.832335 9.993072 0.318118 2115.2252  0.631770 79.251014

