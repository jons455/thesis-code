# SNN Model Comparison Report

**Date:** 2026-01-29 21:04

**Best Overall SNN:** SNN_learned_linear

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step | Avg Sparsity |
|---|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0145 | 0.0 | 100.0% |
| SNN_learned_linear | 2.6271 | 0.9819 | 473.8 | 77.4% |
| SNN_delta | 17.7954 | 6.8117 | 409.7 | 95.6% |
| SNN_ttfs | 31.6662 | 8.3520 | 6186.5 | 52.4% |
| SNN_recurrent | 45.3386 | 12.3159 | 4800.6 | 66.9% |
| SNN_population | 45.4460 | 12.3327 | 4841.6 | 47.2% |
| SNN_membrane | 454.4331 | 136.3435 | 2152.8 | 63.6% |

## 2. Detailed Results by Scenario

### Scenario: A_Nominal (Baseline (1000 rpm, 2A))

             Model       RMSE      ITAE       TV     SyOps  Sparsity        LAC
       PI_Baseline   0.014488  0.000002 0.004965    0.0000  1.000000   0.000000
SNN_learned_linear   0.981880  0.490845 0.233529  473.7856  0.774210   2.627101
         SNN_delta   6.811707  3.296619 0.000048  409.7024  0.955647  17.795370
          SNN_ttfs   8.352010  4.180903 0.006020 6186.5348  0.524034  31.666207
     SNN_recurrent  12.315949  6.033404 0.123953 4800.5952  0.668548  45.338644
    SNN_population  12.332720  6.173120 1.276528 4841.6392  0.471731  45.445979
      SNN_membrane 136.343547 69.310489 0.280615 2152.7846  0.636348 454.433121

