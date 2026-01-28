# SNN Model Comparison Report

**Date:** 2026-01-28 22:05

**Best Overall SNN:** SNN_ttfs

## 1. Leaderboard (Average across 3 Scenarios)

| Model | Avg LAC | Avg RMSE [A] | Avg SyOps/step |
|---|---|---|---|
| PI_Baseline | 0.0000 | 0.0467 | 0.0 |
| SNN_ttfs | 0.0000 | 7.9144 | 0.0 |

## 2. Detailed Results by Scenario

### Scenario: A_Nominal (Baseline (1000 rpm, 2A))

      Model     RMSE     ITAE       TV  SyOps  LAC
PI_Baseline 0.014488 0.000002 0.004965    0.0  0.0
   SNN_ttfs 8.352010 4.180903 0.006020    0.0  0.0

### Scenario: B_HighSpeed (High Speed (3000 rpm))

      Model     RMSE     ITAE       TV  SyOps  LAC
PI_Baseline 0.025938 0.000005 0.003655    0.0  0.0
   SNN_ttfs 7.034186 3.514326 0.004983    0.0  0.0

