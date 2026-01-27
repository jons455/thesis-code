SNN Training and Models
=======================

This folder contains the spiking neural network (SNN) controllers used to
imitate a PI controller for PMSM current control. The goal is to map
normalized state to normalized voltage in a closed-loop benchmark.

What the model learns
---------------------
The SNN learns to approximate the PI control law:
- Input: normalized state [i_d, i_q, e_d, e_q]
- Output: normalized voltage [u_d, u_q]

The challenge is the integral term. The model must keep temporal memory
so small, persistent error accumulates into a steady voltage output.

Implemented model variants
--------------------------
1. MembraneSNNController
   - Output is membrane potential (slow-leak LIF output).
   - High precision but harder to deploy on spike-only hardware.

2. PopulationSNNController
   - Fixed tuning curves, population spikes decoded to a value.
   - Robust and Akida friendly.

3. LearnedLinearSNNController
   - Population spikes decoded by a learned linear layer.
   - Often lower MSE with a simpler decoder.

4. DeltaSNNController
   - Up/down spikes per axis, integrated into voltage.
   - Matches integral control behavior, Akida friendly.
   - Use num_inference_steps=1 or scale delta_scale accordingly.

5. TTFSSNNController
   - Time-to-first-spike coding at the output layer.
   - Single spike per output neuron per control cycle.
   - Voltage decoded from spike latency within a short window.

6. Direct PWM (concept only)
   - Spike trains map directly to inverter duty cycles.
   - Not implemented (requires different training).

Training data requirements
--------------------------
Data must contain PI trajectories with absolute voltage targets:
- Columns (preferred): i_d, i_q, u_d, u_q, i_d_ref, i_q_ref
- Alternates supported: i_sd, i_sq, u_sd, u_sq, i_sd_ref, i_sq_ref

The dataset loader normalizes by motor limits (i_max, u_max) and produces:
- Inputs: [i_d, i_q, e_d, e_q] normalized
- Targets: [u_d, u_q] normalized

Delta coding uses the same target data (absolute voltage). During training,
the voltage state is integrated internally to match those targets.

Generate training data
----------------------
Use the PI controller to generate clean trajectories:

  python scripts/generate_training_data.py --num-files 1000 --validate

Default output directory: data/raw/train/

Validate training data
----------------------

  python scripts/validate_data.py data/raw/train

Training commands
-----------------
Membrane:
  python -m evaluation.snn.train --model_type membrane

Population:
  python -m evaluation.snn.train --model_type population --neurons_per_output 50

Learned linear:
  python -m evaluation.snn.train --model_type learned_linear --neurons_per_output 50

Delta:
  python -m evaluation.snn.train --model_type delta --delta_scale 0.01 --delta_beta 0.8

TTFS:
  python -m evaluation.snn.train --model_type ttfs --ttfs_time_window 20 --ttfs_beta_output 0.9 --ttfs_learn_beta
