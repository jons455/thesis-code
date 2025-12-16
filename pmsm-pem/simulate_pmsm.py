"""
PMSM FOC Simulation mit gym-electric-motor (GEM)
Generiert Trainingsdaten für neuronale Netze
"""

import numpy as np
import pandas as pd
import os
import gym_electric_motor as gem
import gem_controllers as gc


# Motorparameter (aus MATLAB/Simulink)
motor_parameter = dict(
    p=3,
    r_s=0.543,
    l_d=0.00113,
    l_q=0.00142,
    psi_p=0.0169,
)

limit_values = dict(
    i=10.8,
    u=48.0,
    omega=3000 * 2 * np.pi / 60
)

# Simulationsparameter
tau = 1/10000
sim_steps = 2000
number_of_simulations = 1

out_dir = os.path.join(os.getcwd(), 'export', 'gem_standard')
os.makedirs(out_dir, exist_ok=True)

# Environment erstellen
env_id = 'Cont-CC-PMSM-v0'
env = gem.make(
    env_id,
    motor_parameter=motor_parameter,
    limit_values=limit_values,
    tau=tau,
    visualization=None,
)

env_unwrapped = env
while hasattr(env_unwrapped, 'env'):
    env_unwrapped = env_unwrapped.env

# GEM Controller
controller = gc.GemController.make(
    env_unwrapped, env_id,
    decoupling=True,
    current_safety_margin=0.2,
    base_current_controller="PI",
    a=4,
    block_diagram=False,
)
print("GEM Controller initialisiert")


def get_state_indices(env, state_vec):
    try:
        if hasattr(env, 'state_names'):
            return (env.state_names.index('i_sd'), env.state_names.index('i_sq'),
                    env.state_names.index('omega'), env.state_names.index('epsilon'))
    except (AttributeError, ValueError):
        pass
    n = len(state_vec)
    if n > 6:
        return 2, 3, 0, 6
    elif n >= 4:
        return 2, 3, 0, 0
    return 0, 1, 0, 0


def extract_state(state):
    if isinstance(state, tuple):
        state = state[0]
    if isinstance(state, tuple):
        state = state[0]
    return state


# Simulation
print("Starte Datengenerierung...")

for i in range(number_of_simulations):
    # Feste Test-Werte (gleich wie MATLAB!)
    id_ref = 0.0    # [A] - d-Achsen Sollstrom
    iq_ref = 2.0    # [A] - q-Achsen Sollstrom (Drehmoment)
    
    state = extract_state(env.reset())
    
    if hasattr(controller, 'reset'):
        controller.reset()
    
    data_log = {'time': [], 'i_d': [], 'i_q': [], 'n': [], 'u_d': [], 'u_q': [], 'i_d_ref': [], 'i_q_ref': []}
    
    for k in range(sim_steps):
        state_vec = extract_state(state)
        idx_isd, idx_isq, idx_omega, idx_epsilon = get_state_indices(env, state_vec)
        epsilon = state_vec[idx_epsilon] * np.pi
        
        # Step bei t=0.1s (k=1000 bei tau=0.0001) - wie in MATLAB
        step_time_k = 1000  # 0.1s / 0.0001s = 1000
        if k < step_time_k:
            id_ref_active = 0.0
            iq_ref_active = 0.0
        else:
            id_ref_active = id_ref
            iq_ref_active = iq_ref
        
        ref_normalized = np.array([id_ref_active / limit_values['i'], iq_ref_active / limit_values['i']])
        action_abc = controller.control(state_vec, ref_normalized)
        
        # abc -> dq Rücktransformation für Logging
        u_a, u_b, u_c = action_abc[0], action_abc[1], action_abc[2]
        u_alpha = (2/3) * (u_a - 0.5*u_b - 0.5*u_c)
        u_beta = (2/3) * (np.sqrt(3)/2 * u_b - np.sqrt(3)/2 * u_c)
        c, s = np.cos(epsilon), np.sin(epsilon)
        u_d_norm = u_alpha * c + u_beta * s
        u_q_norm = -u_alpha * s + u_beta * c
        
        u_d_val = u_d_norm * limit_values['u']
        u_q_val = u_q_norm * limit_values['u']
        
        # Environment-Schritt
        step_result = env.step(action_abc)
        if len(step_result) == 5:
            state, reward, done, truncated, info = step_result
        else:
            (state, _), reward, done, _, _ = step_result
        
        # Bei Termination Environment resetten aber Simulation fortsetzen
        if done:
            state = extract_state(env.reset())
            if hasattr(controller, 'reset'):
                controller.reset()
        
        state_vec = extract_state(state)
        idx_isd, idx_isq, idx_omega, _ = get_state_indices(env, state_vec)
        
        i_d_val = state_vec[idx_isd] * limit_values['i']
        i_q_val = state_vec[idx_isq] * limit_values['i']
        omega_mech = state_vec[idx_omega] * limit_values['omega']
        n_val = omega_mech * 60 / (2 * np.pi)
        
        data_log['time'].append(k * tau)
        data_log['i_d'].append(i_d_val)
        data_log['i_q'].append(i_q_val)
        data_log['n'].append(n_val)
        data_log['u_d'].append(u_d_val)
        data_log['u_q'].append(u_q_val)
        data_log['i_d_ref'].append(id_ref_active)
        data_log['i_q_ref'].append(iq_ref_active)
        
        # done-Flag ignorieren um volle Simulationsdauer zu erreichen
    
    df = pd.DataFrame(data_log)
    filename = os.path.join(out_dir, f'sim_{i+1:04d}.csv')
    df.to_csv(filename, index=False)
    print(f"-> Simulation {i+1} exportiert: {filename}")

print("Export abgeschlossen.")
