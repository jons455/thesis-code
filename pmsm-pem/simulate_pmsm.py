"""
PMSM FOC Simulation mit gym-electric-motor (GEM)
Generiert Trainingsdaten für neuronale Netze
"""

import numpy as np
import pandas as pd
import os
import argparse
import gym_electric_motor as gem
import gem_controllers as gc
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad


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


def parse_args():
    p = argparse.ArgumentParser(description="GEM PMSM current-control simulation (standard GEM controller)")
    p.add_argument("--n-rpm", type=float, default=716.0, help="Fixed mechanical speed in rpm (ConstantSpeedLoad).")
    p.add_argument("--id-ref", type=float, default=0.0, help="d-axis current reference after step [A].")
    p.add_argument("--iq-ref", type=float, default=2.0, help="q-axis current reference after step [A].")
    p.add_argument("--step-time", type=float, default=0.0, help="Step time in seconds (0 = immediate).")
    p.add_argument("--sim-steps", type=int, default=sim_steps, help="Number of simulation steps.")
    p.add_argument("--tau", type=float, default=tau, help="Control cycle time in seconds.")
    p.add_argument("--output", type=str, default=None, help="Output filename (default: sim_0001.csv)")
    return p.parse_args()


args = parse_args()
out_dir = os.path.join(os.getcwd(), 'export', 'gem_standard')
os.makedirs(out_dir, exist_ok=True)

# Environment erstellen
env_id = 'Cont-CC-PMSM-v0'
omega_fixed = args.n_rpm * 2 * np.pi / 60.0

# Motor und Load DIREKT erstellen, damit unsere Parameter auch wirklich verwendet werden!
motor = PermanentMagnetSynchronousMotor(
    motor_parameter=motor_parameter,
    limit_values=limit_values,
)
load = ConstantSpeedLoad(omega_fixed=float(omega_fixed))

env = gem.make(
    env_id,
    motor=motor,
    load=load,
    tau=args.tau,
    visualization=None,
    constraints=(),  # Keine Constraints -> kein done=True wegen Limit-Überschreitung
)

env_unwrapped = env
while hasattr(env_unwrapped, 'env'):
    env_unwrapped = env_unwrapped.env

# Hole die tatsächlichen Limits vom Physical System
ps = env_unwrapped.physical_system
state_names = list(ps.state_names)
actual_limits = {name: ps.limits[i] for i, name in enumerate(state_names)}
print(f"GEM Limits: omega={actual_limits.get('omega', 0)*60/(2*np.pi):.0f} rpm, "
      f"i_sd={actual_limits.get('i_sd', 0):.1f} A, u_sd={actual_limits.get('u_sd', 0):.1f} V")

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


def get_state_indices(env, state):
    env_unwrapped = env
    while hasattr(env_unwrapped, 'env'):
        env_unwrapped = env_unwrapped.env
    names = list(env_unwrapped.physical_system.state_names)
    return (names.index('i_sd'), names.index('i_sq'), 
            names.index('omega'), names.index('epsilon'))


# Simulation
print("Starte Datengenerierung...")

for i in range(number_of_simulations):
    # Referenzwerte
    id_ref = float(args.id_ref)
    iq_ref = float(args.iq_ref)
    
    # Reset Environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
        if isinstance(state, tuple):
            state = state[0]
    else:
        state = reset_result
    
    # Reset Controller
    controller.reset()
    
    # Data Logging
    data_log = {
        'time': [], 'i_d': [], 'i_q': [], 'n': [], 
        'u_d': [], 'u_q': [], 'i_d_ref': [], 'i_q_ref': []
    }
    
    # Step-Zeit (0 = sofort aktiv)
    step_time_k = int(round(args.step_time / args.tau)) if args.step_time > 0 else 0
    
    for k in range(args.sim_steps):
        # Indizes ermitteln
        idx_isd, idx_isq, idx_omega, idx_epsilon = get_state_indices(env, state)
        epsilon = state[idx_epsilon] * np.pi
        
        # Referenz aktivieren
        if k < step_time_k:
            id_ref_active = 0.0
            iq_ref_active = 0.0
        else:
            id_ref_active = id_ref
            iq_ref_active = iq_ref
        
        # Normalisierte Referenz (Controller erwartet normalisiert)
        ref_normalized = np.array([
            id_ref_active / limit_values['i'], 
            iq_ref_active / limit_values['i']
        ])
        
        # Controller berechnet Action
        action_abc = controller.control(state, ref_normalized)
        
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
        obs, reward, done, truncated, info = env.step(action_abc)
        
        # Observation auspacken
        if isinstance(obs, tuple):
            state = obs[0]
            if isinstance(state, tuple):
                state = state[0]
        else:
            state = obs
        
        # Bei Termination: Environment resetten
        if done:
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                state = reset_result[0]
                if isinstance(state, tuple):
                    state = state[0]
            else:
                state = reset_result
            # Controller NICHT resetten um Integrator-Zustand zu behalten
        
        # Zustand für Logging denormalisieren
        i_d_val = state[idx_isd] * actual_limits.get('i_sd', limit_values['i'])
        i_q_val = state[idx_isq] * actual_limits.get('i_sq', limit_values['i'])
        omega_mech = state[idx_omega] * actual_limits.get('omega', limit_values['omega'])
        n_val = omega_mech * 60 / (2 * np.pi)
        
        # Logging
        data_log['time'].append(k * args.tau)
        data_log['i_d'].append(i_d_val)
        data_log['i_q'].append(i_q_val)
        data_log['n'].append(n_val)
        data_log['u_d'].append(u_d_val)
        data_log['u_q'].append(u_q_val)
        data_log['i_d_ref'].append(id_ref_active)
        data_log['i_q_ref'].append(iq_ref_active)
    
    # CSV Export
    df = pd.DataFrame(data_log)
    df['n_ref'] = args.n_rpm
    filename = os.path.join(out_dir, args.output if args.output else f'sim_{i+1:04d}.csv')
    df.to_csv(filename, index=False)
    print(f"-> Simulation {i+1} exportiert: {filename}")

print("Export abgeschlossen.")
