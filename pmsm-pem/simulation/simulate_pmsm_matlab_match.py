"""
PMSM FOC Simulation - MATLAB-kompatible Version
Verwendet eigenen PI-Controller mit exakten MATLAB-Parametern
"""

import numpy as np
import pandas as pd
import os
import argparse
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad


class PIController:
    """PI-Regler mit Back-EMF Entkopplung (entspricht MATLAB Simulink)"""
    
    def __init__(self, kp_d, ki_d, kp_q, ki_q, ts, u_max, l_d, l_q, psi_pm, p):
        self.kp_d, self.ki_d = kp_d, ki_d
        self.kp_q, self.ki_q = kp_q, ki_q
        self.ts, self.u_max = ts, u_max
        self.l_d, self.l_q = l_d, l_q
        self.psi_pm, self.p = psi_pm, p
        self.integral_d = 0.0
        self.integral_q = 0.0
        
    def control(self, i_d, i_q, i_d_ref, i_q_ref, omega_elec):
        e_d, e_q = i_d_ref - i_d, i_q_ref - i_q
        
        u_d_p = self.kp_d * e_d
        u_q_p = self.kp_q * e_q
        
        self.integral_d += self.ki_d * e_d * self.ts
        self.integral_q += self.ki_q * e_q * self.ts
        
        u_d_pi = u_d_p + self.integral_d
        u_q_pi = u_q_p + self.integral_q
        
        # Back-EMF Entkopplung
        u_d_decouple = -omega_elec * self.l_q * i_q
        u_q_decouple = omega_elec * (self.l_d * i_d + self.psi_pm)
        
        u_d = u_d_pi + u_d_decouple
        u_q = u_q_pi + u_q_decouple
        
        # Anti-windup
        if abs(u_d) > self.u_max:
            self.integral_d = np.clip(self.integral_d, -self.u_max, self.u_max) - u_d_p
            u_d = np.clip(u_d, -self.u_max, self.u_max)
        if abs(u_q) > self.u_max:
            self.integral_q = np.clip(self.integral_q, -self.u_max, self.u_max) - u_q_p
            u_q = np.clip(u_q, -self.u_max, self.u_max)
        
        return u_d / self.u_max, u_q / self.u_max
    
    def reset(self):
        self.integral_d = 0.0
        self.integral_q = 0.0


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
    p = argparse.ArgumentParser(description="GEM PMSM current-control simulation (MATLAB-matching PI controller)")
    p.add_argument("--n-rpm", type=float, default=716.0, help="Fixed mechanical speed in rpm (ConstantSpeedLoad).")
    p.add_argument("--id-ref", type=float, default=0.0, help="d-axis current reference after step [A].")
    p.add_argument("--iq-ref", type=float, default=3.5, help="q-axis current reference after step [A]. Run 003: 3.5A")
    p.add_argument("--step-time", type=float, default=0.1, help="Step time in seconds.")
    p.add_argument("--sim-steps", type=int, default=sim_steps, help="Number of simulation steps.")
    p.add_argument("--tau", type=float, default=tau, help="Control cycle time in seconds.")
    p.add_argument("--output", type=str, default=None, help="Output filename (default: sim_0001.csv)")
    return p.parse_args()

# Export-Verzeichnis relativ zum pmsm-pem Basisordner
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PMSM_PEM_DIR = os.path.dirname(SCRIPT_DIR)  # pmsm-pem/
out_dir = os.path.join(PMSM_PEM_DIR, 'export', 'matlab_match')
os.makedirs(out_dir, exist_ok=True)

# Environment erstellen
env_id = 'Cont-CC-PMSM-v0'
args = parse_args()
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

# Hole die tatsächlichen Limits vom Physical System (GEM ignoriert unsere limit_values!)
ps = env_unwrapped.physical_system
state_names = list(ps.state_names)
actual_limits = {name: ps.limits[i] for i, name in enumerate(state_names)}
print(f"GEM Limits: omega={actual_limits.get('omega', 0)*60/(2*np.pi):.0f} rpm, "
      f"i_sd={actual_limits.get('i_sd', 0):.1f} A, u_sd={actual_limits.get('u_sd', 0):.1f} V")

# PI Controller mit MATLAB-Parametern: Technische Optimaleinstellung
# Kp = L/(2*Ts), Ki = R/(2*Ts)
K_Pd = motor_parameter['l_d'] / (2 * args.tau)  # 5.65
K_Id = motor_parameter['r_s'] / (2 * args.tau)  # 2715
K_Pq = motor_parameter['l_q'] / (2 * args.tau)  # 7.10
K_Iq = motor_parameter['r_s'] / (2 * args.tau)  # 2715

controller = PIController(
    kp_d=K_Pd, ki_d=K_Id,
    kp_q=K_Pq, ki_q=K_Iq,
    ts=args.tau, u_max=limit_values['u'],
    l_d=motor_parameter['l_d'], l_q=motor_parameter['l_q'],
    psi_pm=motor_parameter['psi_p'], p=motor_parameter['p']
)

print(f"PI Controller (MATLAB-Parameter): Kp_d={K_Pd:.1f}, Ki_d={K_Id:.0f}")


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
    # Feste Test-Werte (wie MATLAB)
    id_ref = float(args.id_ref)    # [A]
    iq_ref = float(args.iq_ref)    # [A]
    
    state = extract_state(env.reset())
    controller.reset()
    
    data_log = {'time': [], 'i_d': [], 'i_q': [], 'n': [], 'u_d': [], 'u_q': [], 'i_d_ref': [], 'i_q_ref': []}
    
    for k in range(args.sim_steps):
        state_vec = extract_state(state)
        idx_isd, idx_isq, idx_omega, idx_epsilon = get_state_indices(env, state_vec)
        
        # Verwende die tatsächlichen GEM-Limits für die Denormalisierung!
        i_d = state_vec[idx_isd] * actual_limits.get('i_sd', limit_values['i'])
        i_q = state_vec[idx_isq] * actual_limits.get('i_sq', limit_values['i'])
        omega_mech = state_vec[idx_omega] * actual_limits.get('omega', limit_values['omega'])
        omega_elec = motor_parameter['p'] * omega_mech
        epsilon = state_vec[idx_epsilon] * np.pi
        
        # Step bei t=0.1s (k=1000 bei tau=0.0001) - wie in MATLAB
        step_time_k = int(round(args.step_time / args.tau))
        if k < step_time_k:
            id_ref_active = 0.0
            iq_ref_active = 0.0
        else:
            id_ref_active = id_ref
            iq_ref_active = iq_ref
        
        u_d_norm, u_q_norm = controller.control(i_d, i_q, id_ref_active, iq_ref_active, omega_elec)
        
        # dq -> abc Transformation
        c, s = np.cos(epsilon), np.sin(epsilon)
        u_alpha = u_d_norm * c - u_q_norm * s
        u_beta = u_d_norm * s + u_q_norm * c
        u_a = u_alpha
        u_b = -0.5 * u_alpha + (np.sqrt(3)/2) * u_beta
        u_c = -0.5 * u_alpha - (np.sqrt(3)/2) * u_beta
        action_abc = np.array([u_a, u_b, u_c])
        
        u_d_val = u_d_norm * limit_values['u']
        u_q_val = u_q_norm * limit_values['u']
        
        # Environment-Schritt
        step_result = env.step(action_abc)
        if len(step_result) == 5:
            state, reward, done, truncated, info = step_result
        else:
            (state, _), reward, done, _, _ = step_result
        
        # Bei Termination: Environment resetten (GEM erfordert das), aber
        # Controller-Zustand behalten um den Integrator nicht zu löschen!
        if done:
            state = extract_state(env.reset())
            # WICHTIG: controller.reset() NICHT aufrufen!
            
        state_vec = extract_state(state)
        idx_isd, idx_isq, idx_omega, _ = get_state_indices(env, state_vec)
        
        # Verwende die tatsächlichen GEM-Limits für die Denormalisierung!
        i_d_val = state_vec[idx_isd] * actual_limits.get('i_sd', limit_values['i'])
        i_q_val = state_vec[idx_isq] * actual_limits.get('i_sq', limit_values['i'])
        omega_mech = state_vec[idx_omega] * actual_limits.get('omega', limit_values['omega'])
        n_val = omega_mech * 60 / (2 * np.pi)
        
        data_log['time'].append(k * args.tau)
        data_log['i_d'].append(i_d_val)
        data_log['i_q'].append(i_q_val)
        data_log['n'].append(n_val)
        data_log['u_d'].append(u_d_val)
        data_log['u_q'].append(u_q_val)
        data_log['i_d_ref'].append(id_ref_active)
        data_log['i_q_ref'].append(iq_ref_active)
        
        # done-Flag ignorieren um volle Simulationsdauer zu erreichen
    
    df = pd.DataFrame(data_log)
    df["n_ref"] = float(args.n_rpm)
    out_name = args.output if args.output else f'sim_{i+1:04d}.csv'
    filename = os.path.join(out_dir, out_name)
    df.to_csv(filename, index=False)
    print(f"-> Simulation {i+1} exportiert: {filename}")

print("Export abgeschlossen.")
