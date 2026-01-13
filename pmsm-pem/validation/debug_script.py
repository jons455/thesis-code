"""Debug script to isolate the bug."""

import gem_controllers as gc
import gym_electric_motor as gem
import numpy as np
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad

motor_parameter = dict(p=3, r_s=0.543, l_d=0.00113, l_q=0.00142, psi_p=0.0169)
limit_values = dict(i=10.8, u=48.0, omega=3000 * 2 * np.pi / 60)

motor = PermanentMagnetSynchronousMotor(motor_parameter=motor_parameter, limit_values=limit_values)
load = ConstantSpeedLoad(omega_fixed=500 * 2 * np.pi / 60)

env = gem.make(
    "Cont-CC-PMSM-v0", motor=motor, load=load, tau=1e-4, visualization=None, constraints=()
)

env_unwrapped = env
while hasattr(env_unwrapped, "env"):
    env_unwrapped = env_unwrapped.env

ps = env_unwrapped.physical_system
state_names = list(ps.state_names)
actual_limits = {name: ps.limits[i] for i, name in enumerate(state_names)}

# Exakt wie im Hauptskript
controller = gc.GemController.make(
    env_unwrapped,
    "Cont-CC-PMSM-v0",
    decoupling=True,
    current_safety_margin=0.2,
    base_current_controller="PI",
    a=4,
    block_diagram=False,
)


def extract_state(result):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, tuple):
        result = result[0]
    return result


def get_state_indices(env, state_vec):
    env_unwrapped = env
    while hasattr(env_unwrapped, "env"):
        env_unwrapped = env_unwrapped.env
    names = list(env_unwrapped.physical_system.state_names)
    return (names.index("i_sd"), names.index("i_sq"), names.index("omega"), names.index("epsilon"))


# Simulation - exakt wie im Skript
id_ref = 0.0
iq_ref = 2.0

state = extract_state(env.reset())
if hasattr(controller, "reset"):
    controller.reset()

data_log = {"i_d": [], "i_q": []}

for k in range(600):
    state_vec = extract_state(state)
    idx_isd, idx_isq, idx_omega, idx_epsilon = get_state_indices(env, state_vec)
    epsilon = state_vec[idx_epsilon] * np.pi

    # step_time = 0 -> step_time_k = 0 -> immer aktiv
    id_ref_active = id_ref
    iq_ref_active = iq_ref

    ref_normalized = np.array(
        [id_ref_active / limit_values["i"], iq_ref_active / limit_values["i"]]
    )
    action_abc = controller.control(state_vec, ref_normalized)

    step_result = env.step(action_abc)
    if len(step_result) == 5:
        state, reward, done, truncated, info = step_result
    else:
        (state, _), reward, done, _, _ = step_result

    if done:
        state = extract_state(env.reset())

    state_vec = extract_state(state)
    idx_isd, idx_isq, idx_omega, _ = get_state_indices(env, state_vec)

    i_d_val = state_vec[idx_isd] * actual_limits.get("i_sd", limit_values["i"])
    i_q_val = state_vec[idx_isq] * actual_limits.get("i_sq", limit_values["i"])

    data_log["i_d"].append(i_d_val)
    data_log["i_q"].append(i_q_val)

    if k >= 490 and k <= 510:
        print(f"Step {k}: i_d={i_d_val:.4f}, i_q={i_q_val:.4f}")

print()
print(
    f'Steady-state avg (k >= 300): i_d={np.mean(data_log["i_d"][300:]):.4f}, i_q={np.mean(data_log["i_q"][300:]):.4f}'
)

# CSV schreiben und vergleichen
import pandas as pd

df = pd.DataFrame(data_log)
df["time"] = [k * 1e-4 for k in range(600)]
df.to_csv("export/gem_standard/debug_output.csv", index=False)
print("CSV geschrieben: export/gem_standard/debug_output.csv")
