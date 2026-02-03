"""
Generate Clean Training Data for SNN (Multi-Step APRBS)
=======================================================

Uses the stable benchmark PI controller to generate high-quality
training trajectories for SNN imitation learning.

This version uses a Multi-Step APRBS (Amplitude Pseudo-Random Binary Signal)
profile to force the model to learn dynamics, zero-crossings, and
symmetric control behavior.

Usage:
    poetry run python scripts/generate_training_data.py
    poetry run python scripts/generate_training_data.py --num-files 500
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.pmsm_env import PMSMEnv
from embark.utils.config import DEFAULT_PMSM
from embark.utils.paths import DATA_RAW_DIR


def generate_episode(
    n_rpm: float,
    i_d_initial: float,
    i_q_initial: float,
    max_steps: int = 2000,
    step_time: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a single episode with dynamic multi-step references (APRBS).

    Unlike static step responses, this function changes the target current
    every 500 steps to simulate dynamic load changes, zero-crossings,
    and braking maneuvers.

    Args:
        n_rpm: Constant speed for this episode (simulating mechanical inertia).
        i_d_initial: Initial d-axis current reference.
        i_q_initial: Initial q-axis current reference.
        max_steps: Total length of the episode.
        step_time: Simulation time step.

    Returns:
        pd.DataFrame: Trajectory data including dynamic references.
    """
    env = PMSMEnv(
        n_rpm=n_rpm,
        i_d_ref=i_d_initial,
        i_q_ref=i_q_initial,
        step_time=step_time,
        max_steps=max_steps,
    )

    # Use robust parametric tuning (auto-scales with motor parameters)
    agent = PIControllerAgent(
        kp_d=DEFAULT_PMSM.kp_d_optimum,
        ki_d=DEFAULT_PMSM.ki_stable,
        kp_q=DEFAULT_PMSM.kp_q_optimum,
        ki_q=DEFAULT_PMSM.ki_stable,
    )

    state, info = env.reset()
    agent.reset()

    # Track current active references (they will change during the episode)
    current_iq_ref = i_q_initial
    current_id_ref = i_d_initial

    # APRBS Configuration: Change target every 50ms (500 steps @ 100us)
    step_interval = 500

    data = []

    for step in range(max_steps):
        # --- DYNAMIC PROFILE LOGIC (APRBS) ---
        if step > 0 and step % step_interval == 0:
            # Generate a new random target for q-axis (Torque)
            # Range [-8.0, 8.0] covers motoring, braking, and zero-crossing.
            # We stay slightly below I_max (10.8A) to avoid permanent saturation.
            current_iq_ref = np.random.uniform(-8.0, 8.0)

            # Apply new reference to the environment
            # (Crucial so that the error calculation inside Env is correct)
            env.i_q_ref = current_iq_ref

            # Note: We keep i_d_ref constant (usually 0) for now.
            # To learn Field Weakening, you could vary i_d_ref here too.
        # -------------------------------------

        # Agent calculates voltage based on current error (State - Reference)
        action = agent(state)

        # Physical values for logging
        i_d = float(state[0])
        i_q = float(state[1])
        u_d = float(action[0])
        u_q = float(action[1])

        # Log the dynamic reference values, NOT the initial ones!
        data.append(
            {
                "time": step * 1e-4,
                "i_d": i_d,
                "i_q": i_q,
                "n": n_rpm,
                "u_d": u_d,
                "u_q": u_q,
                "i_d_ref": current_id_ref,  # Current active reference
                "i_q_ref": current_iq_ref,  # Current active reference
            }
        )

        state, reward, done, trunc, info = env.step(action)

        if done:
            break

    env.close()
    return pd.DataFrame(data)


def validate_episode(df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """
    Check if episode achieved good tracking.

    For multi-step data, we check if the final state matches the final reference.
    """
    final_error_q = abs(df["i_q"].iloc[-1] - df["i_q_ref"].iloc[-1])

    # Check sign is correct (same sign as reference)
    ref_end = df["i_q_ref"].iloc[-1]
    val_end = df["i_q"].iloc[-1]

    if abs(ref_end) > 0.1:  # Only check sign if reference is not effectively zero
        sign_correct = np.sign(val_end) == np.sign(ref_end)
    else:
        sign_correct = True

    return final_error_q < threshold and sign_correct


def main():
    parser = argparse.ArgumentParser(description="Generate SNN training data (APRBS)")
    parser.add_argument(
        "--num-files",
        type=int,
        default=500,  # Default increased to 500 for better coverage
        help="Number of training files to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_RAW_DIR / "train"),
        help="Output directory",
    )
    parser.add_argument("--max-steps", type=int, default=2000, help="Steps per episode")
    parser.add_argument(
        "--validate", action="store_true", help="Only keep files with good tracking"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating SNN Training Data (Multi-Step APRBS)")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Files: {args.num_files}")
    print(f"Steps per file: {args.max_steps}")
    print("Profile: Random Step Change every 500 steps (+/- 8A)")
    print()

    # Operating point ranges
    np.random.seed(42)

    generated = 0
    rejected = 0

    for _ in tqdm(range(args.num_files), desc="Generating"):
        # 1. Random Constant Speed per Episode (Simulating Mechanical Inertia)
        n_rpm = np.random.uniform(500, 2500)

        # 2. Initial References (Will be changed by APRBS logic immediately or later)
        i_d_start = 0.0  # Standard MTPA
        i_q_start = 0.0  # Start from zero load

        # 3. Generate Episode
        df = generate_episode(
            n_rpm=n_rpm,
            i_d_initial=i_d_start,
            i_q_initial=i_q_start,
            max_steps=args.max_steps,
        )

        # 4. Validate if requested
        if args.validate and not validate_episode(df):
            rejected += 1
            continue

        # 5. Save
        filename = output_dir / f"sim_{generated+1:04d}.csv"
        df.to_csv(filename, index=False)
        generated += 1

    print()
    print(f"Generated: {generated} files")
    if args.validate:
        print(f"Rejected: {rejected} files (bad tracking)")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
