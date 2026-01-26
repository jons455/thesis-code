"""
Generate Clean Training Data for SNN
=====================================

Uses the stable benchmark PI controller to generate high-quality
training trajectories for SNN imitation learning.

Usage:
    poetry run python scripts/generate_training_data.py
    poetry run python scripts/generate_training_data.py --num-files 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.benchmark.agents import PIControllerAgent
from embark.benchmark.pmsm_env import PMSMEnv
from embark.utils.paths import DATA_RAW_DIR


def generate_episode(
    n_rpm: float,
    i_d_ref: float,
    i_q_ref: float,
    max_steps: int = 2000,
    step_time: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a single episode of PI controller data.

    Returns DataFrame with columns:
    time, i_d, i_q, n, u_d, u_q, i_d_ref, i_q_ref
    """
    env = PMSMEnv(
        n_rpm=n_rpm,
        i_d_ref=i_d_ref,
        i_q_ref=i_q_ref,
        step_time=step_time,
        max_steps=max_steps,
    )
    agent = PIControllerAgent()

    state, info = env.reset()
    agent.reset()

    data = []
    for step in range(max_steps):
        action = agent(state)

        # Physical values for logging
        i_d = float(state[0])
        i_q = float(state[1])
        u_d = float(action[0])
        u_q = float(action[1])

        # Get current reference
        i_d_ref_curr, i_q_ref_curr = env._get_current_reference()

        data.append(
            {
                "time": step * 1e-4,
                "i_d": i_d,
                "i_q": i_q,
                "n": n_rpm,
                "u_d": u_d,
                "u_q": u_q,
                "i_d_ref": i_d_ref_curr,
                "i_q_ref": i_q_ref_curr,
            }
        )

        state, reward, done, trunc, info = env.step(action)

        if done:
            break

    env.close()
    return pd.DataFrame(data)


def validate_episode(df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """Check if episode achieved good tracking."""
    final_error_q = abs(df["i_q"].iloc[-1] - df["i_q_ref"].iloc[-1])

    # Check sign is correct (same sign as reference)
    if df["i_q_ref"].iloc[-1] != 0:
        sign_correct = np.sign(df["i_q"].iloc[-1]) == np.sign(df["i_q_ref"].iloc[-1])
    else:
        sign_correct = True

    return final_error_q < threshold and sign_correct


def main():
    parser = argparse.ArgumentParser(description="Generate SNN training data")
    parser.add_argument(
        "--num-files",
        type=int,
        default=1000,
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
    print("Generating SNN Training Data")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Files: {args.num_files}")
    print(f"Steps per file: {args.max_steps}")
    print()

    # Operating point ranges
    np.random.seed(42)

    generated = 0
    rejected = 0

    for _ in tqdm(range(args.num_files), desc="Generating"):
        # Random operating point
        n_rpm = np.random.uniform(500, 2500)
        i_d_ref = np.random.uniform(-3, 0)  # Usually 0 or small negative
        i_q_ref = np.random.uniform(0.5, 8)  # Positive load current

        # Sometimes negative i_q (regeneration)
        if np.random.random() < 0.2:
            i_q_ref = -i_q_ref

        # Generate episode
        df = generate_episode(
            n_rpm=n_rpm,
            i_d_ref=i_d_ref,
            i_q_ref=i_q_ref,
            max_steps=args.max_steps,
        )

        # Validate if requested
        if args.validate and not validate_episode(df):
            rejected += 1
            continue

        # Save
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
