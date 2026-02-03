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
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from embark.benchmark.agents import PIControllerAgent  # noqa: E402
from embark.benchmark.interfaces import ReferenceDict  # noqa: E402
from embark.benchmark.tasks.pmsm_current_control import (  # noqa: E402
    PMSMCurrentControlTask,
)
from embark.benchmark.tasks.reference_generators import (  # noqa: E402
    ReferenceGenerator,
)
from embark.utils.config import DEFAULT_PMSM  # noqa: E402
from embark.utils.paths import DATA_RAW_DIR  # noqa: E402


@dataclass
class APRBSReference(ReferenceGenerator):
    """
    Amplitude Pseudo-Random Binary Signal (APRBS) Generator.

    Changes the reference value at fixed intervals to random levels.

    """

    step_interval: int = 500
    min_val: float = -8.0
    max_val: float = 8.0
    i_d_ref: float = 0.0

    def __post_init__(self):
        self._current_iq = 0.0
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self._current_iq = 0.0

    def __call__(self, step: int, time_s: float) -> ReferenceDict:
        if step > 0 and step % self.step_interval == 0:
            self._current_iq = self._rng.uniform(self.min_val, self.max_val)
        return {"i_d_ref": self.i_d_ref, "i_q_ref": self._current_iq}


def generate_episode(
    n_rpm: float,
    max_steps: int = 2000,
    step_time: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a single episode with dynamic multi-step references (APRBS).

    Args:
        n_rpm: Constant speed for this episode (simulating mechanical inertia).
        max_steps: Total length of the episode.
        step_time: Simulation time step (not used directly by task, but by physics).

    Returns:
        pd.DataFrame: Trajectory data including dynamic references.

    """
    # Create Task with APRBS Reference
    # We use a custom reference generator to match the APRBS logic
    # of changing every 500 steps.
    reference_generator = APRBSReference(step_interval=500, min_val=-8.0, max_val=8.0)

    task = PMSMCurrentControlTask.from_config(
        n_rpm=n_rpm,
        max_steps=max_steps,
    )
    # Inject our custom reference generator
    task.reference_generator = reference_generator

    # Use robust parametric tuning (auto-scales with motor parameters)
    agent = PIControllerAgent(
        kp_d=DEFAULT_PMSM.kp_d_optimum,
        ki_d=DEFAULT_PMSM.ki_stable,
        kp_q=DEFAULT_PMSM.kp_q_optimum,
        ki_q=DEFAULT_PMSM.ki_stable,
    )

    state, reference = task.reset()
    agent.reset()

    data = []
    done = False
    step = 0

    while not done and step < max_steps:
        # Agent calculates voltage based on current error
        action = agent(state, reference)

        # Physical values for logging
        i_d = float(state.get("i_d", 0.0))
        i_q = float(state.get("i_q", 0.0))
        u_d = float(action.get("v_d", 0.0))
        u_q = float(action.get("v_q", 0.0))
        i_d_ref = float(reference.get("i_d_ref", 0.0))
        i_q_ref = float(reference.get("i_q_ref", 0.0))

        # Log data
        data.append(
            {
                "time": step * 1e-4,  # Assuming 10kHz default
                "i_d": i_d,
                "i_q": i_q,
                "n": n_rpm,
                "u_d": u_d,
                "u_q": u_q,
                "i_d_ref": i_d_ref,
                "i_q_ref": i_q_ref,
            }
        )

        state, reference, done = task.step(action)
        step += 1

    return pd.DataFrame(data)


def validate_episode(df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """
    Check if episode achieved good tracking.

    For multi-step data, we check if the final state matches the final reference.

    """
    if df.empty:
        return False

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
        default=500,
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

        # 2. Generate Episode (Initial refs are handled by APRBS inside generator)
        df = generate_episode(
            n_rpm=n_rpm,
            max_steps=args.max_steps,
        )

        # 3. Validate if requested
        if args.validate and not validate_episode(df):
            rejected += 1
            continue

        # 4. Save
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
