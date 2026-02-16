#!/usr/bin/env python3
"""Helper script to build Sphinx documentation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    """Build Sphinx documentation."""
    docs_dir = Path(__file__).parent
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    print(f"Building documentation from {source_dir}")
    print(f"Output will be in {build_dir / 'html'}")
    print()

    # Run sphinx-build
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(source_dir),
        str(build_dir / "html"),
    ]

    try:
        subprocess.run(cmd, check=True, cwd=docs_dir)
        print()
        print("✓ Documentation built successfully!")
        print(f"  Open {build_dir / 'html' / 'index.html'} in your browser")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed with exit code {e.returncode}")
        return 1
    except FileNotFoundError:
        print("✗ Sphinx not found. Install with: poetry install --with dev")
        return 1


if __name__ == "__main__":
    sys.exit(main())
