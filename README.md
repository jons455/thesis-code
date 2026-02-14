# EMBARK — End-to-end Motor BenchmARK

**Master's Thesis — Jonas Reif**

End-to-end benchmark pipeline for Spiking Neural Network (SNN) vs PI controllers for PMSM current control.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

## Setup

```bash
# Clone the repository
git clone git@git.informatik.fh-nuernberg.de:reifjo96249/embark.git
cd embark

# Install dependencies
poetry install
```

Activate the environment: `poetry shell`

Optional (dev + pre-commit):

```bash
poetry install --with dev
poetry run pre-commit install
```

## Pushing to the repository

From a fresh copy of the project (no git yet):

```bash
poetry install
git init
git remote add origin git@git.informatik.fh-nuernberg.de:reifjo96249/embark.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

(Use `master` instead of `main` if your server default branch is `master`.)

## Run tests

```bash
poetry run pytest
```

## Project layout

- `embark/` — benchmarking pipeline (harness, metrics, controllers, physics)
- `tests/` — pytest tests
- `docs/` — documentation

## Acknowledgments

- [gym-electric-motor (GEM)](https://github.com/upb-lea/gym-electric-motor)
- [NeuroBench](https://neurobench.readthedocs.io/)
- [snnTorch](https://snntorch.readthedocs.io/)
