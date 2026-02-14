# Sphinx Documentation Setup

This document explains the Sphinx auto-documentation setup for the `embark.benchmark` module.

## Overview

Sphinx automatically generates API documentation from Python docstrings in the code. The documentation is built from:

- **Docstrings** in Python modules (Google-style format)
- **Type hints** in function signatures  
- **RST files** in `source/api/` that organize modules

## Quick Start

### Build Documentation

```bash
cd embark/docs

# Using Makefile (Linux/Mac)
make html

# Using make.bat (Windows)
make.bat html

# Using Python script
python build_docs.py

# Or directly with sphinx-build
sphinx-build -b html source build/html
```

### View Documentation

Open `build/html/index.html` in your browser.

## Structure

```
embark/docs/
├── source/              # Sphinx source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main index page
│   └── api/            # API documentation RST files
│       ├── benchmark.rst
│       ├── interfaces.rst
│       ├── metrics.rst
│       └── contrib_neurobench.rst
├── build/              # Generated documentation (gitignored)
├── Makefile            # Build commands (Linux/Mac)
├── make.bat            # Build commands (Windows)
├── build_docs.py       # Python build script
└── README.md           # This file
```

## Configuration

The Sphinx configuration (`source/conf.py`) includes:

- **Napoleon extension** - Parses Google-style docstrings
- **Autodoc** - Auto-generates docs from docstrings
- **Type hints** - Includes type information
- **Intersphinx** - Links to Python, PyTorch, NumPy docs
- **Read the Docs theme** - Clean, readable HTML theme

## Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> dict[str, float]:
    """Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Dictionary with result keys.

    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        {'key': 1.0}
    """
    ...
```

## Module Coverage

The following modules are documented:

- `embark.benchmark` - Main module
- `embark.benchmark.harness` - Benchmark harness
- `embark.benchmark.agents` - Controller agents
- `embark.benchmark.adapters` - Controller adapters
- `embark.benchmark.tasks` - Benchmark tasks
- `embark.benchmark.processors` - State/action processors
- `embark.benchmark.metrics` - Metric accumulators
- `embark.benchmark.physics` - Physics engines
- `embark.benchmark.controllers` - Controller wrappers
- `embark.benchmark.interfaces` - Protocol definitions
- `embark.benchmark.contrib.neurobench` - NeuroBench integration

## Adding New Modules

To add documentation for a new module:

1. **Add docstrings** to your Python code
2. **Create/update RST file** in `source/api/`:

```rst
New Module
==========

.. automodule:: embark.benchmark.new_module
   :members:
   :undoc-members:
   :show-inheritance:
```

3. **Add to index** in `source/index.rst`:

```rst
.. toctree::
   api/new_module
```

4. **Rebuild** documentation

## Troubleshooting

### Import Errors

If Sphinx can't import modules:

1. Ensure dependencies are installed: `poetry install`
2. Check Python path in `conf.py` (should auto-detect project root)
3. Verify you're running from the correct directory

### Missing Documentation

If a module/class doesn't appear:

1. Check that it has docstrings
2. Verify the module is listed in the appropriate `.rst` file
3. Check for import errors in the build log

### Build Warnings

Common warnings (usually safe to ignore):

- `undoc-members` - Expected for private methods
- `autodoc: failed to import` - Check dependencies
- `duplicate object description` - Check for duplicate entries

## Integration with CI/CD

To build docs in CI:

```yaml
# Example GitHub Actions
- name: Build documentation
  run: |
    cd embark/docs
    poetry run sphinx-build -b html source build/html
```

## Related Documentation

- **[BENCHMARK_USER_GUIDE.md](BENCHMARK_USER_GUIDE.md)** - Usage guide (markdown)
- **[BENCHMARK_API.md](BENCHMARK_API.md)** - API reference (markdown)
- **[BENCHMARK_QUICK_REFERENCE.md](BENCHMARK_QUICK_REFERENCE.md)** - Quick reference
