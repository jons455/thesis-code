# Benchmark Documentation

This directory contains documentation for the `embark.benchmark` module.

## Documentation Structure

### Markdown Documentation

- **[BENCHMARK_INDEX.md](BENCHMARK_INDEX.md)** - Overview and navigation
- **[BENCHMARK_API.md](BENCHMARK_API.md)** - Complete API reference
- **[BENCHMARK_USER_GUIDE.md](BENCHMARK_USER_GUIDE.md)** - Usage guide with examples
- **[BENCHMARK_QUICK_REFERENCE.md](BENCHMARK_QUICK_REFERENCE.md)** - Quick reference cheat sheet
- **[METRICS.md](METRICS.md)** - Metric definitions and output formats

### Sphinx Documentation (Auto-generated from Docstrings)

The `source/` directory contains Sphinx configuration for auto-generating API documentation from code docstrings.

## Building Sphinx Documentation

### Prerequisites

Install Sphinx dependencies:

```bash
poetry install --with dev
```

### Build HTML Documentation

From the `embark/docs/` directory:

```bash
# Using Makefile (Linux/Mac)
make html

# Or directly with sphinx-build
sphinx-build -b html source build/html
```

On Windows:

```bash
# Using sphinx-build directly
sphinx-build -b html source build/html
```

### View Documentation

After building, open `build/html/index.html` in your browser.

### Clean Build

```bash
make clean
# or
rm -rf build/
```

## Documentation Sources

The Sphinx documentation is auto-generated from:

- **Docstrings** in Python modules (Google-style format)
- **Type hints** in function signatures
- **RST files** in `source/api/` that organize the modules

### Updating Documentation

1. **Update docstrings** in Python code (Google-style format)
2. **Rebuild** with `make html`
3. **Commit** both code changes and rebuilt docs (if desired)

### Docstring Format

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

The Sphinx documentation covers:

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

## Troubleshooting

### Import Errors

If Sphinx can't import modules:

1. Ensure you're in the project root
2. Install dependencies: `poetry install`
3. Check Python path in `conf.py`

### Missing Documentation

If a module/class doesn't appear:

1. Check that it has docstrings
2. Verify the module is listed in the appropriate `.rst` file
3. Check for import errors in the build log

### Build Warnings

Common warnings:

- `undoc-members` - Expected for private methods
- `autodoc: failed to import` - Check dependencies
- `duplicate object description` - Check for duplicate entries in `.rst` files
