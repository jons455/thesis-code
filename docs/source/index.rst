PMSM Neuromorphic Benchmark
===========================

A benchmark pipeline for systematic evaluation of Spiking Neural Network (SNN)
controllers versus conventional PI controllers for PMSM current control.

Welcome
-------

This documentation is auto-generated from code docstrings using Sphinx.

For markdown guides and tutorials, see the `../` directory:
- `BENCHMARK_USER_GUIDE.md` - Usage guide with examples
- `BENCHMARK_QUICK_REFERENCE.md` - Quick reference cheat sheet
- `BENCHMARK_API.md` - Complete API reference (markdown version)
- `METRICS.md` - Metric definitions

Quick Start
-----------

.. code-block:: python

   from embark.benchmark import (
       ClosedLoopHarness,
       PIControllerAgent,
       PMSMCurrentControlTask,
   )

   # Create task
   task = PMSMCurrentControlTask.from_config(n_rpm=1000, i_q_ref=2.0)

   # Create controller
   controller = PIControllerAgent.from_system_config(task.physics_engine.config)

   # Run benchmark
   harness = ClosedLoopHarness(task=task, controller=controller)
   results = harness.run()

   print(f"Steps: {results['steps']}")

   task.physics_engine.close()

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   api/benchmark
   api/interfaces
   api/metrics
   api/contrib_neurobench


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

