"""Ascend operator (ops-dev pipeline) agent benchmark.

Submodules are imported explicitly (``benchmark.agents.ops.scoring`` etc.),
never here — importing them at package load causes a ``py -m`` re-import
warning and slows down every unrelated tool that merely touches the package.
"""
