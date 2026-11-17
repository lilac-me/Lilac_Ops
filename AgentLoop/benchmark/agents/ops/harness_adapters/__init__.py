"""Harness adapters: turn harness-native session logs into ``run_meta.json``.

Each harness implements the same contract (input_tokens / output_tokens /
total_tokens / cache_read_tokens / wall_ms) so the scorer stays harness-agnostic.
Submodules are imported explicitly (``...harness_adapters.dsh_tokens``), never here.
"""
