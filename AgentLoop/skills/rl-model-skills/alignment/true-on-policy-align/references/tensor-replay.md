# Diagnostic Tensor Replay

Replay is a diagnostic bypass used to continue prefill discovery after a proved
difference. It is never a production repair and never counts as native alignment.

The logical replay contract is backend-independent. The manifest and helper below are
the included **PyTorch adapter**. A non-PyTorch backend must implement the same context,
hash, shape/dtype, activation-count and immediate-downstream gates with a safe
backend-native loader before replay is permitted.

## Canonical direction

Make inference consume the training-side semantic output at the selected boundary. The
training run keeps its real output; the inference hook loads the captured training Tensor
and replaces the matching output.

Use the prompt-sliced training Tensor when training contains prompt+response. Preserve
the manifest transform ID and causal-shift record.

## Entry gate

Require:

1. a validator-clean Difference Packet;
2. exact causal inputs and a finite differing output;
3. an approved diagnostic-replay authorization;
4. a stable boundary with all causal outputs/side effects identified;
5. a session-created training Tensor and SHA256 hash.

Replay may be used immediately after discovery; do not wait for a failed repair attempt.
The purpose of the discovery phase is to bypass first and continue finding independent
differences. Repair happens later.

## Included PyTorch manifest

`replay_chain.json` contains ordered entries:

```json
{
  "schema_version": 1,
  "phase": "prefill",
  "canonical_side": "train",
  "entries": [
    {
      "diff_id": "D001",
      "enabled": true,
      "boundary": "decoder.layer0.attention.output",
      "tensor_file": "discovery/D001/tensors/train_reference.pt",
      "sha256": "sha256:<64 hex>",
      "weights_only": true,
      "context": {
        "model_digest": "sha256:<64 hex>",
        "config_digest": "sha256:<64 hex>",
        "sample_digest": "sha256:<64 hex>",
        "prompt_token_length": 196,
        "token_slice": "slice:0,0,196",
        "layer": 0,
        "symbol": "qualified.symbol",
        "call_index": 0,
        "shape": [196, 16, 128],
        "dtype": "torch.bfloat16"
      },
      "return_selector": "output.0",
      "side_effects_covered": ["context output"],
      "activation_gate": {
        "expected_hits": 1,
        "post_injection_torch_equal": true,
        "post_injection_max_abs_diff": 0,
        "immediate_downstream_input_exact": true
      }
    }
  ]
}
```

Validate before activation:

```bash
python scripts/validate_replay_manifest.py --manifest replay_chain.json
```

## Included PyTorch safe loading

Use `scripts/tensor_replay.py` from a narrowly scoped hook:

```python
from tensor_replay import ReplayStore

store = ReplayStore("/session/replay_chain.json")

def patched_forward(...):
    original = original_forward(...)
    context = {
        "model_digest": "...",
        "config_digest": "...",
        "sample_digest": "...",
        "prompt_token_length": 196,
        "token_slice": "slice:0,0,196",
        "layer": 0,
        "symbol": "qualified.symbol",
        "call_index": 0,
    }
    replacement = store.replacement("D001", original, context)
    return replacement
```

The loader:

- requires `weights_only=True` support;
- verifies SHA256 before loading;
- rejects context mismatch;
- rejects shape/dtype mismatch;
- moves the Tensor to the current device without numeric casting;
- rejects wildcard paths and duplicate activation beyond `expected_hits`.

Do not load an untrusted user-supplied pickle or silently fall back to unrestricted
`torch.load`.

## Boundary escalation

An internal operator boundary is unsafe when it has unobserved auxiliary returns,
mutates buffers/cache, changes an alias, or has native side effects used downstream.

Escalate:

```text
native FA kernel output
  -> attention context semantic output
  -> complete Attention module output
```

Choose the first outer boundary that covers all causal effects. Record the rejected inner
boundary and limitation in the Difference Packet and Module 6 report.

## Activation proof

After installing the hook:

1. confirm the expected worker imported the hook;
2. require exactly the declared number of hits;
3. verify loaded file hash;
4. compare the injected value to the training reference exactly;
5. re-dump the immediate downstream input and require exactness.

If any gate fails, disable the entry and do not use downstream results.

## Replay chain semantics

A downstream Difference Packet lists every active upstream replay ID. Its evidence means:

> This independent difference occurs from exact inputs under the declared replay prefix.

It does not mean the unresolved upstream boundary is fixed. Module 6 must show the replay
dependency chain.

## End-of-repair behavior

- Remove the replay entry when its repair becomes native exact.
- Restore it after a failed candidate rollback.
- Keep irreducible entries such as FA only for the conditional final run.
- Run a separate no-replay native prefill verification.
- Never include replay hooks or Tensor payloads in production patches or the public case.
