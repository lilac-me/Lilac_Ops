# Qwen3.5 on Ascend: Evidence Checklist

Use this optional checklist only when the active model/runtime evidence identifies a
Qwen3.5-family architecture on an Ascend training/rollout stack. It is a routing aid,
not a hardcoded fix list and not proof that another installation has the same gaps.

## Preserve these semantic families while minimizing

- text embedding, decoder final normalization, LM head and padded vocabulary handling;
- every active decoder layer type, including full attention and recurrent/GDN-style
  blocks when both appear in the effective layer schedule;
- Q/K/V projection ordering, head grouping, RoPE and mask construction;
- residual/RMSNorm ordering and accumulation precision;
- MoE router, top-k selection, expert layout and load-time resharding;
- vision attention/projector/multimodal merge when the reproducing sample is multimodal;
- actor-to-rollout weight publication/reload and worker generation transitions.

Do not infer the effective architecture from the source `config.json` alone. Record the
training-side effective config, inference worker config, instantiated layer classes,
loaded parameter inventory and one active call trace.

## High-value call-context fingerprints

When the first mismatch is around a linear/LM-head/attention boundary, capture:

```text
logical and physical sequence/token count
GEMM M/N/K
unpadded and padded head_dim/vocabulary/expert dimensions
Q/K/V concatenation and split order
kernel, fusion and graph identity
input/output/accumulation dtype
contiguity, stride and layout
request/task/worker generation
```

The same operator symbol with a different `M`, padding class or kernel is a different
validation cell. Reproduce the failing context before changing mathematics.

## Common investigation branches

Treat these only as evidence requests:

1. text versus multimodal preprocessing and position/mask construction;
2. recurrent/GDN, RMSNorm and MoE execution order or precision;
3. full-attention QKV ordering and grouped-head reshaping;
4. decoder final normalization, padded-vocabulary masking and log-softmax precision;
5. LM-head GEMM shape changes caused by prompt/response slicing or padding;
6. actor-to-rollout reload, reshard/layout and stale-worker state;
7. vision attention head-dimension padding and fused-kernel selection.

For branches 5-7, a static one-forward pass can be exact while a shape/state transition
still fails. Use Difference Packet v2 and the stateful validation matrix rather than
forcing the observation into a Tensor-only packet.

## Production disposition

Keep capture hooks, forced-response controls, fingerprints and Tensor writers in the
private session tree. Production patches may contain independently gated alignment
features, but must pass `validate_production_patch.py` and a fresh-process clean-runtime
audit with replay/capture disabled.
