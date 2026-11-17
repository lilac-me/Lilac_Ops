"""Adversarial tests for the anti-cheat check (V5).

The anti-cheat exists to stop a run from passing off a *torch wrapper* as a
Triton kernel. Its first version was a light regex, so before trusting it we
throw a set of deliberately evasive wrappers at it and record which ones slip
through. Anything that slips through is a documented limitation until the check
is hardened.

Every variant below is a **cheat**: the operator's numeric result is produced by
torch, not by a Triton kernel.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.agents.ops import metrics  # noqa: E402

# --- legitimate baseline: a real Triton kernel -----------------------------
LEGIT = """import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(y_ptr + o, tl.where(tl.load(x_ptr + o, mask=m) < 0, 0.0,
                                 tl.load(x_ptr + o, mask=m)), mask=m)

def relu(x):
    y = torch.empty_like(x)
    relu_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y
"""

CHEATS = {
    "direct_F_call": """import torch, torch.nn.functional as F
def relu(x):
    return F.relu(x)
""",
    "from_import_bare_name": """from torch.nn.functional import relu
def relu(x):
    return relu(x)
""",
    "aliased_module_call": """import torch
import torch.nn.functional as fn
def relu(x):
    return fn.relu(x)
""",
    "getattr_bypass": """import torch
def relu(x):
    return getattr(torch.nn.functional, "relu")(x)
""",
    "torch_op_direct": """import torch
def relu(x):
    return torch.relu(x)
""",
    "dummy_jit_plus_wrapper": """import triton
import triton.language as tl
import torch, torch.nn.functional as F

@triton.jit
def unused_kernel(p, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(p + o, tl.load(p + o, mask=o < n), mask=o < n)

def relu(x):
    return F.relu(x)     # real work delegated to torch
""",
    "eval_indirection": """import torch
def relu(x):
    return eval("torch.nn.functional.relu")(x)
""",
}

# Harder class: the cheat *also* ships a real-looking @triton.jit kernel (so the
# "is it Triton-native" check passes) and delegates evasively. These are the
# variants that actually stress the delegation check rather than the native check.
_DUMMY_JIT = """import triton
import triton.language as tl
import torch

@triton.jit
def unused_kernel(p, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(p + o, tl.load(p + o, mask=o < n), mask=o < n)
"""

CHEATS.update({
    "dummy_jit__aliased_module": _DUMMY_JIT + """import torch.nn.functional as fn
def relu(x):
    return fn.relu(x)
""",
    "dummy_jit__getattr": _DUMMY_JIT + """
def relu(x):
    return getattr(torch.nn.functional, "relu")(x)
""",
    "dummy_jit__from_import": _DUMMY_JIT + """from torch.nn.functional import relu as _r
def relu(x):
    return _r(x)
""",
    "dummy_jit__module_attr": _DUMMY_JIT + """
import torch.nn.functional
def relu(x):
    return torch.nn.functional.relu(x)
""",
})


def _ws(tmp_path: pathlib.Path, code: str) -> pathlib.Path:
    ws = tmp_path / "run"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "kernel.py").write_text(code, encoding="utf-8")
    return ws


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_cheat_is_caught(tmp_path, name):
    ac = metrics.anti_cheat(_ws(tmp_path, CHEATS[name]))
    assert ac["pass"] is False, (
        f"anti-cheat let '{name}' through: {ac}")


def test_legit_kernel_passes(tmp_path):
    ac = metrics.anti_cheat(_ws(tmp_path, LEGIT))
    assert ac["pass"] is True, ac
