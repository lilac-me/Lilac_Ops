#!/usr/bin/env python3
"""Run AISBench accuracy evaluation on a workspace-managed remote container.

Orchestrates end-to-end accuracy testing:
    1. Service startup (via vllm-ascend-serving, needed for base_url)
    2. Environment preparation (ais-bench-benchmark install, dataset download, config)
    3. Multi-round accuracy evaluation (ais_bench, default 10 rounds)
    4. Service stop + aggregate report

Usage examples:

    # Docker container
    python3 accuracy_run.py --host 192.168.1.100 --container my-container \\
        --key-file ~/.ssh/id_rsa --model /data/models/Qwen3-32B --tp 4 \\
        --extra-env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 --runs 10 \\
        --serve-args --max-model-len 4096

    # Conda environment
    python3 accuracy_run.py --host 192.168.1.100 --conda-env myenv \\
        --model /data/models/Qwen3-32B --tp 2 --runs 5

Progress on stderr as __WORKSPACE_ACCURACY_PROGRESS__=<json>.
Final result on stdout as a single JSON object.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import sys
import urllib.parse
import time
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _common import (
    ROOT,
    AccuracyConfig,
    ExecutionTarget,
    assemble_config,
    call_serve_start,
    call_serve_stop,
    emit_progress,
    now_utc,
    print_json,
    run_remote_cmd,
    write_local_result,
    ascend_env_preamble,
    AIS_BENCH_GIT_URL,
    AIS_BENCH_CLONE_DIR,
    AIS_BENCH_WORK_DIR,
    DEFAULT_MATH_URL,
    TEXTVQA_DATASET_ID,
    CONFIG_REL_PATH,
    DATASET_DIR_REL_PATH,
)

# ---------------------------------------------------------------------------
# Environment preparation
# ---------------------------------------------------------------------------

_ENV_PREP_SCRIPT_TEMPLATE = r'''
set -e

{runtime_env_preamble}
{conda_preamble}
{proxy_preamble}
{log_redirect}
echo "=== Checking ais-bench-benchmark ==="
_PKG=$(pip list 2>/dev/null | awk '{{print $1}}' | grep -iE '^ais[_ -]?bench[_ -]?benchmark$' | head -1)
if [ -n "$_PKG" ]; then
    echo "ais-bench-benchmark already installed (pkg=$_PKG)"
else
    echo "Installing ais-bench-benchmark..."
    if [ ! -d "{clone_dir}" ]; then
        if [ "{git_insecure}" = "1" ]; then
            GIT_SSL_NO_VERIFY=true git clone "{git_url}" "{clone_dir}"
        else
            git clone "{git_url}" "{clone_dir}"
        fi
    fi
    cd "{clone_dir}"
    pip3 install -e ./ --use-pep517 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 2>&1
    _PKG=$(pip list 2>/dev/null | awk '{{print $1}}' | grep -iE '^ais[_ -]?bench[_ -]?benchmark$' | head -1)
    if [ -z "$_PKG" ]; then
        echo "ERROR: ais-bench-benchmark installation failed"
        exit 1
    fi
    echo "ais-bench-benchmark installed successfully (pkg=$_PKG)"
fi

# math-eval deps must install unconditionally, not only when ais-bench was
# freshly installed -- a reused env that already has ais-bench would
# otherwise silently skip them and math500 fails later. Failure is
# non-fatal: these deps are only needed for math500, so aborting on a
# flaky pip would also break textvqa-only runs.
if [ "{dataset_kind}" = "math" ]; then
    echo "=== Checking math-eval dependencies (math_verify, latex2sympy2_extended) ==="
    if python3 -c "import math_verify, latex2sympy2_extended" 2>/dev/null; then
        echo "math-eval deps already available; skipping pip install"
    else
        pip3 install math_verify latex2sympy2_extended -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 2>&1 || true
        if python3 -c "import math_verify, latex2sympy2_extended" 2>/dev/null; then
            echo "math-eval deps installed successfully"
        else
            echo "WARNING: math-eval deps import failed; math500 accuracy will not run" >&2
        fi
    fi
fi

echo "=== Locating ais_bench ==="
LOCATION=$(pip show "$_PKG" 2>/dev/null | grep -E "^Location" | awk '{{print $2}}')
EDITABLE_LOC=$(pip show "$_PKG" 2>/dev/null | grep -E "^Editable project location" | awk -F': ' '{{print $2}}')
echo "pip Location: $LOCATION"
echo "Editable location: $EDITABLE_LOC"
AIS_BENCH_DIR=""
if [ -n "$EDITABLE_LOC" ] && [ -d "$EDITABLE_LOC/ais_bench" ]; then
    AIS_BENCH_DIR="$EDITABLE_LOC/ais_bench"
fi
if [ -z "$AIS_BENCH_DIR" ] && [ -n "$LOCATION" ] && [ -d "$LOCATION/ais_bench" ]; then
    AIS_BENCH_DIR="$LOCATION/ais_bench"
fi
if [ -z "$AIS_BENCH_DIR" ]; then
    # Fallback: search under both editable and regular locations
    for _base in "$EDITABLE_LOC" "$LOCATION"; do
        _F=$(find "$_base" -maxdepth 3 -type d -name "ais_bench" 2>/dev/null | head -1)
        if [ -n "$_F" ] && [ -d "$_F" ]; then
            AIS_BENCH_DIR="$_F"
            break
        fi
    done
fi
if [ -z "$AIS_BENCH_DIR" ] || [ ! -d "$AIS_BENCH_DIR" ]; then
    echo "ERROR: cannot find ais_bench directory under $EDITABLE_LOC or $LOCATION"
    exit 1
fi
echo "ais_bench directory: $AIS_BENCH_DIR"

if [ "{dataset_kind}" = "textvqa" ]; then
    echo "=== Preparing textvqa dataset ==="
    DATASET_DIR="$AIS_BENCH_DIR/{dataset_rel}"
    TEXTVQA_DIR="$DATASET_DIR/textvqa"
    IMG_DIR="$TEXTVQA_DIR/train_images"
    mkdir -p "$DATASET_DIR"
    textvqa_complete() {{
        python3 - "$TEXTVQA_DIR" << 'PYEOF'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
annotations = root / 'textvqa_json/textvqa_val.jsonl'
try:
    samples = [json.loads(line) for line in annotations.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not samples:
        raise ValueError('validation annotations are empty')
    for sample in samples:
        name = Path(sample['image']).name
        image = root / 'train_images' / name
        if not name or not image.is_file() or not image.stat().st_size:
            raise ValueError('missing image: ' + name)
        with image.open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('LFS pointer instead of image: ' + name)
except (OSError, ValueError, KeyError, TypeError) as error:
    print('textvqa cache incomplete:', error)
    sys.exit(1)
print('textvqa cache complete; samples:', len(samples))
PYEOF
    }}
    if ! textvqa_complete; then
        if ! command -v modelscope >/dev/null 2>&1; then
            pip3 install modelscope -i https://mirrors.aliyun.com/pypi/simple/
        fi
        modelscope download --dataset {textvqa_dataset_id} --local_dir "$TEXTVQA_DIR"
    fi

    # Absolutize image paths in every json/jsonl. vLLM rejects relative
    # file:// images unless --allowed-local-media-path is set, so rewrite
    # "data/textvqa/train_images/..." -> absolute "IMG_DIR/..." paths. Also
    # collect the referenced filenames to verify/download the real images.
    if [ ! -d "$TEXTVQA_DIR/textvqa_json" ]; then
        echo "ERROR: textvqa/textvqa_json directory is missing after preparation"
        exit 1
    fi
    python3 - "$TEXTVQA_DIR/textvqa_json" "$IMG_DIR" << 'PYEOF'
import json, os, re, sys, glob
json_dir, img_dir = sys.argv[1], sys.argv[2]
names = set()
# Only the val dataset (textvqa_val.jsonl) is used by the textvqa_gen
# evaluation scenario. Process only it so the 'needed' image count
# reflects the actual eval subset, not the much larger train set.
for fp in glob.glob(os.path.join(json_dir, '*val*.jsonl')):
    lines = open(fp, encoding='utf-8').read().splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except Exception:
            out.append(ln)
            continue
        if isinstance(d, dict) and 'image' in d:
            names.add(os.path.basename(d['image']))
            d['image'] = os.path.join(img_dir, os.path.basename(d['image'])).replace('\\', '/')
            out.append(json.dumps(d, ensure_ascii=False))
        else:
            out.append(ln)
    open(fp, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
pat = re.compile(r'data/textvqa/train_images/([A-Za-z0-9_.-]+\.(?:jpg|jpeg|png|webp))')
for fp in glob.glob(os.path.join(json_dir, '*.json')):
    if fp.endswith('.bak'):
        continue
    txt = open(fp, encoding='utf-8').read()
    txt2 = pat.sub(lambda m: os.path.join(img_dir, m.group(1)).replace('\\', '/'), txt)
    if txt2 != txt:
        open(fp, 'w', encoding='utf-8').write(txt2)
with open(os.path.join(json_dir, '.image_list.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted(names)))
print('textvqa image paths absolutized; referenced images:', len(names))
PYEOF

    # A directory alone does not prove that the download is complete.
    python3 - "$IMG_DIR" "$TEXTVQA_DIR/textvqa_json/.image_list.txt" << 'PYEOF'
from pathlib import Path
import sys
img_dir, list_file = Path(sys.argv[1]), Path(sys.argv[2])
names = list_file.read_text(encoding='utf-8').splitlines()
if not names:
    raise SystemExit("ERROR: textvqa_val.jsonl is missing or has no image references")
missing = []
for name in names:
    image = img_dir / name
    if not image.is_file() or not image.stat().st_size:
        missing.append(name)
    else:
        with image.open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                missing.append(name)
if missing:
    raise SystemExit("ERROR: incomplete ModelScope textvqa images: " + ', '.join(missing[:10]))
list_file.unlink()
print('textvqa referenced images verified:', len(names))
PYEOF
    textvqa_complete
    echo "textvqa dataset ready at $TEXTVQA_DIR"
else
    echo "=== Downloading math dataset ==="
    DATASET_DIR="$AIS_BENCH_DIR/{dataset_rel}"
    mkdir -p "$DATASET_DIR"
    cd "$DATASET_DIR"
    if [ ! -f "math.zip" ]; then
        if command -v wget >/dev/null 2>&1; then
            echo "Downloading {math_url} (wget) ..."
            wget -q "{math_url}" -O math.zip 2>&1 || {{ echo "ERROR: wget download failed"; exit 1; }}
        elif command -v curl >/dev/null 2>&1; then
            echo "Downloading {math_url} (curl) ..."
            curl -sS -o math.zip "{math_url}" 2>&1 || {{ echo "ERROR: curl download failed"; exit 1; }}
        else
            echo "ERROR: neither wget nor curl is available on the remote host"
            exit 1
        fi
        if [ ! -s math.zip ]; then
            echo "ERROR: math.zip is empty — download produced no data (network/proxy issue?)"
            exit 1
        fi
        SIZE=$(wc -c < math.zip)
        echo "math.zip downloaded ($SIZE bytes)"
    fi

    # Extract only if the target dir is missing. Prefer unzip; fall back to
    # python3 zipfile (python3 is guaranteed present for ais_bench) when the
    # container lacks the unzip binary.
    if [ ! -d "math" ] && [ -f "math.zip" ]; then
        if command -v unzip >/dev/null 2>&1; then
            unzip -o math.zip >/dev/null 2>&1 || {{ echo "ERROR: unzip failed to extract math.zip"; exit 1; }}
        else
            echo "unzip not found on remote; extracting with python3 zipfile ..."
            python3 -c "import zipfile; zipfile.ZipFile('math.zip').extractall('.')" 2>&1 || \
                {{ echo "ERROR: python3 zipfile extraction failed"; exit 1; }}
        fi
        if [ ! -d "math" ]; then
            echo "ERROR: extraction did not produce a 'math' directory — math.zip may be corrupt"
            exit 1
        fi
        echo "math.zip extracted"
    fi

    if [ ! -d "math" ]; then
        echo "ERROR: dataset directory 'math' is missing after download+extract"
        exit 1
    fi
    echo "Dataset ready at $DATASET_DIR"
fi

echo "=== Config file location ==="
CONFIG_FILE="$AIS_BENCH_DIR/{config_rel}"
echo "Config file: $CONFIG_FILE"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found at $CONFIG_FILE"
    exit 1
fi

echo "=== Modifying config ==="
python3 - "$CONFIG_FILE" "{model_path}" "{host_port}" "{batch_size}" "{temperature}" "{seed}" "{max_out_len}" << 'PYEOF'
import re, sys
config_file = sys.argv[1]
model_path = sys.argv[2]
host_port = sys.argv[3]
batch_size = sys.argv[4]
temperature = sys.argv[5]
seed_val = sys.argv[6]
max_out_len = sys.argv[7]

with open(config_file, 'r') as f:
    content = f.read()

def _replace_once(text, pattern, repl, *, flags=0, key):
    """Replace *pattern* exactly once in *text*, failing loudly if the
    config did not match. A silently-unedited config would otherwise
    report accuracy against stale settings (old model path/port/...), so
    both no-match and multi-match abort the edit."""
    updated, count = re.subn(pattern, repl, text, flags=flags)
    if count != 1:
        sys.exit('ERROR: expected exactly one "%s" entry in %s (matched %d); '
                 'config edit aborted' % (key, config_file, count))
    return updated

# Update path (preserve an existing trailing comma, or add one so the
# dict stays valid even if a previous edit stripped it)
content = _replace_once(
    content,
    r'^(\s*)path\s*=\s*.+?(,?)\s*$',
    lambda m: m.group(1) + 'path = "' + model_path + '"' + (m.group(2) or ','),
    flags=re.MULTILINE,
    key='path',
)

# Force an explicit IPv4 loopback address.  ``localhost`` may resolve only to
# ``::1`` while the vLLM service listens on IPv4, causing connection refused.
content = _replace_once(
    content,
    r'^(\s*)host_ip\s*=\s*.+?(,?)\s*$',
    lambda m: m.group(1) + 'host_ip = "127.0.0.1"' + (m.group(2) or ','),
    flags=re.MULTILINE,
    key='host_ip',
)

# Update host_port (a bare port number; base_api builds the URL itself)
content = _replace_once(
    content,
    r'^(\s*)host_port\s*=\s*.+?(,?)\s*$',
    lambda m: m.group(1) + 'host_port = ' + host_port + (m.group(2) or ','),
    flags=re.MULTILINE,
    key='host_port',
)

# Update batch_size (preserve an existing trailing comma, or add one)
content = _replace_once(
    content,
    r'^(\s*)batch_size\s*=\s*\d+(,?)\s*$',
    lambda m: m.group(1) + 'batch_size = ' + batch_size + (m.group(2) or ','),
    flags=re.MULTILINE,
    key='batch_size',
)

# Update max_out_len (long enough for long reasoning outputs)
content = _replace_once(
    content,
    r'^(\s*)max_out_len\s*=\s*\d+(,?)\s*$',
    lambda m: m.group(1) + 'max_out_len = ' + max_out_len + (m.group(2) or ','),
    flags=re.MULTILINE,
    key='max_out_len',
)

# Update generation_kwargs block (preserve any trailing comma after the dict)
content = _replace_once(
    content,
    r'generation_kwargs\s*=\s*dict\s*\([^)]*\)(,?)',
    lambda m: 'generation_kwargs = dict(\n            temperature = ' + temperature + ',\n            seed = ' + seed_val + ',\n        )' + m.group(1),
    key='generation_kwargs',
)

with open(config_file, 'w') as f:
    f.write(content)

print("Config updated successfully")
PYEOF

echo "=== Environment preparation complete ==="
'''


def build_env_prep_script(
    *,
    host_port: str,
    model_path: str,
    batch_size: int = 1024,
    temperature: float = 0,
    seed: int = 1234,
    max_out_len: int = 20480,
    conda_env: str | None = None,
    proxy_bash_preamble: str = "",
    runtime_env_preamble: str = "",
    git_insecure: bool = False,
    dataset_kind: str = "auto",
    log_file: str = "",
) -> str:
    """Build the remote environment preparation shell script.

    ``host_port`` is the bare port number (e.g. ``20006``); ais_bench's
    ``base_api`` builds the request URL as ``protocol://host_ip:host_port/``
    itself (``host_ip`` is fixed to ``127.0.0.1`` because ais_bench runs on
    the head node and must not depend on ``localhost`` IPv4/IPv6 resolution).

    When ``conda_env`` is set, the script activates the conda environment
    before installing packages and running commands, ensuring ``pip`` and
    ``ais_bench`` operate within the correct Python environment.
    """
    log_redirect = ""
    if log_file:
        log_redirect = (
            "LOG_DST=" + shlex.quote(log_file) + "\n"
            "mkdir -p \"$(dirname \"$LOG_DST\")\"\n"
            "exec > >(tee -a \"$LOG_DST\") 2>&1\n"
        )

    conda_preamble = ""
    if conda_env:
        conda_preamble = (
            f'eval "$(conda shell.bash hook)" 2>/dev/null\n'
            f"conda activate {shlex.quote(conda_env)} 2>/dev/null\n"
            f'echo "conda env activated: {shlex.quote(conda_env)}"'
        )

    return _ENV_PREP_SCRIPT_TEMPLATE.format(
        conda_preamble=conda_preamble,
        proxy_preamble=proxy_bash_preamble,
        runtime_env_preamble=runtime_env_preamble,
        log_redirect=log_redirect,
        clone_dir=AIS_BENCH_CLONE_DIR,
        git_url=AIS_BENCH_GIT_URL,
        math_url=DEFAULT_MATH_URL,
        textvqa_dataset_id=TEXTVQA_DATASET_ID,
        dataset_rel=DATASET_DIR_REL_PATH,
        config_rel=CONFIG_REL_PATH,
        model_path=model_path,
        host_port=host_port,
        batch_size=str(batch_size),
        temperature=str(temperature),
        seed=str(seed),
        max_out_len=str(max_out_len),
        git_insecure="1" if git_insecure else "0",
        dataset_kind=dataset_kind,
    )


def run_env_prep(config: AccuracyConfig, host_port: str) -> bool:
    """Run environment preparation on all nodes (ais-bench install, dataset
    download, config modification) using the given API service ``host_port``
    (bare port number). ``host_ip`` in the config is fixed to ``127.0.0.1``.

    Returns True on success; on failure prints a JSON error and returns False.
    """
    for target in config.all_targets():
        node_label = "head" if target is config.target else f"worker {target.alias}"
        prep_ts = now_utc().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        prep_log_file = f"{target.runtime_base}/aisbench/prep_{prep_ts}_{target.alias}.log"
        prep_script = build_env_prep_script(
            host_port=host_port,
            log_file=prep_log_file,
            model_path=config.model,
            batch_size=config.batch_size,
            temperature=config.temperature,
            seed=config.seed,
            max_out_len=config.max_out_len,
            conda_env=target.conda_env,
            proxy_bash_preamble=config.proxy_bash_preamble(),
            runtime_env_preamble=ascend_env_preamble(config.cann_dir),
            git_insecure=config.git_insecure,
            dataset_kind=config.dataset_kind,
        )
        emit_progress(
            "env-prep",
            f"preparing remote environment on {node_label} ({target.alias})",
            node=target.alias,
        )
        prep_result = run_remote_cmd(target, prep_script)
        if prep_result["returncode"] != 0:
            print_json({
                "status": "failed",
                "phase": "environment-preparation",
                "node": target.alias,
                "error": f"remote environment preparation failed on {node_label}",
                "stderr_tail": prep_result.get("stderr", "")[-1000:],
                "stdout_tail": prep_result.get("stdout", "")[-500:],
                "log_file": prep_log_file,
            })
            return False
        emit_progress(
            "env-prep",
            f"environment ready on {node_label} ({target.alias})",
            node=target.alias,
        )
    return True


# ---------------------------------------------------------------------------
# Local resource and network preflight
# ---------------------------------------------------------------------------

_RESOURCE_CHECK_TEMPLATE = r'''set +e
{runtime_env_preamble}
{conda_preamble}
_PKG=$(pip list 2>/dev/null | awk '{{print $1}}' | grep -iE '^ais[_ -]?bench[_ -]?benchmark$' | head -1)
if [ -n "$_PKG" ]; then
    echo "aisbench_package=ok"
else
    echo "aisbench_package=missing"
fi
LOCATION=$(pip show "$_PKG" 2>/dev/null | grep -E "^Location" | awk '{{print $2}}')
EDITABLE_LOC=$(pip show "$_PKG" 2>/dev/null | grep -E "^Editable project location" | awk -F': ' '{{print $2}}')
AIS_BENCH_DIR=""
for _base in "$EDITABLE_LOC" "$LOCATION"; do
    if [ -n "$_base" ] && [ -d "$_base/ais_bench" ]; then
        AIS_BENCH_DIR="$_base/ais_bench"
        break
    fi
    _FOUND=$(find "$_base" -maxdepth 3 -type d -name ais_bench 2>/dev/null | head -1)
    if [ -n "$_FOUND" ]; then
        AIS_BENCH_DIR="$_FOUND"
        break
    fi
done
if [ -n "$AIS_BENCH_DIR" ] && [ -f "$AIS_BENCH_DIR/{config_rel}" ]; then
    echo "aisbench_directory=ok"
else
    echo "aisbench_directory=missing"
fi
if [ "{dataset_kind}" = "math" ]; then
    if python3 -c "import math_verify, latex2sympy2_extended" >/dev/null 2>&1; then
        echo "evaluation_dependencies=ok"
    else
        echo "evaluation_dependencies=missing"
    fi
    if [ -n "$AIS_BENCH_DIR" ] && [ -d "$AIS_BENCH_DIR/{dataset_rel}/math" ]; then
        echo "dataset=ok"
    else
        echo "dataset=missing"
    fi
else
    echo "evaluation_dependencies=ok"
    if [ -n "$AIS_BENCH_DIR" ] && [ -d "$AIS_BENCH_DIR/{dataset_rel}/textvqa/textvqa_json" ]; then
        echo "dataset=ok"
    else
        echo "dataset=missing"
    fi
fi
'''


def build_resource_check_script(
    *,
    dataset_kind: str,
    conda_env: str | None = None,
    runtime_env_preamble: str = "",
) -> str:
    """Build a non-mutating check for reusable AISBench resources."""
    conda_preamble = ""
    if conda_env:
        conda_preamble = (
            'eval "$(conda shell.bash hook)" 2>/dev/null\n'
            f"conda activate {shlex.quote(conda_env)} 2>/dev/null\n"
        )
    return _RESOURCE_CHECK_TEMPLATE.format(
        runtime_env_preamble=runtime_env_preamble,
        conda_preamble=conda_preamble,
        dataset_kind=dataset_kind,
        config_rel=CONFIG_REL_PATH,
        dataset_rel=DATASET_DIR_REL_PATH,
    )


def run_resource_preflight(config: AccuracyConfig) -> dict[str, Any]:
    """Check whether every target can evaluate without downloading anything."""
    nodes: list[dict[str, Any]] = []
    all_ready = True
    required_markers = (
        "aisbench_package=ok",
        "aisbench_directory=ok",
        "evaluation_dependencies=ok",
        "dataset=ok",
    )
    for target in config.all_targets():
        emit_progress("resources", f"checking reusable resources on {target.alias}")
        result = run_remote_cmd(
            target,
            build_resource_check_script(
                dataset_kind=config.dataset_kind,
                conda_env=target.conda_env,
                runtime_env_preamble=ascend_env_preamble(config.cann_dir),
            ),
        )
        combined = result.get("stdout", "") + result.get("stderr", "")
        missing = [marker.removesuffix("=ok") for marker in required_markers if marker not in combined]
        ready = result["returncode"] == 0 and not missing
        all_ready = all_ready and ready
        node = {"node": target.alias, "ready": ready, "missing": missing}
        nodes.append(node)
        emit_progress("resources", f"{target.alias}: {node}", **node)
    return {"status": "ready" if all_ready else "incomplete", "nodes": nodes}

_NETCHECK_TEMPLATE = r'''set -e
{proxy_bash_preamble}echo "=== network preflight git: {git_url} ==="
GIT_INSEC_OPT=""
if [ "{git_insecure}" = "1" ]; then
    GIT_INSEC_OPT="-k"
fi
NETCHECK_TIMEOUT={timeout}
if command -v curl >/dev/null 2>&1; then
    if curl -sS -o /dev/null -m "$NETCHECK_TIMEOUT" $GIT_INSEC_OPT -I "{git_url}" >/dev/null 2>&1; then
        echo "git_endpoint=ok"
    elif curl -sS -o /dev/null -m "$NETCHECK_TIMEOUT" -k -I "{git_url}" >/dev/null 2>&1; then
        echo "git_endpoint=ssl-only"
    else
        echo "git_endpoint=fail"
    fi
elif command -v wget >/dev/null 2>&1; then
    if wget --spider -T "$NETCHECK_TIMEOUT" $GIT_INSEC_OPT -q "{git_url}" >/dev/null 2>&1; then
        echo "git_endpoint=ok"
    elif wget --spider -T "$NETCHECK_TIMEOUT" --no-check-certificate -q "{git_url}" >/dev/null 2>&1; then
        echo "git_endpoint=ssl-only"
    else
        echo "git_endpoint=fail"
    fi
else
    echo "git_endpoint=no-tool"
fi
echo "=== network preflight dataset: {math_url} ==="
if command -v curl >/dev/null 2>&1; then
    if curl -sS -o /dev/null -m "$NETCHECK_TIMEOUT" -I "{math_url}" >/dev/null 2>&1; then
        echo "dataset_endpoint=ok"
    else
        echo "dataset_endpoint=fail"
    fi
elif command -v wget >/dev/null 2>&1; then
    if wget --spider -T "$NETCHECK_TIMEOUT" -q "{math_url}" >/dev/null 2>&1; then
        echo "dataset_endpoint=ok"
    else
        echo "dataset_endpoint=fail"
    fi
else
    echo "dataset_endpoint=no-tool"
fi
'''

def build_network_check_script(
    *,
    proxy_bash_preamble: str = "",
    git_insecure: bool = False,
    timeout_seconds: int = 20,
    dataset_kind: str = "math",
) -> str:
    """Build a remote shell script that checks connectivity to the
    ais-bench git repo and the math dataset URL.

    Honors proxy env vars (via ``proxy_bash_preamble``) when set. Prints
    ``git_endpoint=ok/fail/ssl-only/no-tool`` and
    ``dataset_endpoint=ok/fail/no-tool`` markers that the caller parses into
    per-endpoint reachability. ``git_endpoint=ssl-only`` means the host is
    reachable but TLS certificate verification fails (common with internal
    proxies using self-signed certs), so ``--git-insecure`` is required.
    """
    return _NETCHECK_TEMPLATE.format(
        proxy_bash_preamble=proxy_bash_preamble,
        git_insecure="1" if git_insecure else "0",
        git_url=AIS_BENCH_GIT_URL,
        math_url=("https://modelscope.cn/datasets/" + TEXTVQA_DATASET_ID
                  if dataset_kind == "textvqa" else DEFAULT_MATH_URL),
        timeout=timeout_seconds,
    )


def run_network_preflight(
    config: AccuracyConfig,
) -> dict[str, Any]:
    """Check network connectivity on every node (head + workers).

    Returns a dict with ``status`` ("ok" only when every node can reach both
    the git repo and the dataset; ``fail`` otherwise) plus per-node details.
    Failing nodes list which endpoint is unreachable (including the missing
    curl/wget case, which also blocks env preparation) so the agent can ask
    the user whether to configure http_proxy/https_proxy.
    """
    nodes: list[dict[str, Any]] = []
    any_fail = False
    proxy = config.proxy_bash_preamble()
    for target in config.all_targets():
        emit_progress(
            "network", f"checking connectivity on {target.alias}",
            node=target.alias,
        )
        script = build_network_check_script(
            dataset_kind=config.dataset_kind,
            proxy_bash_preamble=proxy,
            git_insecure=config.git_insecure,
        )
        result = run_remote_cmd(target, script)
        combined = result.get("stdout", "") + result.get("stderr", "")
        git_ok = "git_endpoint=ok" in combined
        git_ssl_only = "git_endpoint=ssl-only" in combined
        dataset_ok = "dataset_endpoint=ok" in combined
        git_state = "ok" if git_ok else ("ssl-only" if git_ssl_only else "fail")
        node: dict[str, Any] = {
            "node": target.alias,
            "git_endpoint": git_state,
            "dataset_endpoint": "ok" if dataset_ok else "fail",
        }
        if git_ssl_only:
            node["git_ssl_error"] = True
        if result["returncode"] != 0:
            node["ssh_failed"] = True
            node["error"] = result.get("stderr", "")[:500] or result.get("stdout", "")[-500:]
            node["reachable"] = False
            any_fail = True
        elif not git_ok or not dataset_ok:
            node["reachable"] = False
            any_fail = True
        else:
            node["reachable"] = True
        nodes.append(node)
        emit_progress("network", f"{target.alias}: {node}", **node)

    status = "ok" if not any_fail else "fail"
    return {"status": status, "nodes": nodes}


# ---------------------------------------------------------------------------
# Accuracy run helpers
# ---------------------------------------------------------------------------

def run_single_accuracy_round(
    target: ExecutionTarget,
    *,
    dataset: str,
    round_index: int,
    conda_env: str | None = None,
    proxy_unset_preamble: str = "",
    runtime_env_preamble: str = "",
) -> dict[str, Any]:
    """Execute one round of ais_bench and return parsed metrics."""
    conda_prefix = f"conda run -n {shlex.quote(conda_env)} " if conda_env else ""

    cmd = (
        f"cd {shlex.quote(AIS_BENCH_WORK_DIR)} && "
        f"{conda_prefix}ais_bench "
        f"--models vllm_api_stream_chat "
        f"--datasets {shlex.quote(dataset)}"
    )
    if proxy_unset_preamble:
        cmd = proxy_unset_preamble + cmd
    if runtime_env_preamble:
        cmd = runtime_env_preamble + cmd

    # ais_bench prints progress/info to stdout but the accuracy value is written
    # into a per-run summary file (not echoed). Capture the latest summary so the
    # parser below can extract the metric.
    cmd += (
        " && _LA=$(ls -dt " + shlex.quote(AIS_BENCH_WORK_DIR + "/outputs/default") + "/*/ | head -1)"
        " && echo '---AISBENCH_SUMMARY---'"
        " && cat ${_LA}summary/summary_*.txt 2>/dev/null || true"
    )

    emit_progress("accuracy", f"round {round_index} starting", round=round_index)

    result = run_remote_cmd(target, cmd)

    if result["returncode"] != 0:
        return {
            "round": round_index,
            "status": "failed",
            "error": result.get("stderr", "")[:500],
            "stdout_tail": result.get("stdout", "")[-500:],
        }

    raw_output = result.get("stdout", "") + result.get("stderr", "")

    # Attempt to extract accuracy metrics from ais_bench output
    metrics = _parse_aisbench_output(raw_output)
    metrics["round"] = round_index
    metrics["status"] = "ok"
    return metrics


def _parse_aisbench_output(output: str) -> dict[str, Any]:
    """Parse ais_bench output to extract accuracy-related metrics.

    Handles multiple output formats, preferring JSON when available,
    falling back to regex extraction.
    """
    metrics: dict[str, Any] = {}

    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*"accuracy"[^{}]*\}', output, re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if "accuracy" in parsed:
                metrics["accuracy"] = float(parsed["accuracy"])
            for key in ("total_samples", "correct_samples", "score", "pass_rate"):
                if key in parsed:
                    metrics[key] = parsed[key]
            return metrics
        except json.JSONDecodeError:
            pass

    # Try to find structured text output (e.g. "Accuracy: 0.85" or "accuracy: 85.0%")
    # Patterns return (matched_text, is_percentage) for proper conversion
    accuracy_patterns: list[tuple[str, bool]] = [
        (r'[Aa]ccuracy\s*["\']?\s*[:=]\s*([\d.]+)\s*%', True),   # "Accuracy: 85.0%" → percentage
        (r'[Aa]ccuracy\s*["\']?\s*[:=]\s*([\d.]+)', False),        # "Accuracy: 0.85" / 'accuracy': 94.8 → ratio
        (r'[Ss]core\s*[:=]\s*([\d.]+)', False),
        (r'pass\.rate\s*[:=]\s*([\d.]+)', False),
        (r'correct\.ratio\s*[:=]\s*([\d.]+)', False),
    ]

    for pattern, is_pct in accuracy_patterns:
        m = re.search(pattern, output)
        if m:
            try:
                val = float(m.group(1))
                # ais_bench summaries report accuracy as a percentage (e.g. 94.8);
                # text patterns may match either a ratio (<1) or a percentage (>1),
                # so normalize percentages (>1) to a 0..1 ratio.
                metrics["accuracy"] = val / 100.0 if (is_pct or val > 1.0) else val
                break
            except (ValueError, IndexError):
                continue

    # Try to find sample counts
    sample_patterns = [
        (r'[Tt]otal\s*(?:samples)?\s*[:=]\s*(\d+)', 'total_samples'),
        (r'[Cc]orrect\s*(?:samples)?\s*[:=]\s*(\d+)', 'correct_samples'),
    ]
    for pattern, key in sample_patterns:
        m = re.search(pattern, output)
        if m:
            try:
                metrics[key] = int(m.group(1))
            except (ValueError, IndexError):
                pass

    # If nothing found, record raw output tail for diagnosis
    if not metrics:
        metrics["raw_parse_failed"] = True
        metrics["output_tail"] = output[-2000:]

    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics across rounds."""
    accuracy_values = []
    for r in rounds:
        if r.get("status") == "ok" and "accuracy" in r:
            accuracy_values.append(r["accuracy"])

    aggregated: dict[str, Any] = {
        "total_rounds": len(rounds),
        "successful_rounds": len(accuracy_values),
        "failed_rounds": len(rounds) - len(accuracy_values),
    }

    if accuracy_values:
        aggregated["accuracy_mean"] = round(statistics.mean(accuracy_values), 6)
        aggregated["accuracy_stdev"] = round(statistics.stdev(accuracy_values), 6) if len(accuracy_values) > 1 else 0.0
        aggregated["accuracy_min"] = round(min(accuracy_values), 6)
        aggregated["accuracy_max"] = round(max(accuracy_values), 6)
    else:
        aggregated["accuracy_mean"] = None
        aggregated["accuracy_stdev"] = None
        aggregated["accuracy_min"] = None
        aggregated["accuracy_max"] = None

    return aggregated


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _report_target_suffix(config: AccuracyConfig) -> str:
    """Return a human-readable suffix describing the execution target."""
    if config.container:
        return f" (container: {config.container})"
    if config.conda_env:
        return f" (conda: {config.conda_env})"
    return ""


def generate_report(
    *,
    config: AccuracyConfig,
    rounds: list[dict[str, Any]],
    aggregated: dict[str, Any],
    duration_seconds: float,
    output_dir: Path,
) -> Path:
    """Generate a Markdown accuracy test report."""
    ts = now_utc().replace(":", "-")
    safe_host = (config.host or "unknown").replace(".", "-")
    filename = f"{ts}_{safe_host}_accuracy_report.md"
    report_path = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# AISBench 精度测试报告",
        "",
        f"> **执行时间**: {now_utc()}",
        f"> **总耗时**: {duration_seconds:.0f}s",
        f"> **目标机器**: {config.host}" + _report_target_suffix(config),
        f"> **模型**: {config.model}",
        f"> **配置**: TP={config.tp or 'N/A'}, DP={config.dp or 'N/A'}",
        f"> **数据集**: {config.dataset}",
        f"> **采样参数**: batch_size={config.batch_size}, temperature={config.temperature}, seed={config.seed}",
        f"> **测试轮数**: {config.runs}",
        "",
        "## 精度结果",
        "",
        "| 轮次 | 精度 (Accuracy) | 状态 |",
        "|------|-----------------|------|",
    ]

    for r in rounds:
        idx = r.get("round", "?")
        acc = r.get("accuracy", "N/A")
        if isinstance(acc, float):
            acc_str = f"{acc:.4%}"
        elif isinstance(acc, (int,)):
            acc_str = f"{acc:.4f}"
        else:
            acc_str = str(acc)
        status = r.get("status", "unknown")
        status_icon = "✅" if status == "ok" else "❌"
        lines.append(f"| {idx} | {acc_str} | {status_icon} {status} |")

    lines.append("")

    # Summary
    lines.extend([
        "## 汇总统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ])

    mean_val = aggregated.get("accuracy_mean")
    if mean_val is not None:
        lines.append(f"| 平均精度 | {mean_val:.4%} |")
        lines.append(f"| 标准差 | {aggregated['accuracy_stdev']:.6f} |")
        lines.append(f"| 最低精度 | {aggregated['accuracy_min']:.4%} |")
        lines.append(f"| 最高精度 | {aggregated['accuracy_max']:.4%} |")
    else:
        lines.append("| 平均精度 | N/A (无有效结果) |")

    lines.append(f"| 成功轮数 | {aggregated['successful_rounds']} / {aggregated['total_rounds']} |")

    lines.extend([
        "",
        "## 结论",
        "",
    ])

    if mean_val is not None:
        if mean_val >= 0.80:
            lines.append(f"- ✅ 精度达到预期水平（平均 {mean_val:.2%}）")
        else:
            lines.append(f"- ⚠️ 精度偏低（平均 {mean_val:.2%}），建议排查模型与服务配置")
        if aggregated.get("accuracy_stdev", 0) > 0.02:
            lines.append(f"- ⚠️ 多轮结果波动较大（std={aggregated['accuracy_stdev']:.4f}），建议检查服务稳定性")
        else:
            lines.append(f"- 多轮结果稳定（std={aggregated['accuracy_stdev']:.4f}）")
    else:
        lines.append("- ❌ 未能获取有效精度数据，请检查 ais_bench 安装和配置")

    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Split on bare -- for serve-args
    serve_args: list[str] = []
    own_argv: list[str] = argv
    if "--serve-args" in argv:
        idx = argv.index("--serve-args")
        own_argv = argv[:idx]
        serve_args = argv[idx + 1:]

    parser = argparse.ArgumentParser(
        description="Run AISBench accuracy evaluation on vLLM Ascend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- target ---
    parser.add_argument("--employee-id", required=True, help="个人工号；用于 serving 运行目录及 AISBench 准备日志")
    parser.add_argument("--host", help="SSH host IP; omitted in local mode")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--user", default="root", help="SSH user (default: root)")
    parser.add_argument("--password", help="SSH password")
    parser.add_argument("--key-file", help="SSH private key path")
    parser.add_argument("--container", help="Docker container name")
    parser.add_argument("--conda-env", help="Conda environment name (mutually exclusive with --container)")
    parser.add_argument("--mode", "--connection-mode", dest="connection_mode",
                       choices=["docker", "conda", "direct", "local"], default="docker",
                       help="连接模式: docker / conda / direct / local — 默认: docker")
    parser.add_argument("--cann-dir", default="/usr/local/Ascend", help="CANN root directory")

    # --- model / serving ---
    parser.add_argument("--model", required=True, help="Model path on remote")
    parser.add_argument("--tp", type=int, help="Tensor parallel size")
    parser.add_argument("--dp", type=int, help="Data parallel size")
    parser.add_argument("--extra-env", action="append", default=[], help="KEY=VALUE (repeatable)")
    parser.add_argument("--port", type=int, help="Force a specific port")
    # --- network / proxy (optional) ---
    parser.add_argument("--http-proxy", help="http_proxy for remote AISBench git/pip/dataset downloads, including ModelScope (e.g. http://host:port)")
    parser.add_argument("--https-proxy", help="https_proxy for remote AISBench git/pip/dataset downloads, including ModelScope (e.g. http://host:port)")
    parser.add_argument("--no-proxy", help="no_proxy bypass list (e.g. localhost,127.0.0.1,10.0.0.0/8)")
    parser.add_argument("--git-insecure", action="store_true",
                        help="skip TLS certificate verification for git (self-signed internal proxy certs)")

    # --- accuracy ---
    parser.add_argument("--runs", type=int, default=10, help="Number of accuracy rounds (default: 10)")
    parser.add_argument("--dataset", default="math500_gen_0_shot_cot_chat_prompt", help="Dataset name")
    parser.add_argument("--dataset-kind", default="auto", choices=["auto", "math", "textvqa"],
                        help="Dataset preparation kind: auto (infer from --dataset), math, or textvqa (default: auto)")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature (default: 0)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--max-out-len", type=int, default=20480,
                        help="ais_bench max_out_len in the config (default: 20480); must be smaller than serve --max-model-len")

    # ---- multi-node (optional) ----
    parser.add_argument("--worker-hosts", help="comma-separated worker IPs for multi-node deployment")
    parser.add_argument("--master-port", type=int, help="DP RPC port (default: auto-allocate)")
    parser.add_argument("--worker-key-file", help="SSH key for worker nodes (default: same as head)")
    parser.add_argument("--worker-password", help="SSH password for worker nodes (default: same as head)")
    parser.add_argument("--worker-container", help="Docker container for worker nodes (default: same as head)")
    parser.add_argument("--worker-conda-env", help="Conda env for worker nodes (default: same as head)")
    parser.add_argument("--worker-extra-env", action="append", default=None,
                       help="extra env vars for worker nodes (repeatable)")

    args = parser.parse_args(own_argv)

    # --- assemble config ---
    try:
        config = assemble_config(args, serve_args)
    except ValueError as exc:
        print_json({"status": "needs_input", "error": str(exc)})
        return 1

    start_time = time.monotonic()
    all_rounds: list[dict[str, Any]] = []

    # Validate that the serving max-model-len covers the benchmark max_out_len
    # plus a 2048-token input allowance. max-model-len is the total input+output budget,
    # while max_out_len is the per-request output cap; if max-model-len is too
    # small, long outputs get truncated and the server returns 400.
    _mout = config.max_out_len
    _mml = config.max_model_len()
    if _mml is not None and _mml < _mout + 2048:
        print_json({
            "status": "needs_input",
            "phase": "config-validation",
            "error": (
                f"serve --max-model-len ({_mml}) 不得小于 ais_bench 配置的 "
                f"max_out_len ({_mout}) + 2048（当前阈值 {_mout + 2048}）。"
                "max-model-len 是输入+输出的总长度上限，若小于 max_out_len+2048，"
                "长输出会被截断导致服务端返回 400。请增大 --max-model-len "
                f"（建议至少 {_mout + 2048}）或减小 --max-out-len 后重试。"
            ),
        })
        return 1

    try:
        # =====================================================================
        # Phase 0: Reuse local resources before requiring external network
        # =====================================================================
        emit_progress("phase", "local resource preflight")
        resources = run_resource_preflight(config)
        if resources["status"] == "ready":
            preflight = {
                "status": "skipped",
                "reason": "local_resources_ready",
                "nodes": resources["nodes"],
            }
            emit_progress("network", "external network preflight skipped: local resources ready")
        else:
            emit_progress("phase", "network preflight for missing resources")
            preflight = run_network_preflight(config)
            emit_progress("network", f"preflight status: {preflight['status']}")
            if preflight["status"] != "ok":
                print_json({
                    "status": "needs_network_proxy",
                    "phase": "network-preflight",
                    "message": (
                        "本地评测资源不完整，且无法访问 ais-bench git 仓库或数据集端点；"
                        "需要配置 http_proxy/https_proxy 后重试（见 --http-proxy/--https-proxy/--no-proxy）；"
                        "若端点可达但提示 SSL 证书校验失败（git_ssl_error），可追加 --git-insecure 跳过证书校验。"
                    ),
                    "resources": resources,
                    "network": preflight,
                })
                return 2

        # =====================================================================
        # Phase 1: Environment preparation (before starting the service)
        #   host_port can be computed from --port, so we can install ais-bench /
        #   download the dataset / patch the config before the (expensive) vLLM
        #   service is up. If --port is auto-allocated (None), we fall back to
        #   preparing after service startup once the real base_url is known.
        # =====================================================================
        emit_progress("phase", "environment preparation")

        if config.port:
            if not run_env_prep(config, str(config.port)):
                call_serve_stop(config)
                return 1
            emit_progress("env-prep", "environment ready on all nodes")

        # =====================================================================
        # Phase 2: Start service (get the real base_url for ais_bench config)
        # =====================================================================
        emit_progress("phase", "service startup")

        serve_output = call_serve_start(config)
        if serve_output.get("status") != "ready":
            print_json({
                "status": "failed",
                "phase": "service-startup",
                "error": "service failed to become ready",
                "serve_output": serve_output,
            })
            call_serve_stop(config)
            return 1

        base_url = serve_output.get("base_url", "")
        emit_progress("service", f"service ready at {base_url}", base_url=base_url)

        # Auto-allocated port: run environment preparation now that the real
        # base_url is known.
        if not config.port:
            # Auto-allocated port: derive host/port from the real base_url now
            # that the service is up, then prepare the remote environment.
            _bu = urllib.parse.urlsplit(base_url)
            _fb_port = str(_bu.port) if _bu.port else ""
            if not base_url or not _fb_port:
                print_json({
                    "status": "failed",
                    "phase": "environment-preparation",
                    "error": "cannot determine service base_url/port from serve_start output",
                })
                call_serve_stop(config)
                return 1
            if not run_env_prep(config, _fb_port):
                call_serve_stop(config)
                return 1
            emit_progress("env-prep", "environment ready on all nodes")


        # =====================================================================
        # Phase 3: Accuracy evaluation
        # =====================================================================
        emit_progress("phase", f"accuracy evaluation ({config.runs} rounds)")

        for i in range(1, config.runs + 1):
            round_result = run_single_accuracy_round(
                config.target,
                dataset=config.dataset,
                round_index=i,
                conda_env=config.conda_env,
                proxy_unset_preamble=config.proxy_unset_preamble(),
                runtime_env_preamble=ascend_env_preamble(config.cann_dir),
            )
            all_rounds.append(round_result)

            status = round_result.get("status", "unknown")
            acc = round_result.get("accuracy")
            acc_str = f"{acc:.4%}" if isinstance(acc, float) else "N/A"
            emit_progress(
                "accuracy",
                f"round {i}/{config.runs}: {status}, accuracy={acc_str}",
                round=i,
                status=status,
                accuracy=acc,
            )

        # =====================================================================
        # Phase 4: Stop service
        # =====================================================================
        emit_progress("phase", "stopping service")
        call_serve_stop(config)

        # =====================================================================
        # Phase 5: Aggregate and report
        # =====================================================================
        emit_progress("phase", "aggregating results and generating report")

        aggregated = aggregate_results(all_rounds)
        duration = time.monotonic() - start_time

        output_dir = ROOT / ".workspace-local" / "aisbench-accuracy"
        report_path = generate_report(
            config=config,
            rounds=all_rounds,
            aggregated=aggregated,
            duration_seconds=duration,
            output_dir=output_dir,
        )

        # Save raw results
        raw_path = write_local_result(
            output_dir,
            f"accuracy_raw_{now_utc().replace(':', '-')}.json",
            {
                "config": config.summary_dict(),
                "resources": resources,
                "network": preflight,
                "nodes": [t.alias for t in config.all_targets()],
                "rounds": all_rounds,
                "aggregated": aggregated,
                "duration_seconds": round(duration, 1),
                "report": str(report_path),
            },
        )

        # =====================================================================
        # Final output
        # =====================================================================
        output = {
            "status": "ok",
            "config": config.summary_dict(),
            "resources": resources,
            "network": preflight,
            "nodes": [t.alias for t in config.all_targets()],
            "rounds": all_rounds,
            "aggregated": aggregated,
            "duration_seconds": round(duration, 1),
            "report": str(report_path),
            "raw_result": str(raw_path),
        }
        print_json(output)
        return 0

    except Exception as exc:
        # Best-effort cleanup
        try:
            call_serve_stop(config)
        except Exception:
            pass

        print_json({
            "status": "failed",
            "error": str(exc),
            "rounds_completed": all_rounds,
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
