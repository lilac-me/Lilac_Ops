#!/bin/bash
# ============================================================================
# 组件体检：这个仓用了哪些「NPU 上必须先定方案才能跑通」的组件
#
# 用法：
#   bash scan_gpu_deps.sh <目标目录> > component_check.md
#
# 为什么只查这些：
#   一键迁移默认开启（训练入口注入 transfer_to_npu），
#   "cuda"/"nccl" 字面量与 API 实参由它在运行期兜底，不必提前逐处扫 ——
#   agent 开箱即开，跑不通再按报错定点修复。
#   真正要先定路线的是组件级的东西：它们不是改个字符串的事，会在 import 或
#   算子调用处硬失败，必须先决定「替换 / 规避 / 剥离」再动手。
#
#   每节末尾的箭头指向处理办法所在章节，本文不展开。
# ============================================================================

set -uo pipefail

ROOT="${1:-.}"

# 目标不存在必须硬失败：否则扫不到任何东西会打出一份「无组件」的空报告，
# 看起来像体检结论 —— 路径打错与真的干净无法区分（静默假阴性）。
if [ ! -d "$ROOT" ]; then
    echo "错误：目标目录不存在或不是目录：$ROOT" >&2
    exit 1
fi
ROOT_ABS=$(cd "$ROOT" && pwd -P)
# 排除本脚本所在的 skill 目录：脚本自身的判据字符串会被自己的规则扫到
SKILL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)

exec python3 - "$ROOT_ABS" "$SKILL_ROOT" <<'PY'
import datetime
import io, os, re, sys, tokenize

root = sys.argv[1]
skip_root = sys.argv[2] if len(sys.argv) > 2 else None

EXTS = {'.py', '.sh', '.yaml', '.yml', '.json', '.toml', '.txt', '.cfg'}
# 只排除纯技术性目录（版本控制、缓存、虚拟环境、构建产物），不排除任何
# 按仓习惯划分的目录 —— 不是每个仓都叫 tests/，也没有哪个目录天然可以跳过
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', 'build', 'dist',
             '.tox', '.mypy_cache', '.pytest_cache', 'site-packages'}

# 组件判据：命中 = 这个仓用了它，迁移时必须先定路线
COMPONENTS = [
    ("flash-attn",
     re.compile(r'''flash[-_]attn|flash_attention_2|_flash_attention_forward|is_flash_attn_2_available'''),
     "references/flow/1-migrate.md §3.4"),
    ("图模式（torch.compile / torch.jit / CUDA Graph）",
     re.compile(r'''torch\.compile|@torch\.compile|use_compile|jit_compile|torch\.jit\.(script|trace|load)|
                    cuda_graph|CUDAGraph|make_graphed_callables|set_compile_mode''', re.X),
     "references/flow/1-migrate.md §3.3"),
    ("TransformerEngine",
     re.compile(r'''transformer[-_]engine|te\.pytorch|from\s+te\s+import|import\s+te\b'''),
     "references/ref-models.md（TE 部分）"),
    ("自定义 CUDA 扩展 / 算子",
     re.compile(r'''CUDAExtension|cpp_extension|load_inline|ext_modules|torch\.utils\.cpp_extension'''),
     "references/flow/1-migrate.md §3.3"),
    ("vLLM / triton",
     re.compile(r'''from\s+vllm|import\s+vllm|from\s+triton|import\s+triton|triton\.language'''),
     "references/flow/1-migrate.md §3.3"),
    ("CUDA 诊断代码（NPU 上会硬失败，需删）",
     re.compile(r'''pynvml|nvml|nvidia-smi|libcudart|
                    torch\.cuda\.(temperature|power_draw|clock_rate)|sched_setaffinity''', re.X),
     "references/flow/1-migrate.md §3.3"),
]

# 注释行、docstring 内的行不是代码，必须剔除 —— 否则文档示例与说明文字会被当成真引用
CODE_TOKENS = {tokenize.NAME, tokenize.NUMBER, tokenize.OP}
_codelines = {}
MAX_FILES = 15  # 单节最多展开的文件数，超出只报计数


def code_lines(path):
    """该文件里带真实代码 token 的行号集合；None = 非 .py / 解析不了，不做剔除"""
    if path in _codelines:
        return _codelines[path]
    result = None
    if path.endswith('.py'):
        found = set()
        try:
            with open(path, 'rb') as fh:
                for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
                    if tok.type in CODE_TOKENS:
                        found.update(range(tok.start[0], tok.end[0] + 1))
            result = found
        except Exception:
            result = None
    _codelines[path] = result
    return result


hits = {name: {} for name, _, _ in COMPONENTS}  # 组件 -> {相对路径: [行号, ...]}
n_files = 0

for dirpath, dirnames, filenames in os.walk(root):
    if skip_root and (dirpath == skip_root or dirpath.startswith(skip_root + os.sep)):
        dirnames[:] = []
        continue
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if os.path.splitext(fn)[1] not in EXTS:
            continue
        path = os.path.join(dirpath, fn)
        n_files += 1
        rel = path.replace(root + '/', '')
        try:
            lines = open(path, errors='replace').read().split('\n')
        except OSError:
            continue
        cl = code_lines(path)
        for lineno, line in enumerate(lines, 1):
            if re.match(r'^\s*#', line):
                continue
            # echo 里提到组件名只是提示文字，.sh 用不了 tokenize 过滤字符串
            if re.match(r'^\s*echo\b', line):
                continue
            if cl is not None and lineno not in cl:
                continue
            for name, rx, _ in COMPONENTS:
                if rx.search(line):
                    hits[name].setdefault(rel, []).append(lineno)

out = ["# 组件体检报告", ""]
out.append(f"- 目标：{root}")
out.append(f"- 生成：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out.append(f"- 扫描：{n_files} 个文件（{' '.join('*' + e for e in sorted(EXTS))}）")
out.append("- 范围：已排除注释与 docstring 内的行")
out.append("- **扫描边界**：只做文本/正则匹配，**动态加载的组件扫不到**"
           "（`importlib.import_module(\"flash_attn\")`、由配置/环境变量决定的后端、"
           "被包在自家脚本里再调用的启动器）——本节未命中不等于不存在，"
           "仍要按 §3.1 第 1 步把入口追到底")
out.append("- 说明：只查 NPU 上必须先定方案的组件；cuda/nccl 字面量与 API 实参由一键迁移"
           "（transfer_to_npu）在运行期兜底，不在扫描范围")

found, missing = [], []
for name, _, ref in COMPONENTS:
    got = hits[name]
    total = sum(len(v) for v in got.values())
    if not total:
        missing.append(name)
        continue
    found.append(name)
    out.append("")
    out.append(f"## {name}  {total} 处 / {len(got)} 个文件  → {ref}")
    out.append("")
    out.append("```")
    for rel, linenos in sorted(got.items(), key=lambda kv: -len(kv[1]))[:MAX_FILES]:
        shown = ", ".join(str(n) for n in linenos[:8])
        more = f" (+{len(linenos) - 8})" if len(linenos) > 8 else ""
        out.append(f"{rel}: {shown}{more}")
    if len(got) > MAX_FILES:
        out.append(f"... (还有 {len(got) - MAX_FILES} 个文件)")
    out.append("```")

out.append("")
if not found:
    out.append("以上 6 类组件**均未命中**（不代表不存在——见上方「扫描边界」，"
               "动态加载与配置驱动的用法扫不到）。")
elif missing:
    out.append(f"未命中的组件（仅为文本匹配结果，非「本仓没有」）：{'、'.join(missing)}")

print("\n".join(out))
PY
