"""Implementation-language routing, separate from performance objective.

A native CANN dispatch is an Ascend C *target*, not proof of vendor source language.
Kernel name/core type alone cannot prove Triton or Ascend C.
"""
import re
from pathlib import Path

TRITON_PLUGIN = "triton-op-generator"
ASCENDC_PLUGIN = "tilelang2ascendc-ops-generator"
NATIVE = re.compile(r"^(aclnn|aclop|aclrt)|^(ascendc|ascend_c)(_|$)", re.I)
NON_COMPUTE = re.compile(r"^(hccl|hcom|aicpu|aclrt|memcpy|memset)", re.I)
TRITON = re.compile(r"(^|[^a-z0-9])triton([^a-z0-9]|$)", re.I)


def classify(name, mapping=None, text=""):
    mapping = mapping or {}
    if NON_COMPUTE.search(name):
        return {"backend": "excluded", "target": None, "evidence": "communication/runtime"}
    explicit = mapping.get("backend")
    if explicit is not None and explicit not in ("triton", "ascendc", "unknown"):
        raise ValueError(f"invalid backend for {name}: {explicit}")
    triton_source = bool(re.search(r"@(?:triton\.|triton\.runtime\.)jit\b", text))
    native_source = bool(re.search(r"\bAscendC::|\b__aicore__\b|\bGET_TILING_DATA\b", text))
    native_dispatch = bool(NATIVE.search(name))
    triton_name = bool(TRITON.search(name))
    # A wrapper file may contain both implementations; name or user mapping alone is not dispatch proof.
    if explicit == "unknown":
        return {"backend": "unknown", "target": None, "evidence": "explicit unknown"}
    if explicit and mapping.get("evidence"):
        conflict = ((explicit == "triton" and native_dispatch) or
                    (explicit == "ascendc" and triton_name))
        if not conflict:
            return {"backend": explicit, "target": explicit, "evidence": str(mapping["evidence"])}
        return {"backend": "unknown", "target": None, "evidence": "mapping conflicts with dispatch name"}
    if native_dispatch and not triton_name:
        return {"backend": "cann_native", "target": "ascendc", "evidence": "native CANN dispatch; implementation language not proven"}
    if triton_source and not native_source and not native_dispatch:
        # Frozen source has Triton code, but source-to-kernel binding still comes from source_map.
        return {"backend": "triton", "target": "triton", "evidence": "mapped source contains @triton.jit"}
    if native_source and not triton_source and not triton_name:
        return {"backend": "ascendc", "target": "ascendc", "evidence": "mapped source contains Ascend C kernel markers"}
    if triton_name and not native_source:
        return {"backend": "triton_hint", "target": "triton", "evidence": "Triton name hint; verify actual dispatch"}
    return {"backend": "unknown", "target": None, "evidence": "no conclusive implementation evidence"}


def route_plan(opportunities, candidates, source_map, sources, model_root):
    def resolve(name):
        m = source_map.get(name, {})
        text = ""
        if m.get("source"):
            source = Path(model_root, m["source"]).resolve()
            try:
                rel = str(source.relative_to(Path(model_root).resolve()))
            except ValueError:
                rel = None
            if rel in sources:
                text = Path(sources[rel]["snapshot"]).read_text(errors="replace")
        return classify(name, m, text)

    singles = {}
    for item in candidates:
        singles[item["name"]] = ("matmul_mfu" if item.get("route") == "matmul_mfu" else "latency", item)
    for item in opportunities.get("native_hotspots", []):
        singles.setdefault(item["name"], ("latency", item))
    for item in opportunities.get("matmul_mfu", []):
        singles[item["name"]] = ("matmul_mfu", item)

    def plugin(target):
        return TRITON_PLUGIN if target == "triton" else ASCENDC_PLUGIN if target == "ascendc" else None

    result = []
    for name, (objective, item) in singles.items():
        impl = resolve(name)
        target = impl["target"]
        if impl["backend"] == "triton":
            action = "optimize_existing"
            status = "TRITON_EXISTING_PATH"
        elif target == "ascendc":
            action = "optimize_existing" if impl["backend"] == "ascendc" else "generate_replacement"
            status = "NEEDS_REFERENCE_AND_REVIEW"
        else:
            action = "resolve_implementation"
            status = "NEEDS_IMPLEMENTATION_REVIEW"
        if impl["backend"] == "excluded":
            continue
        result.append({"name": name, "objective": objective, "implementation": impl,
                       "plugin": plugin(target), "action": action, "status": status,
                       "mean_op_ratio": item.get("mean_op_ratio"), "profile_evidence": item})

    for item in opportunities.get("vector_fusion", []):
        names = item["names"]
        impls = [resolve(name) for name in names]
        targets = {i["target"] for i in impls}
        confirmed = all(i["backend"] in ("triton", "ascendc", "cann_native") for i in impls)
        target = next(iter(targets)) if len(targets) == 1 and None not in targets and confirmed else None
        result.append({"name": " -> ".join(names), "members": names, "objective": "vector_fusion",
                       "implementations": impls, "plugin": plugin(target),
                       "action": "generate_fused" if target else "split_or_review",
                       "status": "NEEDS_DATAFLOW_REVIEW" if target else "MIXED_OR_UNKNOWN_REVIEW",
                       "mean_op_ratio": item.get("mean_op_ratio"), "profile_evidence": item})
    return result


def generation_prompt(item):
    target = item["plugin"]
    return (
        f"目标插件: {target}; action={item['action']}; objective={item['objective']}\n"
        "先确认实际 dispatch 和源码绑定，禁止仅凭 kernel 名或 Core 类型判断实现语言。\n"
        "只在当前任务输出目录准备真实 PyTorch Model/独立可运行 reference，"
        "冻结实际原推理 shape/dtype/stride/dispatch、前向及状态和性能基线；不能修改模型仓。\n"
        "融合须确认生产者消费者、外部消费者、别名、原地写入、KVCache 原地更新、量化尺度、跨请求隔离、图捕获和通信边界，"
        "不合法则报告并停止；不默认跨 Triton/Ascend C 融合。\n"
        "使用目标插件生成或优化对应语言实现；不静默转换后端。"
        "MatMul/GMM 以实际 FLOPs、相同形态 latency 和 dtype 峰值计算 MFU；"
        "Vector 以原始完整序列 latency、访存和 launch 数为目标。\n"
        "精度参考与性能分母分离。验证真实 forward、KVCache/state 及图模式、真实推理 shape 和完整原始序列；"
        "产出代码、性能/精度证据、接回模型的补丁建议。当前任务不能自动应用补丁，"
        "需要独立图级/Ascend C 工程集成门禁及整网回归。"
    )
