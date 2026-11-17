#!/usr/bin/env python3
"""Render a Chinese report from a generic layerwise alignment run record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", required=True, type=Path)
    parser.add_argument("--report-file", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.run_record.read_text(encoding="utf-8"))
    lines = [
        f"# {record['model_label']} FSDP2 精度对齐报告",
        "",
        "## 验收合同",
        "",
        f"- 原生 modeling: {record['modeling_source']}",
        f"- 权重路径: `{record['model_dir']}`",
        f"- 代码提交: `{record['repo_commit']}`",
        f"- Transformers: `{record['transformers_version']}`",
        f"- Candidate 执行路径: `{record.get('execution_label', '未声明')}`",
        f"- Reference 设备: `{record['device']}`，Accelerate 映射: `{record.get('device_map_json') or record.get('device_map', 'none')}`，计算 dtype: `{record['dtype']}`",
        f"- Candidate FSDP2 配置: `{record.get('candidate_config', '未声明')}`",
        f"- Candidate distributed world size: `{record.get('candidate_world_size', '未声明')}`",
        f"- Candidate 配置覆盖: `{record.get('candidate_config_overrides', [])}`",
        f"- Checkpoint config 显式修正: `{record.get('checkpoint_config_overrides', [])}`",
        f"- 主门禁: 有效 token 的完整词表 logits 平均余弦 `>= {record['threshold']}`",
        f"- 固定输入 SHA-256: `{record.get('input_sha256', '未生成')}`",
        f"- 固定输入 shape: `{record.get('input_shape', '未生成')}`",
        f"- 精确序列长度: `{record.get('exact_sequence_length', '未设置')}`",
        f"- 有效 token 数: `{record.get('input_valid_token_count', '未生成')}`，全位置有效: `{record.get('input_all_tokens_valid', False)}`",
        "",
        "## 减层与推理结果",
        "",
        "| 层数 | checkpoint 视图 | Native | FSDP2 | 完整 logits 比较 | 平均 token 余弦 | 推理 forward |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stage in record["stages"]:
        metric = stage.get("average_token_cosine")
        metric_text = f"{metric:.8f}" if isinstance(metric, (int, float)) else "-"
        lines.append(
            "| {layer_count} | {checkpoint_view} | {reference} | {candidate} | {comparison} | {metric} | {inference} |".format(
                layer_count=stage["layer_count"],
                checkpoint_view=stage["checkpoint_view_status"],
                reference=stage["reference_status"],
                candidate=stage["candidate_status"],
                comparison=stage["comparison_status"],
                metric=metric_text,
                inference=stage["inference_status"],
            )
        )

    lines.extend(["", "## FSDP2 训练链路证明", ""])
    for stage in record["stages"]:
        execution = stage.get("candidate_execution", {})
        lines.extend(
            [
                f"### {stage['layer_count']} 层",
                "",
                f"- execution kind: `{execution.get('kind', '未生成')}`",
                f"- entrypoint: `{execution.get('entrypoint', '未生成')}`",
                f"- model factory: `{execution.get('model_factory', '未生成')}`",
                f"- parallel engine: `{execution.get('parallel_engine', '未生成')}`",
                f"- trainer forward: `{execution.get('trainer_forward', '未生成')}`",
                f"- world size / FSDP / EP: `{execution.get('world_size', '-')}` / `{execution.get('fsdp_size', '-')}` / `{execution.get('ep_size', '-')}`",
                f"- FSDP2 wrapped module count: `{execution.get('fsdp_wrapped_module_count', '未生成')}`",
                f"- model training mode: `{execution.get('model_training', '未生成')}`，Trainer loss 已计算: `{execution.get('loss_computed_by_trainer', '未生成')}`",
                f"- weight load missing parameter count: `{execution.get('weight_load_missing_parameter_count', '未生成')}`",
                "",
            ]
        )

    lines.extend(["## 数值指标明细", ""])
    lines.extend(
        [
            "| 层数 | 平均/最小/中位 token 余弦 | P01/P05 | Max Abs | Mean Abs | RMS | Top-1 | Top-5 overlap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for stage in record["stages"]:
        metrics = stage.get("metrics", {})
        if not metrics:
            lines.append(f"| {stage['layer_count']} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {layer} | {average:.8f} / {minimum:.8f} / {median:.8f} | "
            "{p01:.8f} / {p05:.8f} | {max_abs:.6f} | {mean_abs:.6f} | "
            "{rms:.6f} | {top1:.6f} | {top5:.6f} |".format(
                layer=stage["layer_count"],
                average=metrics["average_token_cosine"],
                minimum=metrics["minimum_token_cosine"],
                median=metrics["median_token_cosine"],
                p01=metrics["p01_token_cosine"],
                p05=metrics["p05_token_cosine"],
                max_abs=metrics["max_abs"],
                mean_abs=metrics["mean_abs"],
                rms=metrics["rms_error"],
                top1=metrics["top1_match_rate"],
                top5=metrics["top5_mean_overlap"],
            )
        )
    lines.append("")

    lines.extend(["## 环境与 checkpoint 审计", ""])
    audit = record.get("checkpoint_audit", {})
    lines.extend(
        [
            f"- 索引声明分片数: `{audit.get('index_shard_count', '未知')}`",
            f"- 物理缺失分片数: `{audit.get('missing_shard_count', '未知')}`",
            f"- 当前裁剪层实际需要的缺失分片数: `{audit.get('missing_required_shard_count', '未知')}`",
            f"- 当前裁剪层实际缺失分片: `{', '.join(audit.get('missing_required_shards', [])) or '无'}`",
            f"- Checkpoint config 显式修正: `{record.get('checkpoint_config_overrides', [])}`",
            "- checkpoint 视图只省略目标模型不会消费的层与 MTP 权重；不会用随机权重补齐缺失参数。",
        ]
    )
    for stage in record["stages"]:
        manifest = stage.get("checkpoint_manifest", {})
        if manifest:
            lines.append(
                f"- {stage['layer_count']} 层 checkpoint 专家数: "
                f"`{manifest.get('checkpoint_expert_count', '未知')}`"
            )
    lines.extend(["", "## 执行记录", ""])
    for stage in record["stages"]:
        lines.append(f"### {stage['layer_count']} 层")
        lines.append("")
        for command in stage.get("commands", []):
            lines.append(f"- `{command['name']}`: exit code `{command['returncode']}`，日志 `{command['log']}`")
        if stage.get("failure"):
            lines.append(f"- 最早失败: `{stage['failure']}`")
        lines.append("")

    passed = (
        record.get("status") == "PASS"
        if "status" in record
        else all(stage.get("comparison_status") == "PASS" for stage in record["stages"])
    )
    failed_stage = record.get("failed_stage")
    lines.extend(
        [
            "## 结论",
            "",
            f"- 总体状态: `{'PASS' if passed else 'FAIL/BLOCKED'}`。",
            f"- 最早失败阶段: `{failed_stage or '无'}`。",
            f"- `PASS` 只认证本报告记录的 `{record.get('execution_label', '未声明')}`、FSDP2 配置和并行规模；未列出的融合算子或并行组合需要单独记录。",
            "- Candidate 必须由 `train_fsdp2.py` 参数体系、`ModelFactory`、`MindSpeedParallelEngine` 和 `Trainer._compute_loss` 产生；直接调用适配 modeling 的 standalone forward 不具备 FSDP2 认证资格。",
            "- 理论模型分析与本报告的实测精度结论相互独立；本报告不把随机权重 smoke 当成 checkpoint 精度证明。",
        ]
    )
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report_file)


if __name__ == "__main__":
    main()
