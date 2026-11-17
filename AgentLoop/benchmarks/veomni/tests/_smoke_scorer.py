#!/usr/bin/env python3
"""评分器冒烟测试：构造 Omni 场景五阶段最小合规产物，库级调用评分器断言全 PASS。

被 score.py 的 --scenario 路径间接依赖，也验证产物契约（schemas/）与门禁表一致。
非 unittest 用例（文件名不以 test 开头），不会被 `unittest discover` 收集。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "scorer"))

import score  # noqa: E402

SCENARIO = BENCH / "scenarios" / "omni-qwen3-omni.yaml"

ARTIFACTS = {
    "environment_report.json": {
        "mode": "static",
        "hardware": {"chip": "910B", "cpu_arch": "aarch64", "os": "Linux", "device_count": 0},
        "software": {"python": "3.10", "veomni": "0.1.0", "pytorch": "2.1.0",
                      "torchnpu": "6.0.0", "cann": "8.0", "transformers": "4.46.0"},
        "smoke_test": {"executed": False, "passed": False, "error": "static mode"},
        "compatibility": [{"component": "cann", "status": "OK", "note": "spec matched"}],
        "issues": [],
        "suggestions": ["install torch_npu before NPU run"],
    },
    "model_inventory.json": {
        "model": "qwen3-omni-toy",
        "model_type": "omni",
        "structure": {"layers": 2, "hidden_size": 256, "params_total": "10M",
                       "params_active": None, "experts": None},
        "assets": {"config": "config.json", "modeling": ["modeling_qwen3_omni.py"],
                    "processor": "processor/", "tokenizer": "tokenizer/",
                    "weight_format": "safetensors"},
        "integration": {"patchgen_configs": ["gpu.yaml", "npu.yaml"],
                         "registry_points": ["model_registry"],
                         "trainer": "omni",
                         "data_fields": ["input_ids", "audio", "image"]},
        "operator_gaps": [],
        "parallel_needs": {"fsdp2": True, "ulysses_sp": False, "ep": False},
        "reference_model": "qwen3-omni",
        "workload": "medium",
        "risks": ["preview scenario"],
    },
    "patch_manifest.json": {
        "model": "qwen3-omni-toy",
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "patch_configs": [{"path": "patches/qwen3_omni/npu.yaml", "target": "npu"},
                           {"path": "patches/qwen3_omni/gpu.yaml", "target": "gpu"}],
        "generated_files": [{"path": "modeling_qwen3_omni.py", "sha256": "a" * 64}],
        "registry_items": ["qwen3_omni_toy"],
        "weight_conversion": {"script": "convert.py", "mapping_table": "mapping.json"},
        "training_configs": ["train_omni_toy.yaml"],
        "tests": {"load": "PASS", "forward": "PASS", "train": "NOT_RUN",
                   "commands": ["python -m tests.load_test"]},
    },
    "functional_report.json": {
        "model": "qwen3-omni-toy",
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "cases": [
            {"name": "build", "stage": "build", "executed": True, "status": "PASS",
             "evidence": "toy config built"},
            {"name": "forward", "stage": "forward", "executed": True, "status": "PASS",
             "evidence": "logits shape ok"},
            {"name": "backward", "stage": "backward", "executed": False, "status": "SKIPPED",
             "evidence": "preview scenario, min_steps=1"},
            {"name": "checkpoint", "stage": "checkpoint", "executed": False, "status": "SKIPPED",
             "evidence": "checkpoint=false for omni preview"},
        ],
        "conclusion": "PASS",
    },
    "golden_manifest.json": {
        "model": "qwen3-omni-toy",
        "code_revision": "0123456789abcdef0123456789abcdef01234567",
        "weights": "weights/toy.safetensors",
        "inputs": {"path": "fixtures/minimal-dataset.jsonl", "seq_len": 128, "batch_size": 1},
        "seeds": {"torch": 42, "python": 42, "numpy": 42},
        "dtype": "bf16",
        "training_config": "toy-omni.yaml",
        "generation": {"platform": "gpu", "artifacts": ["golden/layer0.pt"],
                        "command": "python gen_golden.py --platform gpu"},
    },
    "accuracy_report.json": {
        "model": "qwen3-omni-toy",
        "golden_manifest": "golden_manifest.json",
        "pairs": [{"baseline": "gpu", "target": "npu_single",
                    "threshold": {"metric": "cosine", "value": 0.98,
                                   "rationale": "omni preview, model-defined"},
                    "first_divergence": None, "nan_inf": False, "conclusion": "PASS"}],
        "parallel_impact": {"fsdp2": "N/A", "sp": "N/A", "ep": "N/A"},
        "regression": {"executed": True, "passed": True, "scope": "omni-preview unit"},
        "conclusion": "preview aligned",
    },
    "acceptance_report.json": {
        "model": "qwen3-omni-toy",
        "stages": [
            {"stage": "environment", "status": "PASSED", "artifacts": ["environment_report.json"]},
            {"stage": "model-analysis", "status": "PASSED", "artifacts": ["model_inventory.json"]},
            {"stage": "adaptation", "status": "PASSED", "artifacts": ["patch_manifest.json"]},
            {"stage": "functional-validation", "status": "PASSED", "artifacts": ["functional_report.json"]},
            {"stage": "precision-alignment", "status": "PASSED", "artifacts": ["accuracy_report.json"]},
            {"stage": "delivery", "status": "PASSED", "artifacts": ["delivery_coverage_report.md", "migration_process_record.md"]},
        ],
        "regression": {"passed": True, "scope": "omni preview unit"},
        "limitations": ["静态冒烟，不代表真实 NPU 训练"],
        "reproduce_commands": ["python benchmarks/veomni/tests/_smoke_scorer.py"],
        "conclusion": "PASS",
    },
}


def main() -> int:
    names = score.load_scenario_stages(SCENARIO)
    expected = ["environment", "model-analysis", "adaptation",
                "functional-validation", "precision-alignment", "delivery"]
    assert names == expected, f"场景阶段解析异常: {names}"

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for name, payload in ARTIFACTS.items():
            (base / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (base / "delivery_coverage_report.md").write_text(
            "# VeOmni 迁移交付件覆盖报告\n" * 40, encoding="utf-8"
        )
        (base / "migration_process_record.md").write_text(
            "# VeOmni 迁移过程记录\n" * 40, encoding="utf-8"
        )

        failures = []
        for stage in names:
            result = score.check_stage(base, score.STAGE_INDEX[stage])
            print(f"[{result['status']:>7}] {stage}")
            for a in result["artifacts"]:
                detail = ""
                if a["status"] == "SCHEMA_INVALID":
                    detail = " :: " + "; ".join(a.get("errors", []))
                print(f"           - {a['artifact']:<28} {a['status']}{detail}")
            if result["status"] != "PASSED":
                failures.append(stage)

        if failures:
            print(f"\n冒烟失败: {failures}")
            return 1
        print(f"\n评分器冒烟通过: Omni 场景 {len(names)} 阶段全部 PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
