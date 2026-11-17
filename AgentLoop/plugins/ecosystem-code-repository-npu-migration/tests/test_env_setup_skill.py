"""Prompt contract for artifact-free environment setup."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "skills" / "env-setup"


def test_environment_workflow_contract() -> None:
    text = (ENV / "SKILL.md").read_text(encoding="utf-8")
    for phrase in (
        "当前官方来源",
        "README*",
        "pyproject.toml",
        "<repo-slug>-npu-py<major><minor>-<repo-path-hash>",
        "BLOCKED_ENVIRONMENT",
        "BLOCKED_TARGET_INSTALL",
        "目标软件包从预期 checkout 导入",
        "import torch; torch.npu.set_device(0)",
        "唯一的 mandatory base gate",
        "不要创建环境产物",
    ):
        assert phrase in text
    assert "trusted resume" not in text
    assert "validate_env_setup_artifacts.py" not in text


def test_runtime_model_and_minimal_gate_scope() -> None:
    model = (ENV / "references" / "torch_npu_runtime_model.md").read_text(encoding="utf-8")
    assert "torch==2.10.0" in model
    assert "torch_npu==2.10.0.post4" in model
    assert "2.10.0 基础发布线" in model
    assert "必要条件，但不是充分条件" in model
    assert "当前官方兼容矩阵" in model
    assert "重新安装固定的官方匹配组合" in model

    base = (ENV / "references" / "base_runtime_policy.md").read_text(encoding="utf-8")
    assert "退出码为 0 是环境 PASS 标准" in base
    assert "forward/backward" in base
    assert "不是通用检查" in base
    assert "仅用于分布式范围" in base


def test_environment_resources_are_probe_only() -> None:
    assert not (ENV / "examples").exists()
    assert not (ENV / "references" / "env_setup_contract.md").exists()
    assert not (ENV / "scripts" / "validate_env_setup_artifacts.py").exists()
    assert {path.name for path in (ENV / "scripts").iterdir() if path.is_file()} == {
        "detect_npu_env.py", "hccl_smoke.py", "npu_smoke.py", "ray_npu_smoke.py",
    }
