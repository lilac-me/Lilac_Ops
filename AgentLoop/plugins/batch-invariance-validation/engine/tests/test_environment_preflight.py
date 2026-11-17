"""Hardware-free regression tests for environment admission and chip routing."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

SKILLS_ROOT = Path(__file__).resolve().parents[4] / "skills/inference-model-skills/batch-invariance-validation"
sys.path.insert(0, str(SKILLS_ROOT / "batch-invariance-environment/scripts"))
sys.path.insert(0, str(SKILLS_ROOT / "batch-invariance-operator-analysis/scripts"))
env = importlib.import_module("check_batch_invariant_env")
analyzer = importlib.import_module("analyze_op_statistic")
chip_detection = importlib.import_module("chip_detection")


@pytest.mark.parametrize("raw,family,platform", [
    ("Ascend910B", "A2", "ascend910b"),
    ("A2g3", "A2", "ascend910b"),
    ("Ascend910", "A3", "ascend910_93"),
    ("910_9391", "A3", "ascend910_93"),
    ("Ascend950", "A5", "ascend950"),
    ("A2", "A2", "ascend910b"),
    ("A3", "A3", "ascend910_93"),
    ("A5", "A5", "ascend950"),
    ("910b950", "A5", "ascend950"),
])
def test_chip_classification(raw, family, platform):
    assert env.classify_chip(raw) == family
    assert env.pick_platform(raw) == platform
    assert analyzer.soc_version_to_chip_family(raw) == family


@pytest.mark.parametrize("raw", ["", "unknown", "Ascend310"])
def test_unknown_chip_never_defaults_to_a3(raw):
    with pytest.raises(ValueError):
        env.classify_chip(raw)
    assert env.pick_platform(raw) is None


def test_a5_auto_install_uses_upstream_platform_and_build_environment(monkeypatch, tmp_path):
    script = tmp_path / "csrc/build_batch_invariant_ops.sh"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, "installed")

    monkeypatch.setattr(env.subprocess, "run", fake_run)
    ok, result = env.install_batch_invariant_ops(
        {"build_script": str(script), "build_platform": "ascend950"}, 30
    )
    assert ok is True
    assert result["command"][-1] == "ascend950"
    assert captured["kwargs"]["env"]["VLLM_BATCH_INVARIANT"] == "1"
    assert captured["kwargs"]["env"]["COMPILE_CUSTOM_KERNELS"] == "1"


def test_opp_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ASCEND_OPP_PATH", str(tmp_path / "environment"))
    assert env.resolve_opp_path(str(tmp_path / "explicit")) == (str(tmp_path / "explicit"), "--opp-path")
    assert env.resolve_opp_path() == (str(tmp_path / "environment"), "ASCEND_OPP_PATH")
    monkeypatch.setenv("ASCEND_OPP_PATH", "  ")
    assert env.resolve_opp_path()[1] == "default"


def test_a2_classification_does_not_change_allowlist():
    assert analyzer.resolve_chip_family("A2") == ("A2", "A2")
    assert analyzer.load_allowlist(analyzer.DEFAULT_ALLOWLIST, "A2") == analyzer.load_allowlist(analyzer.DEFAULT_ALLOWLIST, "A3")


@pytest.fixture
def mocked_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SOC_VERSION", "Ascend910")
    monkeypatch.setattr(env, "validate_tp_confirmation", lambda _: True)
    monkeypatch.setattr(env, "package_provenance", lambda name: {"distribution": name})
    monkeypatch.setattr(env, "show_recovery_guidance", lambda *_: None)
    so = tmp_path / "vendors/batch_invariant/op_api/lib/libcust_opapi.so"
    so.parent.mkdir(parents=True)
    so.write_bytes(b"mock")
    commands = []

    def run(command, timeout=None):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "mock success")

    monkeypatch.setattr(env, "run", run)
    return tmp_path, commands


def test_generic_910_ready_without_building(mocked_environment):
    root, commands = mocked_environment
    report = {}
    assert env.check_batch_invariant(str(root), 1, "config.json", report) == 0
    assert report["chip"] == "A3"
    assert report["installation_state"] == "READY"
    assert report["installation_capability"] == "SUPPORTED"
    assert not any(command[0] in {"bash", "pip"} for command in commands)
    probe = next(command for command in commands if "-c" in command)
    assert json.loads(probe[-1]) == list(env.REQUIRED_OPS)
    assert "ctypes.CDLL" in probe[2]


@pytest.mark.parametrize("chip,code,reason", [
    ("Ascend910", 1, "BLOCKED_OPERATOR_INSTALLATION"),
    ("Ascend950", 1, "BLOCKED_OPERATOR_INSTALLATION"),
])
def test_missing_so_blocks_even_when_package_is_installed(mocked_environment, monkeypatch, chip, code, reason):
    root, _ = mocked_environment
    monkeypatch.setenv("SOC_VERSION", chip)
    (root / "vendors/batch_invariant/op_api/lib/libcust_opapi.so").unlink()
    report = {}
    assert env.check_batch_invariant(str(root), 1, "config.json", report) == code
    assert report["reason_code"] == reason
    assert report["installation_state"] == "MISSING"


def test_unknown_chip_blocks_before_package_probe(mocked_environment, monkeypatch):
    root, commands = mocked_environment
    monkeypatch.setenv("SOC_VERSION", "unknown")
    report = {}
    assert env.check_batch_invariant(str(root), 1, "config.json", report) == 7
    assert report["reason_code"] == "BLOCKED_CHIP_DETECTION"
    assert not commands


def test_registration_timeout_is_broken(mocked_environment, monkeypatch):
    root, _ = mocked_environment
    original = env.run

    def run(command, timeout=None):
        if "-c" in command:
            raise subprocess.TimeoutExpired(command, timeout)
        return original(command, timeout)

    monkeypatch.setattr(env, "run", run)
    report = {}
    assert env.check_batch_invariant(str(root), 1, "config.json", report) == 1
    assert report["installation_state"] == "BROKEN"


def test_tp_gate_does_not_probe_hardware(mocked_environment, monkeypatch):
    root, commands = mocked_environment
    monkeypatch.setattr(env, "validate_tp_confirmation", lambda _: False)
    report = {}
    assert env.check_batch_invariant(str(root), 1, "config.json", report) == 6
    assert report["installation_state"] == "NOT_CHECKED"
    assert not commands


@pytest.mark.parametrize("missing", env.REQUIRED_OPS)
def test_probe_rejects_each_missing_required_operator(mocked_environment, monkeypatch, missing):
    root, commands = mocked_environment
    env.check_batch_invariant(str(root), 1, "config.json", {})
    probe = next(command for command in commands if "-c" in command)
    namespace = SimpleNamespace(**{name: SimpleNamespace(overloads=lambda: ["default"])
                                  for name in env.REQUIRED_OPS if name != missing})
    monkeypatch.setitem(sys.modules, "ctypes", SimpleNamespace(CDLL=lambda _: None))
    monkeypatch.setitem(sys.modules, "batch_invariant_ops", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(ops=SimpleNamespace(batch_invariant_ops=namespace)))
    monkeypatch.setattr(sys, "argv", ["probe", probe[3], probe[4]])
    with pytest.raises(SystemExit) as result:
        exec(probe[2], {})
    assert result.value.code == 1


def test_editable_source_build_script_is_found(tmp_path):
    script = tmp_path / "source/csrc/build_batch_invariant_ops.sh"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture", encoding="utf-8")
    output = f"Location: {tmp_path / 'site-packages'}\nEditable project location: {tmp_path / 'source'}"
    assert env.find_build_script(output) == script


def test_report_reused_by_analyzer(mocked_environment, monkeypatch, capsys):
    root, _ = mocked_environment
    monkeypatch.setenv("SOC_VERSION", "910B")
    report = {}
    env.check_batch_invariant(str(root), 1, "config.json", report)
    report_path = root / "preflight.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    csv = root / "op_statistic.csv"
    csv.write_text("op_type\nCast\n", encoding="utf-8")
    monkeypatch.setenv("SOC_VERSION", "unknown")
    monkeypatch.setattr(sys, "argv", ["analyze", str(csv), "--environment-report", str(report_path), "--json"])
    capsys.readouterr()
    assert analyzer.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["chip_type"] == "A2"
    assert result["allowlist_family"] == "A3"
    report["outcome"] = "BLOCKED"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert analyzer.main() == 4


def test_cli_writes_report_for_blocked_result(mocked_environment, monkeypatch):
    root, _ = mocked_environment
    destination = root / "environment/preflight.json"
    monkeypatch.setenv("SOC_VERSION", "unknown")
    monkeypatch.setattr(sys, "argv", ["preflight", "--run-config", "config.json", "--report-json", str(destination)])
    assert env.main() == 7
    assert json.loads(destination.read_text(encoding="utf-8"))["reason_code"] == "BLOCKED_CHIP_DETECTION"


def test_detect_only_never_checks_packages_or_writes_config(mocked_environment, monkeypatch, capsys):
    root, commands = mocked_environment
    monkeypatch.setattr(sys, "argv", ["preflight", "--detect-chip"])
    assert env.main() == 0
    assert json.loads(capsys.readouterr().out)["chip"] == "A3"
    assert not commands
    assert not (root / "run_config.json").exists()


@pytest.mark.parametrize("name,family", [("Ascend910", "A3"), ("950", "A5"), ("A2g3", "A2")])
def test_npu_smi_name_without_subtype_is_preserved(monkeypatch, name, family):
    monkeypatch.delenv("SOC_VERSION", raising=False)
    monkeypatch.setattr(chip_detection.subprocess, "check_output", lambda command:
                        b"NPU ID : 0" if command[-1] == "-l" else f"Chip Name : {name}".encode())
    raw, source = env.detect_chip()
    assert raw == name.lower()
    assert source == "npu-smi"
    assert env.classify_chip(raw) == family
