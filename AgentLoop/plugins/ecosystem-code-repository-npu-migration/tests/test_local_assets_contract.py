"""Prompt and deterministic contracts for the local asset inventory."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "local-assets"
SCRIPT = SKILL_ROOT / "scripts" / "local_asset_inventory.py"


def load_assets():
    spec = importlib.util.spec_from_file_location("local_asset_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def make_dataset(path: Path) -> None:
    (path / "data").mkdir(parents=True)
    (path / "data" / "train.parquet").write_bytes(b"dataset")


def test_skill_requires_one_validated_inventory_without_binding_artifact() -> None:
    text = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for phrase in (
        "将唯一的持久化清单写入 `.npu-agent/local_asset_inventory.json`",
        "读取 `models`、`datasets`、绝对 `path`、完整性 `evidence`",
        "运行脚本的 `query` 命令",
        "仅当存在一个完整匹配时返回 `FOUND`",
        "在真实示例命令中绑定选定路径",
        "不要另建 entry-to-asset 绑定产物、workflow 状态或 Python 集合作为第二套事实来源",
    ):
        assert phrase in text
    assert "asset_bindings.json" not in text
    assert "entry_asset_bindings.json" not in text


def test_inventory_discovers_complete_assets_and_queries_unique_ambiguous_and_missing(tmp_path: Path) -> None:
    assets = load_assets()
    models = tmp_path / "models"
    datasets = tmp_path / "datasets"
    first_model = models / "org-one" / "shared-model"
    second_model = models / "org-two" / "shared-model"
    dataset = datasets / "org" / "training-data"
    make_model(first_model)
    make_model(second_model)
    make_dataset(dataset)

    inventory = assets.build_inventory([models.resolve()], [datasets.resolve()])
    validation = assets.validate_inventory(inventory, [models.resolve()], [datasets.resolve()])

    assert inventory["status"] == "PASS"
    assert validation["status"] == "PASS"
    assert len(inventory["models"]) == 2
    assert len(inventory["datasets"]) == 1
    assert assets.query_inventory(inventory, "model", "org-one/shared-model")["status"] == "FOUND"
    assert assets.query_inventory(inventory, "model", "shared-model")["status"] == "AMBIGUOUS"
    assert assets.query_inventory(inventory, "dataset", "org/training-data")["selected"]["path"] == str(
        dataset.resolve()
    )
    assert assets.query_inventory(inventory, "dataset", "org/missing")["status"] == "MISSING"


def test_inventory_rejects_missing_shards_stale_evidence_and_truncation(tmp_path: Path) -> None:
    assets = load_assets()
    models = tmp_path / "models"
    good = models / "org" / "good"
    make_model(good)
    broken = models / "org" / "broken"
    broken.mkdir(parents=True)
    (broken / "config.json").write_text("{}\n", encoding="utf-8")
    (broken / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )

    inventory = assets.build_inventory([models.resolve()], [])
    assert [item["path"] for item in inventory["models"]] == [str(good.resolve())]

    stale = deepcopy(inventory)
    stale["models"][0]["evidence"] = ["config.json"]
    stale["inventory_digest"] = assets.inventory_digest(stale)
    stale_result = assets.validate_inventory(stale)
    assert stale_result["status"] == "BLOCKED"
    assert any("evidence is stale" in error for error in stale_result["errors"])

    truncated = deepcopy(inventory)
    truncated["truncated"] = True
    truncated["inventory_digest"] = assets.inventory_digest(truncated)
    truncated_result = assets.validate_inventory(truncated)
    assert truncated_result["status"] == "BLOCKED"
    assert "inventory scan is truncated" in truncated_result["errors"]
