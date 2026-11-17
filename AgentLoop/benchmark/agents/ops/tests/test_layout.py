"""Layout invariants for the committed result tree.

These guard a class of bug that already bit once: ``sensitivity.py`` discovered
results with ``*/result.json``, so when results moved to
``results/ops/<case>/<harness>__<model>/result.json`` (one directory per run) the
tool silently reported "no ops result.json" instead of failing loudly. Every
consumer of the tree must agree on where results live, and the aggregator must be
able to see all of them.

The tests read the committed results, so they double as a check that what is in the
repository is what the tooling can actually consume.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.agents.ops import report, runner, sensitivity  # noqa: E402

RESULTS = ROOT / "benchmark" / "results"

pytestmark = pytest.mark.skipif(not RESULTS.is_dir(),
                                reason="committed results tree not present")


def _all_results() -> list[tuple[pathlib.Path, dict]]:
    return [(p, json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(RESULTS.glob("**/result.json"))]


def test_every_result_lives_in_a_per_run_directory():
    """``<mode or domain>/.../<run>/result.json`` — the run id is a directory.

    Flat ``<domain>/<case>/result.json`` can hold exactly one run per case, so a
    second harness or model would overwrite the first; the run id therefore has to
    be part of the path, and the file is always named ``result.json`` so the
    aggregator's single glob finds it.
    """
    found = _all_results()
    assert found, "no committed results at all"
    for p, d in found:
        assert p.name == "result.json", p
        rel = p.relative_to(RESULTS)
        assert len(rel.parts) >= 4, f"{rel} is not <tree>/<case>/<run>/result.json"
        assert isinstance(d.get("case"), str) and d["case"], p


def test_ops_results_are_one_run_directory_per_run():
    runs = sorted((RESULTS / "ops").glob("*/*/result.json"))
    assert len(runs) == 6, [str(p.relative_to(RESULTS)) for p in runs]
    for p in runs:
        d = json.loads(p.read_text(encoding="utf-8"))
        run_dir = p.parent.name
        assert run_dir == f"{d['harness']}__{d['model']}", (p, run_dir)


def test_report_glob_sees_every_result():
    """The aggregator reads `**/result.json`; nothing may hide from it."""
    seen = {p.resolve() for p in RESULTS.glob("**/result.json")}
    loaded = report.load_results(RESULTS)
    assert len(loaded) == len(seen), (len(loaded), len(seen))
    # every row must carry the two columns the cross-environment comparison uses
    assert all(r.get("harness") and r.get("model") for r in loaded)


def test_every_result_is_ranked_in_the_single_table():
    """Swapping harness or model is just another run — one table, no extra mode."""
    loaded = report.load_results(RESULTS)
    md = report.render_markdown(loaded)
    assert md.count("| # |") == 1, "exactly one ranked table"
    for r in loaded:
        assert str(r["case"]) in md
    # The table must also declare its own scope: with one *evaluated* harness it is
    # that harness's baseline and not a cross-harness comparison (review comment:
    # "harness 基本都是 dsh，cc 和 codex 是否需要单独补测").
    note = [line for line in md.splitlines() if line.startswith("> 覆盖范围")]
    assert len(note) == 1, note
    for r in loaded:
        assert str(r["harness"]) in note[0], (r["harness"], note[0])
    evaluated = {r["harness"] for r in loaded} - report.REFERENCE_HARNESSES
    if len(evaluated) == 1:
        assert "不是**跨 harness / 跨模型对比" in md
        assert "补测尚未进行" in md


def test_sensitivity_finds_all_ops_results():
    """Regression guard: this tool must not depend on the tree's depth."""
    res = sensitivity.load_ops_results(RESULTS / "ops")
    assert len(res) == 6, [r["case"] for r in res]
    assert {r["case"] for r in res} >= {"gelu", "relu", "rmsnorm", "matmul"}


def test_sensitivity_glob_is_recursive():
    src = pathlib.Path(sensitivity.__file__).read_text(encoding="utf-8")
    assert 'glob("**/result.json")' in src


def test_default_result_path_is_inside_the_tree_report_reads():
    """`runner score` without --out must land where report.py looks."""
    result = {"domain": "ops", "harness": "dsh", "model": "m"}
    p = runner.default_result_path(result, {"id": "gelu"}, pathlib.Path("outputs/gelu"))
    assert p == runner.RESULTS / "ops" / "gelu" / "dsh__m" / "result.json"
    # RESULTS must be the same tree the aggregator is pointed at
    assert runner.RESULTS.resolve() == RESULTS.resolve()


def test_no_two_results_describe_the_same_run():
    """A duplicated (domain, case, harness, model) would show up as a phantom row.

    Exactly this happened: one case had been scored once with an explicit ``--out``
    under a short directory name (``sdpa``) and later through the default path
    (``scaled-dot-product-attention``), so the report listed it twice with identical
    numbers. Two files for one run is always a mistake, never a result.
    """
    seen: dict[tuple, pathlib.Path] = {}
    dupes = []
    for p, d in _all_results():
        key = (d.get("domain"), d.get("case"), d.get("harness"), d.get("model"))
        if key in seen:
            dupes.append((key, str(seen[key].relative_to(RESULTS)),
                          str(p.relative_to(RESULTS))))
        seen[key] = p
    assert not dupes, dupes


def test_result_directory_matches_the_manifest_case_id():
    """The result path is derived from the manifest case id, not a display name."""
    ids = {c["id"] for c in runner.load_manifest()["cases"]}
    for p, d in _all_results():
        if d.get("domain") != "ops":
            continue
        case_dir = p.relative_to(RESULTS).parts[1]
        assert case_dir == d["case"], (case_dir, d["case"])
        assert case_dir in ids, case_dir
