from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from alignment_utils import (  # noqa: E402
    input_sha256,
    normalize_fixed_input,
    normalize_logits_layout,
    select_output_tensor,
)
from compare_logits import build_report  # noqa: E402
from dump_transformers_logits import (  # noqa: E402
    TransformersAlignmentAdapter,
    _loading_list,
    copy_logits_to_cpu_fp32,
    load_weight_manifest,
    module_output_finite_checks_enabled,
    resolve_device_map,
    tensor_sha256,
)
from dump_mindspeed_fsdp2_logits import (  # noqa: E402
    _apply_override,
    _copy_logits_to_cpu_fp32 as copy_fsdp2_logits_to_cpu_fp32,
)
from make_fixed_input import build_fixed_input_payload, normalize_token_ids  # noqa: E402
from prepare_checkpoint_view import (  # noqa: E402
    infer_checkpoint_expert_count,
    parse_config_override,
    select_weight_keys,
)
from run_alignment import command_environment  # noqa: E402


class _FakeTokenizer:
    padding_side = "left"
    pad_token_id = None
    eos_token_id = 99
    bos_token_id = 1
    chat_template = "fake"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        length = len(messages[0]["content"])
        suffix = [7] if add_generation_prompt else []
        return {"input_ids": [[*range(1, length + 1), *suffix]]}

    def encode(self, prompt, *, add_special_tokens):
        prefix = [self.bos_token_id] if add_special_tokens else []
        return [*prefix, *range(10, 10 + len(prompt))]


def test_checkpoint_expert_count_and_explicit_config_override() -> None:
    weight_map = {
        f"layers.0.ffn.experts.{expert_id}.w1.weight": "weights.safetensors"
        for expert_id in range(4)
    }
    assert infer_checkpoint_expert_count(weight_map) == 4
    assert parse_config_override("n_routed_experts=256") == ("n_routed_experts", 256)

    del weight_map["layers.0.ffn.experts.2.w1.weight"]
    with pytest.raises(ValueError, match="contiguous"):
        infer_checkpoint_expert_count(weight_map)


def test_checkpoint_layer_selection_accepts_model_prefix() -> None:
    weight_map = {
        "model.embed_tokens.weight": "shared.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "layer-0.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "layer-1.safetensors",
        "layers.0.mlp.down_proj.weight": "legacy-layer-0.safetensors",
        "layers.2.mlp.down_proj.weight": "legacy-layer-2.safetensors",
        "model.mtp.weight": "mtp.safetensors",
        "lm_head.weight": "shared.safetensors",
    }

    selected = select_weight_keys(weight_map, layer_count=1, include_mtp=False)

    assert set(selected) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    }


def test_candidate_pythonpath_does_not_contaminate_reference(tmp_path: Path) -> None:
    args = argparse.Namespace(
        transformers_source=tmp_path / "transformers-source",
        repo_root=tmp_path / "repo",
        candidate_pythonpath=["/candidate/dependency"],
    )
    reference_path = command_environment(args, candidate=False)["PYTHONPATH"].split(":")
    candidate_path = command_environment(args, candidate=True)["PYTHONPATH"].split(":")
    assert "/candidate/dependency" not in reference_path
    assert "/candidate/dependency" in candidate_path


def test_fixed_input_contract_and_right_padding() -> None:
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    digest = input_sha256(input_ids, mask)

    actual_ids, actual_mask, actual_digest = normalize_fixed_input(
        {"input_ids": input_ids, "attention_mask": mask, "input_sha256": digest}
    )
    assert torch.equal(actual_ids, input_ids)
    assert torch.equal(actual_mask, mask)
    assert actual_digest == digest

    with pytest.raises(ValueError, match="right padding"):
        normalize_fixed_input(
            {"input_ids": input_ids, "attention_mask": torch.tensor([[1, 0, 1], [1, 1, 1]])}
        )


def test_output_selector_and_layout_normalization() -> None:
    input_ids = torch.ones((2, 3), dtype=torch.long)
    sbv = torch.arange(3 * 2 * 5).reshape(3, 2, 5)
    selected = select_output_tensor({"scores": sbv}, "key:scores")
    normalized = normalize_logits_layout(selected, input_ids, "sbv")
    assert normalized.shape == (2, 3, 5)
    assert torch.equal(normalized, sbv.transpose(0, 1))


def test_transformers_fixed_input_builder_normalizes_and_right_pads() -> None:
    tokenizer = _FakeTokenizer()
    payload, valid_lengths, truncated = build_fixed_input_payload(
        tokenizer,
        ["ab", "abcd"],
        tokenizer_dir="fake-tokenizer",
        max_length=4,
        use_chat_template=True,
        add_generation_prompt=True,
        add_special_tokens=True,
        fail_on_truncation=False,
    )
    assert valid_lengths == [3, 4]
    assert truncated == [False, True]
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert payload["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert payload["input_sha256"] == input_sha256(
        payload["input_ids"], payload["attention_mask"]
    )

    assert normalize_token_ids(torch.tensor([[1, 2, 3]])) == [1, 2, 3]
    with pytest.raises(ValueError, match="exceeds"):
        build_fixed_input_payload(
            tokenizer,
            ["abcd"],
            tokenizer_dir="fake-tokenizer",
            max_length=2,
            use_chat_template=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            fail_on_truncation=True,
        )


def test_transformers_fixed_input_builder_creates_exact_valid_sequence() -> None:
    payload, valid_lengths, truncated = build_fixed_input_payload(
        _FakeTokenizer(),
        ["ab"],
        tokenizer_dir="fake-tokenizer",
        max_length=8,
        use_chat_template=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        fail_on_truncation=False,
        exact_sequence_length=8,
    )

    assert payload["input_ids"].shape == (1, 8)
    assert payload["input_ids"].tolist() == [[10, 11, 10, 11, 10, 11, 10, 11]]
    assert payload["attention_mask"].tolist() == [[1] * 8]
    assert valid_lengths == [8]
    assert truncated == [False]
    assert payload["metadata"]["exact_sequence_length"] == 8
    assert payload["metadata"]["sequence_expanded"] == [True]


def test_transformers_manifest_and_tensor_digest(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"placeholder")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"model.weight":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    manifest = load_weight_manifest(str(tmp_path))
    assert manifest.keys == frozenset({"model.weight"})
    assert manifest.missing_files == ()
    TransformersAlignmentAdapter().validate_weight_manifest(manifest, None, argparse.Namespace())

    tensor = torch.tensor([[1.0, 2.0]])
    assert tensor_sha256(tensor) == tensor_sha256(tensor.clone())


def test_explicit_device_map_overrides_automatic_map() -> None:
    args = argparse.Namespace(
        device_map="balanced",
        device_map_json='{"model.layers.0": 1, "lm_head": 2}',
    )
    assert resolve_device_map(args) == {"model.layers.0": 1, "lm_head": 2}

    args.device_map_json = None
    assert resolve_device_map(args) == "balanced"
    args.device_map = "none"
    assert resolve_device_map(args) is None


def test_adapter_can_force_module_output_finite_checks() -> None:
    default_args = argparse.Namespace(diagnose_nonfinite=False)
    assert module_output_finite_checks_enabled(
        TransformersAlignmentAdapter(), default_args
    ) == (False, False)

    diagnostic_args = argparse.Namespace(diagnose_nonfinite=True)
    assert module_output_finite_checks_enabled(
        TransformersAlignmentAdapter(), diagnostic_args
    ) == (True, False)

    forced_adapter = TransformersAlignmentAdapter()
    forced_adapter.force_module_output_finite_checks = True
    assert module_output_finite_checks_enabled(forced_adapter, default_args) == (True, True)


def test_loading_info_collections_are_normalized() -> None:
    assert set(_loading_list({"missing_keys": {"b", "a"}}, "missing_keys")) == {"a", "b"}
    assert _loading_list({"unexpected_keys": ("x",)}, "unexpected_keys") == ["x"]
    assert _loading_list({}, "error_msgs") == []

    with pytest.raises(TypeError, match="must be a collection"):
        _loading_list({"missing_keys": "model.weight"}, "missing_keys")


def test_logits_copy_converts_on_cpu_by_token_chunks() -> None:
    source = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7).to(torch.bfloat16)
    actual = copy_logits_to_cpu_fp32(source, token_chunk_size=2)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert torch.equal(actual, source.float())

    with pytest.raises(ValueError, match="positive"):
        copy_logits_to_cpu_fp32(source, token_chunk_size=0)


def test_fsdp2_runner_override_and_logits_copy() -> None:
    root = argparse.Namespace(
        optimization=argparse.Namespace(use_fused_rmsnorm=False),
    )
    _apply_override(root, "optimization.use_fused_rmsnorm=true")
    assert root.optimization.use_fused_rmsnorm is True
    with pytest.raises(ValueError, match="Unknown config field"):
        _apply_override(root, "optimization.not_a_field=true")

    logits = torch.arange(2 * 3 * 4, dtype=torch.bfloat16).reshape(2, 3, 4)
    copied = copy_fsdp2_logits_to_cpu_fp32(logits, token_chunk_size=2)
    assert copied.dtype == torch.float32
    assert copied.device.type == "cpu"
    assert torch.equal(copied, logits.float())


def _arguments(reference: Path, candidate: Path, report: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reference_file=str(reference),
        candidate_file=str(candidate),
        reference_framework="framework-a",
        candidate_framework="framework-b",
        reference_label="golden",
        candidate_label="actual",
        skip_framework_check=False,
        candidate_execution_kind=None,
        minimum_candidate_world_size=1,
        require_candidate_fsdp_wrapped=False,
        report_file=str(report),
        threshold=0.98,
        topk=2,
        token_chunk_size=2,
        max_token_cosines_in_report=256,
    )


def test_generic_comparator_passes_identical_logits(tmp_path: Path) -> None:
    input_ids = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1, 1, 0]])
    digest = input_sha256(input_ids, mask)
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]]]
    )
    common = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "input_sha256": digest,
        "logits": logits,
        "metadata": {"case": "unit"},
    }
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save({**common, "framework": "framework-a"}, reference)
    torch.save({**common, "framework": "framework-b"}, candidate)

    report, passed = build_report(_arguments(reference, candidate, tmp_path / "report.json"))
    assert passed
    assert report["metrics"]["average_token_cosine"] == pytest.approx(1.0)
    assert report["reference"]["label"] == "golden"
    assert report["candidate"]["label"] == "actual"
    assert "hf" not in report and "mindspeed" not in report


def test_comparator_requires_real_fsdp2_training_execution(tmp_path: Path) -> None:
    input_ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(input_ids)
    digest = input_sha256(input_ids, mask)
    logits = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])
    common = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "input_sha256": digest,
        "logits": logits,
    }
    reference = tmp_path / "reference.pt"
    candidate = tmp_path / "candidate.pt"
    torch.save({**common, "framework": "framework-a", "metadata": {}}, reference)
    torch.save(
        {
            **common,
            "framework": "framework-b",
            "metadata": {
                "execution": {
                    "kind": "mindspeed-llm-fsdp2-training-forward",
                    "world_size": 2,
                    "fsdp_wrapped_module_count": 4,
                    "weight_load_missing_parameter_count": 0,
                }
            },
        },
        candidate,
    )
    args = _arguments(reference, candidate, tmp_path / "report.json")
    args.candidate_execution_kind = "mindspeed-llm-fsdp2-training-forward"
    args.minimum_candidate_world_size = 2
    args.require_candidate_fsdp_wrapped = True
    args.require_candidate_complete_weight_load = True
    report, passed = build_report(args)
    assert passed
    assert report["candidate_execution"]["world_size"] == 2

    payload = torch.load(candidate, weights_only=True)
    payload["metadata"]["execution"]["weight_load_missing_parameter_count"] = 1
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="missing_parameter_count"):
        build_report(args)

    payload = torch.load(candidate, weights_only=True)
    payload["metadata"] = {"execution": {"kind": "standalone-modeling", "world_size": 1}}
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="execution kind"):
        build_report(args)
