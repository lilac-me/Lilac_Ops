# test_batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Batch invariance test with environment variable control."""

import json
import math
import os
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

import pytest
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['VLLM_VERSION'] = '0.24.0'


def _read_temperature() -> float:
    raw = os.getenv("VLLM_TEST_TEMPERATURE", "0.6")
    try:
        temperature = float(raw)
    except ValueError as exc:
        pytest.fail(f"VLLM_TEST_TEMPERATURE must be a number, got {raw!r}: {exc}")
    if not math.isfinite(temperature) or temperature < 0:
        pytest.fail(
            "VLLM_TEST_TEMPERATURE must be finite and non-negative, "
            f"got {raw!r}"
        )
    return temperature

def _build_exact_token_prompts(tokenizer, target_tokens: int, count: int):
    """构造 tokenizer 后长度严格等于 target_tokens 的 TokensPrompt。"""
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    templates = [
        "Question: What is the capital of France?\nAnswer: The capital of France is",
        "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
        "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
        "Once upon a time in a distant galaxy, there lived",
        "To implement a binary search tree in Python, first we need to",
        "Climate change is caused by several factors including",
    ]
    filler_text = (
        "This is an interesting topic that deserves more explanation."
    )
    filler_ids = tokenizer.encode(filler_text, add_special_tokens=False)
    if not filler_ids:
        raise RuntimeError("tokenizer produced no token IDs for the filler text")

    prompt_inputs = []
    prompt_previews = []
    prompt_token_ids = []
    for _ in range(count):
        prefix_ids = tokenizer.encode(
            random.choice(templates), add_special_tokens=False
        )
        ids = list(prefix_ids[:target_tokens])
        remaining = target_tokens - len(ids)
        if remaining:
            repeats = math.ceil(remaining / len(filler_ids))
            ids.extend((filler_ids * repeats)[:remaining])
        if len(ids) != target_tokens:
            raise RuntimeError(
                f"exact prompt construction failed: {len(ids)} != {target_tokens}"
            )
        prompt_token_ids.append(ids)
        prompt_inputs.append({"prompt_token_ids": ids})
        prompt_previews.append(
            tokenizer.decode(ids[: min(64, len(ids))], skip_special_tokens=False)
        )

    return prompt_inputs, prompt_previews, prompt_token_ids


def _cleanup_llm_resources(llm_holder: dict) -> None:
    """关闭本轮 engine/worker，并清理分布式状态与设备缓存。"""
    if llm_holder.get("cleanup_complete"):
        return
    llm = llm_holder.pop("llm", None)
    cleanup_errors = []
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", None)
            engine_core = getattr(engine, "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown(timeout=60)
                except TypeError:
                    shutdown()
        except Exception as exc:  # cleanup must not hide the primary test result
            cleanup_errors.append(f"engine shutdown: {exc}")
        finally:
            del llm
    try:
        cleanup_dist_env_and_memory()
    except Exception as exc:  # runner performs an independent NPU process gate
        cleanup_errors.append(f"distributed cleanup: {exc}")

    if cleanup_errors:
        print(
            "BATCH_INVARIANCE_CLEANUP_WARNING: " + "; ".join(cleanup_errors),
            flush=True,
        )
    else:
        llm_holder["cleanup_complete"] = True
        print("BATCH_INVARIANCE_CLEANUP_COMPLETE", flush=True)


def _extract_step_logprobs(request_output):
    """从 RequestOutput 提取 chosen-token logprob 和生成 token ID。"""
    if getattr(request_output, "outputs", None):
        inner = request_output.outputs[0]
        token_ids = list(inner.token_ids)
        if hasattr(inner, "logprobs") and inner.logprobs is not None:
            t = torch.tensor(
                [inner.logprobs[i][tid].logprob for i, tid in enumerate(token_ids)],
                dtype=torch.float32,
            )
            return t, token_ids
    return None, None


def _write_json_artifact(env_name: str, payload: dict) -> Path | None:
    """将当前 attempt 的结构化证据写入 runner 分配的独立路径。"""
    raw_path = os.getenv(env_name)
    if not raw_path:
        return None
    path = Path(raw_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_logprobs_bitwise_batch_invariance_bs1_vs_bsN(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """
    验证 batch 不变性：BS=1 与 BS=N 时，相同 prompt 应输出相同的
    token ID 序列和 chosen-token logprob。
    参数通过环境变量控制；采样参数仅泛化 temperature：
        VLLM_TEST_TP_SIZE          - TP 尺寸 (默认 2)
        VLLM_TEST_TEMPERATURE      - 采样温度 (默认 0.6；如 0、0.6)
        VLLM_TEST_MAX_TOKENS       - 最大生成 token 数 (默认 50)
        VLLM_TEST_MAX_PROMPT_NUM   - 每批最大 prompt 数量 (默认 64)
        VLLM_TEST_PROMPT_LEN_VARIATION - 每条 prompt 随机增加的长度上限 (默认 50)
        VLLM_TEST_SEED             - 随机种子 (默认 12345)
        VLLM_TEST_MODEL            - 模型路径 (默认 /home/data/Qwen3-1.7B)
    """
    if os.getenv("VLLM_BATCH_INVARIANT") != "1":
        pytest.fail(
            "BATCH_INVARIANT_MODE_REQUIRED: VLLM_BATCH_INVARIANT must be exactly 1"
        )

    # ---------- 从环境变量读取参数 ----------
    tp_size = int(os.getenv("VLLM_TEST_TP_SIZE", "1"))
    temperature = _read_temperature()
    max_tokens = int(os.getenv("VLLM_TEST_MAX_TOKENS", "1"))
    max_prompt_num = int(os.getenv("VLLM_TEST_MAX_PROMPT_NUM", "64"))
    max_prompt_len = int(os.getenv("VLLM_TEST_MAX_PROMPT_LEN", "20480"))
    seed = int(os.getenv("VLLM_TEST_SEED", "12345"))
    model_name = os.getenv("VLLM_TEST_MODEL", DEFAULT_MODEL)
    max_num_batched_tokens = int(os.getenv("VLLM_TEST_MAX_NUM_BATCHED_TOKENS", "65535"))
    enable_prefix_caching = os.getenv("VLLM_TEST_ENABLE_PREFIX_CACHING", "False").lower() in ("true", "1", "yes")
    prompt_len_variation = int(os.getenv("VLLM_TEST_PROMPT_LEN_VARIATION", "50"))
    logprobs = 1
    random.seed(seed)

    max_model_len = int(os.getenv("VLLM_TEST_MAX_MODEL_LEN", "22578"))
    if prompt_len_variation < 0:
        pytest.fail(
            "VLLM_TEST_PROMPT_LEN_VARIATION must be non-negative, "
            f"got {prompt_len_variation}"
        )
    worst_prompt_len = max_prompt_len + prompt_len_variation
    if worst_prompt_len + max_tokens > max_model_len:
        pytest.fail(
            "BLOCKED_PROMPT_LENGTH_EXCEEDS_MODEL_LIMIT: "
            f"max_prompt_len({max_prompt_len}) + variation({prompt_len_variation}) "
            f"+ max_tokens({max_tokens}) > max_model_len({max_model_len})"
        )

    # ---------- 启用 batch_invariant 模式（禁用 custom all-reduce）----------
    # 在 Ascend NPU 上，batch_invariant 模式通过环境变量 VLLM_BATCH_INVARIANT=1 启用
    # enable_batch_invariant_mode() 内部调用 CUDA-specific API，在 Ascend 上会失败，
    # 因此直接通过环境变量判断
    disable_custom_ar = True

    if disable_custom_ar:
        print(f"\n{'=' * 80}")
        print(f"BATCH INVARIANCE MODE: Disabling custom all-reduce (TP={tp_size})")
        print("COMPARE METRIC: token_ids + chosen-token logprob")
        print(f"{'=' * 80}\n")

    # ---------- 初始化 LLM ----------
    llm_holder = {}
    request.addfinalizer(lambda: _cleanup_llm_resources(llm_holder))
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            enable_prefix_caching=enable_prefix_caching,
            max_num_seqs=max_prompt_num,
            max_model_len=max_model_len,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            distributed_executor_backend="mp",
            max_num_batched_tokens=max_num_batched_tokens,
            async_scheduling=False,
            enable_chunked_prefill=False,
            block_size=128,
            disable_log_stats=False,
        )
        llm_holder["llm"] = llm
    except Exception as e:
        error_text = str(e)
        resource_markers = (
            "out of memory",
            "failed to allocate",
            "no available memory for the cache blocks",
        )
        if any(marker in error_text.lower() for marker in resource_markers):
            print(
                "BATCH_INVARIANCE_RESOURCE_BLOCKED: "
                + json.dumps(
                    {
                        "classification": "SKIPPED_TP_RESOURCE_UNAVAILABLE",
                        "tp": tp_size,
                        "batch_size": max_prompt_num,
                        "prompt_length": max_prompt_len,
                        "max_num_batched_tokens": max_num_batched_tokens,
                        "error": error_text,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            pytest.fail(
                "SKIPPED_TP_RESOURCE_UNAVAILABLE: LLM initialization could not "
                f"fit the requested TP={tp_size}, BS={max_prompt_num} scenario: "
                f"{error_text}"
            )
        pytest.fail(f"LLM initialization failed: {str(e)}")

    # ---------- 生成 prompts ----------
    tokenizer = llm.get_tokenizer()

    prompt_target_lengths = [
        max_prompt_len + random.randint(0, prompt_len_variation)
        for _ in range(max_prompt_num)
    ]

    all_prompts = []
    all_previews = []
    all_prompt_token_ids = []

    for target_len in prompt_target_lengths:
        p, previews, tids = _build_exact_token_prompts(tokenizer, target_len, 1)
        all_prompts.extend(p)
        all_previews.extend(previews)
        all_prompt_token_ids.extend(tids)

    prompt_token_lengths = [len(ids) for ids in all_prompt_token_ids]
    if prompt_token_lengths != prompt_target_lengths:
        pytest.fail(
            "BLOCKED_EXACT_PROMPT_LENGTH: generated token lengths do not match "
            f"the requested lengths: actual={prompt_token_lengths}, "
            f"expected={prompt_target_lengths}"
        )
    if len(all_prompts) != max_prompt_num:
        pytest.fail(
            "BLOCKED_EXACT_PROMPT_COUNT: generated prompt count does not match "
            f"the requested batch size: actual={len(all_prompts)}, "
            f"expected={max_prompt_num}"
        )
    print(
        "BATCH_INVARIANCE_EXACT_PROMPT_LENGTH: "
        + json.dumps(
            {
                "target_tokens": max_prompt_len,
                "variation": prompt_len_variation,
                "prompt_target_lengths": prompt_target_lengths,
                "total_prompt_count": len(all_prompts),
                "all_exact": True,
                "unique_lengths": sorted(set(prompt_token_lengths)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    cache_config = llm.llm_engine.vllm_config.cache_config
    block_size = int(getattr(cache_config, "block_size", 128))
    capacity_kv_tokens = getattr(cache_config, "kv_cache_size_tokens", None)
    required_kv_tokens = sum(
        math.ceil((prompt_tokens + max_tokens) / block_size) * block_size
        for prompt_tokens in prompt_token_lengths
    )
    kv_requirement = {
        "batch_size": len(all_prompts),
        "block_size": block_size,
        "prompt_token_lengths": sorted(set(prompt_token_lengths)),
        "max_tokens": max_tokens,
        "required_kv_tokens": required_kv_tokens,
        "capacity_kv_tokens": capacity_kv_tokens,
    }
    print(
        "BATCH_INVARIANCE_KV_REQUIREMENT: "
        + json.dumps(kv_requirement, sort_keys=True),
        flush=True,
    )
    if capacity_kv_tokens is not None and required_kv_tokens > capacity_kv_tokens:
        print(
            "BATCH_INVARIANCE_EVICTION_UNSUPPORTED: "
            + json.dumps(kv_requirement, sort_keys=True),
            flush=True,
        )
        pytest.fail(
            "BLOCKED_EVICTION_UNSUPPORTED: configured batch requires "
            f"{required_kv_tokens} KV tokens, capacity is {capacity_kv_tokens}"
        )

    # ---------- 采样参数 ----------
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        logprobs=logprobs,
        ignore_eos=True,
    )

    # ---------- BS=1 逐个运行 ----------
    print("\n" + "=" * 80)
    print("STARTING BS=1 RUNS (each prompt individually)")
    print("=" * 80 + "\n")
    bs1_logprobs = []
    bs1_tokens = []
    for idx, p in enumerate(all_prompts):
        print(
            f"\n[BS=1] Running prompt {idx}/{len(all_prompts)} - "
            f"Preview: {all_previews[idx][:80]}"
        )
        try:
            outs = llm.generate([p], sp, use_tqdm=False)
            assert len(outs) == 1
            step_logprobs, token_ids = _extract_step_logprobs(outs[0])
            if step_logprobs is None:
                pytest.skip("Logits are not available on RequestOutput; "
                            "enable logprobs return to run this test.")
            if len(step_logprobs) != max_tokens:
                pytest.fail(
                    f"BS=1 prompt {idx} generated {len(step_logprobs)} steps; "
                    f"expected {max_tokens} with ignore_eos=True"
                )
            bs1_logprobs.append(step_logprobs)
            bs1_tokens.append(token_ids)
            print(
                f"[BS=1] Prompt {idx} generated {len(token_ids)} tokens "
                f"and {len(step_logprobs)} logprob steps"
            )
        except Exception as e:
            pytest.fail(f"BS=1 generation failed for prompt {idx}: {str(e)}")

    # ---------- BS=N 批量运行 ----------
    print("\n" + "=" * 80)
    print(f"STARTING BS={len(all_prompts)} RUN (all prompts batched)")
    print("=" * 80 + "\n")

    try:
        outs_batch = llm.generate(all_prompts, sp, use_tqdm=False)
        assert len(outs_batch) == len(all_prompts)
    except Exception as e:
        pytest.fail(f"BS=N generation failed: {str(e)}")
    # prof.step()

    bsN_logprobs = []
    bsN_tokens = []

    print(f"\n[BS={len(all_prompts)}] Processing batched outputs...")
    for idx, o in enumerate(outs_batch):
        step_logprobs, token_ids = _extract_step_logprobs(o)
        if step_logprobs is None:
            pytest.skip("Logits are not available on RequestOutput; "
                        "enable logprobs return to run this test.")
        if len(step_logprobs) != max_tokens:
            pytest.fail(
                f"BS=N prompt {idx} generated {len(step_logprobs)} steps; "
                f"expected {max_tokens} with ignore_eos=True"
            )
        bsN_logprobs.append(step_logprobs)
        bsN_tokens.append(token_ids)
        print(
            f"[BS={len(all_prompts)}] Prompt {idx} generated {len(token_ids)} tokens "
            f"and {len(step_logprobs)} logprob steps"
        )

    # ---------- 释放资源 ----------
    _cleanup_llm_resources(llm_holder)

    # ---------- 详细失败信息收集 ----------
    failed_prompts = []
    failure_stats = {
        "total": 0,
        "step_mismatch": 0,
        "token_mismatch": 0,
        "logprob_mismatch": 0,
        "other": 0
    }

    for i, (lp1, lpN, tk1, tkN) in enumerate(
        zip(bs1_logprobs, bsN_logprobs, bs1_tokens, bsN_tokens)
    ):
        # 1. 检查步数是否一致
        if len(lp1) != len(lpN):
            failure_stats["step_mismatch"] += 1
            failure_stats["total"] += 1
            failed_prompts.append({
                "prompt_idx": i,
                "failure_type": "step_mismatch",
                "reason": f"Different number of steps: {len(lp1)} (BS=1) vs {len(lpN)} (BS=N)",
                "prompt_preview": all_previews[i][:100],
                "bs1_tokens": tk1,
                "bsN_tokens": tkN,
            })
            continue

        # 2. token 序列必须完全一致，才能比较同一路径上的 logprob。
        if tk1 != tkN:
            failure_stats["token_mismatch"] += 1
            failure_stats["total"] += 1
            diff_pos = next(
                (pos for pos, (t1, tN) in enumerate(zip(tk1, tkN)) if t1 != tN),
                -1,
            )
            failed_prompts.append({
                "prompt_idx": i,
                "failure_type": "token_mismatch",
                "reason": f"Different tokens sampled at position {diff_pos}",
                "prompt_preview": all_previews[i][:100],
                "bs1_tokens": tk1,
                "bsN_tokens": tkN,
                "diff_position": diff_pos,
                "bs1_token_at_diff": tk1[diff_pos] if diff_pos >= 0 else None,
                "bsN_token_at_diff": tkN[diff_pos] if diff_pos >= 0 else None,
            })
            continue

        # 3. 仅当 token 序列一致时逐 token 比较 chosen-token logprob。
        logprob_mismatch_found = False
        max_diff = 0.0
        first_mismatch_pos = -1
        max_diff_pos = -1
        first_bs1_logprob = None
        first_bsN_logprob = None

        for t, (a, b) in enumerate(zip(lp1, lpN)):
            if not torch.equal(a, b):
                if first_mismatch_pos < 0:
                    first_mismatch_pos = t
                    first_bs1_logprob = a.item()
                    first_bsN_logprob = b.item()
                diff = torch.abs(a - b).max().item()
                if diff > max_diff:
                    max_diff = diff
                    max_diff_pos = t
                logprob_mismatch_found = True

        if logprob_mismatch_found:
            failure_stats["logprob_mismatch"] += 1
            failure_stats["total"] += 1
            failed_prompts.append({
                "prompt_idx": i,
                "failure_type": "logprob_mismatch",
                "reason": (
                    "Logprob bitwise mismatch "
                    f"(first_pos={first_mismatch_pos}, max_diff={max_diff:.6e} "
                    f"at pos {max_diff_pos})"
                ),
                "prompt_preview": all_previews[i][:100],
                "max_diff": max_diff,
                "mismatch_position": first_mismatch_pos,
                "max_diff_position": max_diff_pos,
                "bs1_logprob_at_mismatch": first_bs1_logprob,
                "bsN_logprob_at_mismatch": first_bsN_logprob,
            })

    # ---------- 失败汇总与断言 ----------
    if failed_prompts:
        manifest_failures = []
        for failure in failed_prompts:
            prompt_idx = failure["prompt_idx"]
            mismatch_position = failure.get(
                "diff_position", failure.get("mismatch_position", -1)
            )
            token_prefix_end = (
                mismatch_position + 1
                if mismatch_position is not None and mismatch_position >= 0
                else 0
            )
            manifest_failure = {
                key: value
                for key, value in failure.items()
                if key not in {"bs1_tokens", "bsN_tokens"}
            }
            manifest_failure.update(
                {
                    "logical_request_id": f"batchinv-{prompt_idx:05d}",
                    "prompt_token_ids": all_prompt_token_ids[prompt_idx],
                    "prompt_token_count": len(all_prompt_token_ids[prompt_idx]),
                    "bs1_token_prefix_through_mismatch": bs1_tokens[prompt_idx][
                        :token_prefix_end
                    ],
                    "bsN_token_prefix_through_mismatch": bsN_tokens[prompt_idx][
                        :token_prefix_end
                    ],
                }
            )
            manifest_failures.append(manifest_failure)

        failure_manifest = {
            "schema_version": "1.0",
            "status": "FAIL",
            "comparison": "BS=1_vs_BS=N",
            "scenario": os.getenv("VLLM_TEST_SCENARIO", "phase2"),
            "model": model_name,
            "parallel": {"mode": "TP", "tp_size": tp_size},
            "sampling": {
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "logprobs": logprobs,
            },
            "runtime": {
                "batch_size": len(all_prompts),
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_num_batched_tokens,
                "enable_prefix_caching": enable_prefix_caching,
                "enable_chunked_prefill": False,
                "async_scheduling": False,
                "block_size": block_size,
                "all_requests_completed_expected_steps": True,
                "total_prompt_tokens": sum(prompt_token_lengths),
                "prompt_token_lengths": prompt_token_lengths,
            },
            "failure_counts": failure_stats,
            "failures": manifest_failures,
        }
        manifest_path = _write_json_artifact(
            "VLLM_TEST_FAILURE_MANIFEST", failure_manifest
        )
        if manifest_path is not None:
            print(f"BATCH_INVARIANCE_FAILURE_MANIFEST: {manifest_path}", flush=True)

        print(f"\n{'=' * 80}")
        fail_msg = f"BATCH INVARIANCE FAILURES: {failure_stats['total']}/{len(all_prompts)} prompts failed"
        print(fail_msg)
        print(f"{'=' * 80}")

        # 打印统计汇总
        print("\nFAILURE STATISTICS:")
        print(f"  - Step Mismatch: {failure_stats['step_mismatch']}")
        print(f"  - Token Mismatch: {failure_stats['token_mismatch']}")
        print(f"  - Logprob Mismatch: {failure_stats['logprob_mismatch']}")
        print(f"  - Other: {failure_stats['other']}")

        # 打印详细的失败信息（最多显示前5个）
        print("\nDETAILED FAILURES (first 5):")
        for fail in failed_prompts[:5]:
            print(f"\nPrompt {fail['prompt_idx']}:")
            print(f"  Type: {fail['failure_type']}")
            print(f"  Reason: {fail['reason']}")
            print(f"  Preview: {fail['prompt_preview'][:80]}...")
            if fail['failure_type'] == 'token_mismatch':
                print(f"  BS=1 token at diff: {fail.get('bs1_token_at_diff')}")
                print(f"  BS=N token at diff: {fail.get('bsN_token_at_diff')}")
            elif fail['failure_type'] == 'logprob_mismatch':
                print(f"  Max diff: {fail.get('max_diff', 0):.6e}")

        print(f"{'=' * 80}\n")

        # 将失败统计写入当前 attempt 的独立文件，供 shell runner 读取。
        stats_path = Path(
            os.getenv(
                "VLLM_TEST_FAILURE_STATS_FILE",
                "batch_invariant_failure_stats.txt",
            )
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as f:
            f.write(f"TOTAL_FAILURES={failure_stats['total']}\n")
            f.write(f"STEP_MISMATCH={failure_stats['step_mismatch']}\n")
            f.write(f"TOKEN_MISMATCH={failure_stats['token_mismatch']}\n")
            f.write(f"LOGPROB_MISMATCH={failure_stats['logprob_mismatch']}\n")
            f.write(f"FIRST_FAILURE_REASON={failed_prompts[0]['reason'] if failed_prompts else ''}\n")

        pytest.fail(fail_msg)
    else:
        # 没有失败，删除当前 attempt 可能残留的统计或 manifest。
        for env_name in (
            "VLLM_TEST_FAILURE_STATS_FILE",
            "VLLM_TEST_FAILURE_MANIFEST",
        ):
            raw_path = os.getenv(env_name)
            if raw_path and os.path.exists(raw_path):
                os.remove(raw_path)
