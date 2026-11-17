# test_allreduce_batch_invariance.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""AllReduce batch invariance test - each rank holds different data."""

import os
import random
import pickle
from typing import List, Tuple

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp


# ==================== 配置参数（环境变量控制） ====================

def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


# 测试配置
TP_SIZE = get_env_int("TEST_TP_SIZE", 8)              # TP并行度，默认8卡
HIDDEN_SIZE = get_env_int("TEST_HIDDEN_SIZE", 2048)   # 隐藏层维度
NUM_PROMPTS = get_env_int("TEST_NUM_PROMPTS", 32)     # prompt数量
PROMPT_LEN = get_env_int("TEST_NUM_PROMPT_LEN", 100)     # prompt长度
SEED = get_env_int("TEST_SEED", 12345)                # 随机种子
MASTER_IP = get_env_str("TEST_MASTER_IP", "90.90.97.71") # 本机IP
MASTER_PORT_BASE = get_env_int("TEST_MASTER_PORT", "11234")

DTYPE = torch.bfloat16
os.environ["HCCL_DETERMINISTIC"] = "strict"

# 保存路径
SAVE_DIR = get_env_str("TEST_SAVE_DIR", "path_to_save/")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================== 模拟Prompt生成 ====================

PROMPT_TEMPLATES = [
    "Question: What is the capital of France?\nAnswer: The capital of France is",
    "Q: How does photosynthesis work?\nA: Photosynthesis is the process by which",
    "User: Can you explain quantum mechanics?\nAssistant: Quantum mechanics is",
    "Once upon a time in a distant galaxy, there lived",
    "To implement a binary search tree in Python, first we need to",
    "Climate change is caused by several factors including",
    "The history of artificial intelligence began in",
    "In machine learning, neural networks are",
    "The solar system consists of eight planets which are",
    "Python is a high-level programming language that",
]


def generate_random_prompt(min_len: int = 10, max_len: int = 50) -> str:
    """生成随机prompt"""
    base = random.choice(PROMPT_TEMPLATES)
    extra_len = random.randint(min_len, max_len)
    if extra_len > 20:
        base += " " + "This is additional context. " * (extra_len // 20)
    return base


def tokenize_prompt_dummy(prompt: str) -> int:
    """模拟tokenize，返回seq_len"""
    word_count = len(prompt.split())
    return max(5, int(word_count * 1.3))


def generate_prompts_with_seqlen(num_prompts: int, seed: int) -> Tuple[List[str], List[int]]:
    """生成prompts及其seq_len"""
    random.seed(seed)
    prompts = []
    seqlens = []

    for i in range(num_prompts):
        prompt = generate_random_prompt(PROMPT_LEN + 10, PROMPT_LEN + 50)
        prompt = f"[Prompt{i:03d}] " + prompt
        prompts.append(prompt)
        seq_len = tokenize_prompt_dummy(prompt)
        seqlens.append(seq_len)

    return prompts, seqlens


# ==================== 关键：每个rank生成不同的数据 ====================

def generate_rank_specific_data(
    rank: int,
    prompts: List[str],
    seqlens: List[int],
    hidden_size: int,
    dtype: torch.dtype,
    base_seed: int
) -> List[torch.Tensor]:
    """
    为指定rank生成特定的输入数据
    每个rank使用不同的seed，确保数据不同
    """
    # 每个rank使用不同的seed: base_seed + rank * 10000
    torch.manual_seed(base_seed + rank * 10000)

    rank_inputs = []
    for prompt_idx, (prompt, seq_len) in enumerate(zip(prompts, seqlens)):
        # 为当前rank和当前prompt生成数据 [seq_len, hidden_size]
        # 再添加rank特定的偏移
        torch.manual_seed(base_seed + rank * 10000 + prompt_idx)

        tensor = torch.randn(seq_len, hidden_size, dtype=dtype)

        # 添加rank特定的偏置，确保不同rank数据差异明显
        rank_bias = float(rank) * 0.1
        position_bias = torch.arange(seq_len, dtype=dtype).view(-1, 1) * 0.01
        tensor = tensor + rank_bias + position_bias

        rank_inputs.append(tensor)

    return rank_inputs


# ==================== AllReduce 工作进程 ====================

def allreduce_worker(
    rank: int,
    world_size: int,
    mode: str,  # 'single' 或 'batch'
    prompts: List[str],
    seqlens: List[int],
    prompt_indices: List[int],  # 当前mode处理的prompt索引
    master_ip: str,
    master_port: int,
    save_dir: str,
    test_id: str,
    base_seed: int
):
    """
    AllReduce工作进程
    每个rank生成自己的数据，AllReduce SUM后所有rank结果相同
    """
    # 设置设备
    torch_npu.npu.set_device(rank)

    # 初始化进程组
    init_method = f'tcp://{master_ip}:{master_port}'
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method=init_method)

    print(f"[{mode}] Rank {rank} started, generating rank-specific data...")

    # 每个rank生成自己的数据（与其他rank不同）
    rank_inputs = generate_rank_specific_data(rank, prompts, seqlens, HIDDEN_SIZE, DTYPE, base_seed)

    if mode == 'single':
        # 单条模式：逐个处理每个prompt
        results = []
        for pidx in prompt_indices:
            # 获取当前prompt的数据 [seq_len, hidden_size]
            tensor = rank_inputs[pidx].npu()

            # AllReduce SUM: 所有rank的数据相加
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.npu.synchronize()

            if rank == 0:
                results.append(tensor.cpu())
                print(f"[Single][Rank{rank}] Prompt {pidx} done, shape={tensor.shape}")

        # rank 0保存结果
        if rank == 0:
            # 按原始顺序重组结果
            combined = torch.cat(results, dim=0)  # [total_seq_len, hidden_size]
            save_path = os.path.join(save_dir, f"{test_id}_single_results.pt")
            torch.save(combined, save_path)
            print(f"[Single] Rank 0 saved {len(prompt_indices)} prompts results, shape={combined.shape}")

    elif mode == 'batch':
        # 批量模式：一次性处理所有prompt
        # 构造batch输入：将所有prompt的tensor拼接 [total_seq_len, hidden_size]
        batch_tensor = torch.cat([rank_inputs[pidx] for pidx in prompt_indices], dim=0).npu()
        original_shape = batch_tensor.shape

        print(f"[Batch][Rank{rank}] Batch input shape: {original_shape}")

        # AllReduce SUM
        dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()

        if rank == 0:
            save_path = os.path.join(save_dir, f"{test_id}_batch_results.pt")
            torch.save(batch_tensor.cpu(), save_path)
            print(f"[Batch] Rank 0 saved batch result, shape={batch_tensor.shape}")

    dist.destroy_process_group()


# ==================== 验证辅助函数 ====================

def check_tensor_close(tensor1, tensor2, rtol=1e-3, atol=1e-5, context=""):
    """检查两个tensor是否接近"""
    if tensor1.shape != tensor2.shape:
        print(f"[{context}] Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False


    # if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
    #     return True

    if torch.equal(tensor1, tensor2):
        return True

    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    max_idx = torch.argmax(diff.view(-1)).item()

    # 转换回多维索引
    flat_idx = max_idx
    indices = []
    for dim in reversed(tensor1.shape):
        indices.append(flat_idx % dim)
        flat_idx //= dim
    indices = tuple(reversed(indices))

    print(f"[{context}] Max diff: {max_diff:.6e} at index {indices}")
    print(f"  Tensor1: {tensor1[indices].item():.6f}")
    print(f"  Tensor2: {tensor2[indices].item():.6f}")
    return False


# ==================== 主测试逻辑 ====================

def test_allreduce_batch_invariant():
    """
    验证AllReduce的batch不变性

    核心逻辑：
    1. 生成32个不同的prompt
    2. 每个rank为每个prompt生成不同的数据（rank-specific）
    3. 方式1：逐个prompt做AllReduce（32次通信），收集所有结果
    4. 方式2：将所有prompt拼接后一次性AllReduce（1次通信）
    5. 验证两种方式的结果逐元素相等（都是所有rank数据的和）
    """

    print("=" * 80)
    print("AllReduce Batch Invariance Test")
    print(f"Each rank holds DIFFERENT data, AllReduce SUM")
    print(f"Config: TP_SIZE={TP_SIZE}, HIDDEN_SIZE={HIDDEN_SIZE}, NUM_PROMPTS={NUM_PROMPTS}")
    print(f"SEED={SEED}, DTYPE={DTYPE}")
    print("=" * 80)

    # Step 1: 生成prompts
    print("\n[Step 1] Generating prompts...")
    prompts, seqlens = generate_prompts_with_seqlen(NUM_PROMPTS, SEED)

    total_seq_len = sum(seqlens)
    print(f"Generated {NUM_PROMPTS} prompts, total_seq_len={total_seq_len}")
    for i in range(min(5, NUM_PROMPTS)):
        print(f"  Prompt {i}: seq_len={seqlens[i]}, preview='{prompts[i][:50]}...'")

    # 保存prompts信息
    meta_path = os.path.join(SAVE_DIR, "test_meta.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'prompts': prompts,
            'seqlens': seqlens,
            'config': {
                'tp_size': TP_SIZE,
                'hidden_size': HIDDEN_SIZE,
                'num_prompts': NUM_PROMPTS,
                'seed': SEED
            }
        }, f)

    # Step 2: 方式1 - 逐条AllReduce
    print("\n" + "=" * 80)
    print("[Mode 1] Single AllReduce (32 individual communications)")
    print("Each rank generates different data, AllReduce SUM for each prompt")
    print("=" * 80)

    mp.spawn(
        allreduce_worker,
        args=(
            TP_SIZE, 'single', prompts, seqlens, list(range(NUM_PROMPTS)),
            MASTER_IP, MASTER_PORT_BASE, SAVE_DIR, "test1", SEED
        ),
        nprocs=TP_SIZE
    )

    # Step 3: 方式2 - 批量AllReduce
    print("\n" + "=" * 80)
    print("[Mode 2] Batch AllReduce (1 communication for all 32 prompts)")
    print("Same data generation, but AllReduce on concatenated tensor")
    print("=" * 80)

    mp.spawn(
        allreduce_worker,
        args=(
            TP_SIZE, 'batch', prompts, seqlens, list(range(NUM_PROMPTS)),
            MASTER_IP, MASTER_PORT_BASE + 1, SAVE_DIR, "test1", SEED
        ),
        nprocs=TP_SIZE
    )

    # Step 4: 结果对比
    print("\n" + "=" * 80)
    print("[Step 4] Result Comparison")
    print("=" * 80)

    single_results = torch.load(os.path.join(SAVE_DIR, "test1_single_results.pt"))
    batch_results = torch.load(os.path.join(SAVE_DIR, "test1_batch_results.pt"))

    print(f"\nSingle mode result shape: {single_results.shape}")
    print(f"Batch mode result shape: {batch_results.shape}")
    print(f"Expected: Each element = sum of rank0~rank{TP_SIZE-1} original data")

    # 逐prompt对比（关键验证点）
    offset = 0
    failed_prompts = []
    failure_stats = {"total": 0, "shape_mismatch": 0, "value_mismatch": 0}

    print(f"\nPer-prompt comparison:")
    for pidx in range(NUM_PROMPTS):
        seq_len = seqlens[pidx]

        # 提取当前prompt的结果
        single_prompt_result = single_results[offset:offset+seq_len, :]
        batch_prompt_result = batch_results[offset:offset+seq_len, :]

        # 对比
        match = check_tensor_close(
            single_prompt_result,
            batch_prompt_result,
            rtol=1e-3,
            atol=1e-5,
            context=f"Prompt{pidx}"
        )

        if match:
            print(f"  Prompt {pidx:2d}: ✅ PASS (seq_len={seq_len:3d}, offset={offset:4d})")
        else:
            print(f"  Prompt {pidx:2d}: ❌ FAIL (seq_len={seq_len:3d}, offset={offset:4d})")
            failure_stats["total"] += 1
            failure_stats["value_mismatch"] += 1
            failed_prompts.append({
                "prompt_idx": pidx,
                "prompt": prompts[pidx][:80],
                "seq_len": seq_len,
                "offset": offset
            })

        offset += seq_len

    # 整体对比
    print(f"\nOverall tensor comparison:")
    overall_match = check_tensor_close(single_results, batch_results, context="Overall")

    # 统计信息
    diff = torch.abs(single_results - batch_results)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    print(f"\nStatistics:")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Std of diff:        {std_diff:.6e}")
    print(f"  Failed prompts:     {failure_stats['total']}/{NUM_PROMPTS}")

    # 数值范围检查（验证是否确实是SUM结果）
    print(f"\nValue range check:")
    print(f"  Single results: min={single_results.min():.4f}, max={single_results.max():.4f}")
    print(f"  Batch results:  min={batch_results.min():.4f}, max={batch_results.max():.4f}")
    # 理论上，8个rank的随机数（~N(0,1)）相加，范围应该在[-10, 10]左右

    # 最终结论
    print("\n" + "=" * 80)
    if failure_stats["total"] == 0 and overall_match:
        print("✅ TEST PASSED: AllReduce is batch invariant!")
        print("   Single-mode and batch-mode produce identical SUM results.")
        print(f"   All {NUM_PROMPTS} prompts match perfectly.")
    else:
        print("❌ TEST FAILED: AllReduce is NOT batch invariant!")
        print(f"   {failure_stats['total']} prompts have mismatched results.")
        if failed_prompts:
            print(f"\n   First failure details:")
            print(f"   - Prompt index: {failed_prompts[0]['prompt_idx']}")
            print(f"   - Seq len: {failed_prompts[0]['seq_len']}")
            print(f"   - Preview: {failed_prompts[0]['prompt'][:60]}...")
    print("=" * 80)

    # 保存详细报告
    report_path = os.path.join(SAVE_DIR, "test_report.txt")
    with open(report_path, 'w') as f:
        f.write("AllReduce Batch Invariance Test Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"TP_SIZE={TP_SIZE}, HIDDEN_SIZE={HIDDEN_SIZE}, NUM_PROMPTS={NUM_PROMPTS}\n")
        f.write(f"Each rank holds DIFFERENT data\n")
        f.write(f"Result: {'PASS' if failure_stats['total'] == 0 else 'FAIL'}\n")
        f.write(f"Max diff: {max_diff:.6e}\n")
        f.write(f"Mean diff: {mean_diff:.6e}\n")
        f.write(f"Failed prompts: {failure_stats['total']}\n")
        if failed_prompts:
            f.write(f"First failure: Prompt {failed_prompts[0]['prompt_idx']}\n")

    print(f"\nTest artifacts saved to: {SAVE_DIR}")

    return failure_stats["total"] == 0


if __name__ == "__main__":
    success = test_allreduce_batch_invariant()
    exit(0 if success else 1)
