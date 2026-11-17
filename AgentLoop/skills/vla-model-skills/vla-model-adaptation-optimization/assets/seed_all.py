#!/usr/bin/env python3
"""固定全局随机种子（NPU/GPU），迁移/优化 A/B 前必须调用。

若训练入口未固定全局种子，模型随机初始化部分（未从 checkpoint 加载的模块）每进程初值
不同 → 同配置两跑 loss 不一致、A/B 精度对齐被初值噪声污染（实测可把 40 步 max|dloss|
从真实 0.245 抬到 1.470）。必须在任何随机初始化之前调用（train() 开头）。

注意：PYTHONHASHSEED 仅解释器启动前生效（启动脚本内 export），进程内设置无效。
多进程传相同 seed；需 rank 差异化时用 seed+rank 且 A/B 两臂保持同一规则。
"""
import random

import numpy as np
import torch

__all__ = ["seed_all"]


def seed_all(seed: int = 1234, is_npu: bool = True) -> None:
    """固定全局随机种子。

    Args:
        seed:   随机种子。
        is_npu: True=昇腾（torch.npu.manual_seed_all），False=CUDA。
                自动探测优先：torch.npu 可用时忽略 is_npu 传值按 NPU 处理。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "npu") and torch.npu.is_available():
        # import torch_npu 后 torch.npu 即 torch_npu.npu；manual_seed_all 覆盖所有设备
        # （含当前 device，故不再单独调 manual_seed）
        torch.npu.manual_seed_all(seed)
        return

    if is_npu:
        raise RuntimeError("is_npu=True 但未检测到 torch.npu (需要 import torch_npu / 先 source CANN set_env.sh)")

    if not torch.cuda.is_available():
        # CPU 兜底（cudnn 属性在 cpu-only 构建不存在，勿直接访问）
        return
    torch.cuda.manual_seed_all(seed)   # 已含当前 device，无需再调 manual_seed
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001  cpu-only torch 无 cudnn
        pass


if __name__ == "__main__":
    # 用法示例
    seed_all(1024, is_npu=True)  # NPU
    # seed_all(1024, is_npu=False)  # CUDA
    print("seed fixed:", torch.initial_seed())
