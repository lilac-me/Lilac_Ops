# vllm / vllm-ascend 环境安装要求（skill 规范）

> 结论：安装 vllm、vllm-ascend 时，**依赖清单里的包必须安装**。

## vllm —— 每次安装先跑 use_existing_torch.py

- 脚本：https://github.com/vllm-project/vllm/blob/main/use_existing_torch.py
  （共享盘副本：/mnt/share/{shared}/{vllm,0709/vllm,A3-OOM/vllm}/use_existing_torch.py）
- 作用：原地删除 vllm requirements/**/*.txt、*.in、pyproject.toml 中所有含 torch/torchvision/torchaudio 的行，然后 `pip install -r requirements.txt` 只装其余依赖，保留已装 torch/torch_npu。

```bash
cd <vllm源码>
python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt
pip install -e .   # 或 wheel
```

## vllm-ascend —— 必须 pip install -r requirements.txt

- 在 vllm-ascend 源码目录执行 `pip install -r requirements.txt`。
- 内容要点（torch2.10 版本）：cmake、decorator、einops、numpy、pyyaml、scipy、pandas、psutil、xgrammar>=0.1.30、compressed_tensors>=0.11.0、msgpack、quart、numba、memfabric_hybrid、memcache_hybrid、arctic-inference==0.1.1、transformers==5.5.4/5.13.0、fastapi<0.124.0、opencv-python-headless 等。
- 硬性 pin：torch==2.10.0 / torch-npu==2.10.0 / triton-ascend==3.2.1 / torchvision==0.25.0 / torchaudio==2.10.0。

## ⚠️ 红线

- 看护环境 ops_nightly_agent = torch_npu 2.10.0.post1.dev + triton_ascend 3.6.0+dev（nightly 验证通过）。
- vllm-ascend 要求 triton-ascend==3.2.1 / torch-npu==2.10.0 → **必须独立环境安装**，勿装入看护环境。
- 推荐组合（参考 byh_vllm_py11_122b）：torch 2.10.0 + torch_npu 2.10.0 + triton_ascend 3.2.1 + vllm 0.23.1rc1.dev752（/mnt/share/{shared}/0709/vllm）+ vllm-ascend 0.19.1rc2.dev934（/mnt/share/{shared}/0709/vllm-ascend）。
