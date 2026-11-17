# 阶段三 Math500 调用

## Math500 验收证据

先判断当前进程是否与目标评测进程处于同一服务器、容器或 Conda 运行空间，并能直接访问模型、目标 NPU 和证据目录：满足时设置 `<TARGET_MODE>=LOCAL`，否则设置为 `REMOTE`。替换所有 `<...>` 占位符后，将下面整段内容作为 `/mindspeed-agent:vllm-ascend-aisbench-accuracy` 的调用 prompt。不要传入该 skill 的本地文件路径，也不要要求调用方直接执行其内部脚本。

```text
/mindspeed-agent:vllm-ascend-aisbench-accuracy
请在 batch invariance Phase 1/2 通过后完成 Phase 3 全量 Math500 数据集评测；A2/A3 和 A5 均要求环境准入、对应芯片白名单分析和 Phase 2 通过：
1. target-mode=<LOCAL|REMOTE>，本次运行记录目录=<RUN_RECORD_DIR>，CANN 路径=<CANN_DIR>；所有 Phase 3 证据归档到 `<RUN_RECORD_DIR>/phase3/`。使用模型 <MODEL_PATH>、TP=<TP>、设备 <ASCEND_RT_VISIBLE_DEVICES>。当前正式支持范围仅为 TP。
   LOCAL：当前环境就是目标服务器、容器或 Conda 运行空间；只设置 mode=local，不传 host、container、conda-env、user、password、key-file 等连接参数。local 仅支持单机评测。AISBench 使用公共 skill 写入的 `host_ip=127.0.0.1` 访问本地 vLLM 服务，不依赖 `localhost` 的 IPv4/IPv6 解析，也不要通过 serve_args 增加 `--host ::` 改变监听地址。
   REMOTE：连接参数为 host=<HOST>、mode=<docker|direct|conda>、container/conda=<CONTAINER_OR_CONDA>、认证信息=<AUTH>。
2. 必须传入 VLLM_BATCH_INVARIANT=1；Phase 3 标准验收固定传入 temperature=0、seed=1234、Math500 全量数据集、连续 10 轮；不要沿用 Phase 2 默认的 temperature=0.6。
3. 直接使用公共 skill 的数据集配置：math500_gen_0_shot_cot_chat_prompt、dataset_kind=math、max_out_len=20480；不修改公共数据集、模型配置类型或 import/PYTHONPATH。
4. 服务侧必须将以下正式验收参数作为显式 serve_args 传入并在 raw_result.config.serve_args 中原样回显：--max-model-len 22528、--max-num-seqs <MAX_NUM_SEQS>、--max-num-batched-tokens 22528、--block-size 128、--no-enable-chunked-prefill、--distributed-executor-backend mp、--gpu-memory-utilization 0.9、--enforce-eager、--generation-config vllm、--no-enable-prefix-caching、--no-async-scheduling。`MAX_NUM_SEQS` 由调用方按服务资源在执行前固定，10 轮保持一致。chunked prefill、prefix caching 和 async scheduling 三项必须关闭，不得为完成评测临时启用。load format 保持默认 `auto` 即可。vLLM 0.23.1 的 `disable_log_stats` 默认值为 False，CLI 仅提供 `--disable-log-stats`，不支持 `--no-disable-log-stats`；因此不要传入二者，验收器按缺省 False 校验。不要启用 dummy load。profiling、阶段二和 Math500 的 enable_chunked_prefill 与 block_size 必须保持一致。
5. 由本 skill 自动完成网络预检、ais-bench 安装、Math500 下载、服务启动/停止、10 轮评测和公共精度报告，不调用 batch-invariance-validation 中已移除的旧 Phase 3 脚本。
6. 保持 disable_log_stats=False，并保留完整场景配置和每轮评测结果。
7. 成功时返回调用退出码、每轮 accuracy、唯一分数集合，以及归档到运行记录目录后的公共报告、raw_result 和 10 轮预测文件的相对位置。
8. 失败或阻塞时返回明确 status、phase、error、已生成证据路径和服务清理状态；不得只返回一句执行失败。
9. batch-invariance-validation 会在调用返回后完成最终验收，并将结果写入 `<RUN_RECORD_DIR>/phase3/phase3_acceptance.json`。不要用平均分、标准差或公共 skill 的 status=ok 代替最终 batch invariance 结论，也不要展示 skill 安装目录或内部脚本路径。
10. PP、EP、DP 只能报告理论支持且尚未验证；其他并行方式只能报告可能支持且尚未分析/验证。不要传入 temperature 之外的其他采样参数。
```
