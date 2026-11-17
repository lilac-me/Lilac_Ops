# RLHF E2E 回移与排查

## 入口条件与分类

仅当“单 curl 请求或普通单推理不复现，只有经过 RL rollout、权重同步或 RLHF 状态切换后才出现”时进入本流程。

进入前必须确认：

- 用户已有 RL 侧复现入口，或能提供完整 RL/RLHF 复现脚本。
- 已记录单 curl/单推理的 URL、模型、请求体、输出和结果。未测过时先补做最小单请求基线。
- 单请求能复现时，回到主流程 2-5，不要用 RLHF 测试替代最小复现。

如果 RL 用例也无法复现，先检查训练与推理链路差异，不要直接判定 vLLM-Ascend 精度正常。

## 目录检查

以实际导入的 vLLM-Ascend 路径为根目录，检查：

```text
tests/e2e/pull_request/one_card/rlhf
tests/e2e/pull_request/rlhf/weight_transfer_test_utils.py
```

不要只检查当前 shell 所在目录。

## 获取最新代码

容器无法访问 GitHub 时，可在能联网的主机下载，再复制到目标容器。代理按实际环境配置，不要把凭据写入 skill 或提交到仓库。

```bash
export IP=<proxy-host>
export http_proxy="http://<user>:<password>@$IP:8080"
export https_proxy="$http_proxy"
```

验证代理：

```bash
curl -I --max-time 10 https://github.com
git ls-remote https://github.com/vllm-project/vllm-ascend HEAD
```

记录下载代码的 commit：

```bash
git -C <latest-vllm-ascend> rev-parse HEAD
git -C <latest-vllm-ascend> log -1 --format='%H %ci %s'
```

## 回移范围

至少复制：

- `tests/e2e/pull_request/one_card/rlhf/`
- `tests/e2e/pull_request/rlhf/weight_transfer_test_utils.py`

复制前确认目标路径和源路径，避免覆盖现有同名测试。

## conftest 适配

逐项核对，不要机械复制：

1. 模型路径是否与目标服务脚本一致。
2. 是否需要 `--trust-remote-code`。
3. 大模型首次加载是否需要更长超时。
4. 测试日志目录是否可写。
5. 每次启动服务是否生成唯一日志文件名。
6. 端口是否固定；固定端口便于排查，但每次启动前必须确认已释放。
7. `pytest` 进程是否提前初始化 NPU 上下文。

唯一日志示例：

```python
log_path = os.path.join(log_dir, f"server_{port}_{time.time_ns()}.log")
```

## 兼容性门禁

按当前代码能力明确跳过，而不是让用例在运行中崩溃：

- FA3 测试：检查 `flash_attn_npu_v3` 是否存在。
- `synchronize_device`：检查 `NPUWorker` 是否提供该接口。
- trace replay：检查 `SamplingParams` 是否支持所需字段。
- weight transfer：检查 `vllm.distributed.weight_transfer.clients` 是否存在。

每个跳过必须写明缺失能力，不能只写 `skip`。

## 执行顺序

```bash
pytest -vv -s tests/e2e/pull_request/one_card/rlhf/state_transitions/test_pause_resume.py
pytest -vv -s tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py
pytest -vv -s tests/e2e/pull_request/one_card/rlhf
```

先跑文件可以快速区分：

- 启动失败
- pause/resume 失败
- sleep/wake 恢复失败
- 全量测试之间的状态污染

## 首轮失败分类

不要直接相信 `FAILED` 汇总。按以下顺序读：

1. 第一条失败的完整 traceback。
2. 对应服务端日志的首个 EngineCore 异常。
3. 是否同一端口、同一服务实例。
4. 后续失败是否为 `Connection refused`、`ReadTimeout`、`500`。
5. 单条复跑是否通过。

单条通过、全量失败时，优先检查模块导入副作用和环境变量污染。

## sleep/wake 专项判断

睡眠内存测试通过，不代表权重恢复正确。

需要区分：

- `sleep` 返回 200，且显存确实下降：释放链路正常。
- `wake_up` 返回 200，`/health` 正常：控制面恢复。
- 唤醒后第一次推理崩溃：数据面/权重恢复失败。

典型真实故障：

```text
AclNN_Parameter_Error(EZ1001):
Dim 1 value of x[0] should be equal with dim 1 value of weight[0],
but now is 2048 and 2816 respectively.
```

若该错误只出现在 `sleep` 后 `wake_up` 的首次推理，而睡眠前推理正常，应优先检查：

- MoE 权重缓冲区的重映射
- 权重命名和 shape 映射
- 分层 reload 路径
- fused MoE 派生权重恢复
- sleep level 与 wake tags 的对应关系

## 测试框架污染模式

危险写法：

```python
# 模块导入阶段执行，即使该模块所有测试都被 skip 也会生效
os.environ["VLLM_BATCH_INVARIANT"] = "1"
```

安全写法：

```python
@pytest.fixture(autouse=True)
def _runtime_env(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
```

修复后必须重新单独运行受影响文件，确认环境变量不再泄漏。
