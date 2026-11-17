# `<ModelName>` for PyTorch

## 目录
<!-- 扁平锚点列表，镜像下面每个标题 -->

# 简介

## 模型介绍
<!-- 1~2 段中文，从论文摘要转述 -->

## 支持任务列表

| 模型  | 任务列表 | 是否支持 |
| :---: | :------: | :------: |
| `<M>` |   训练   |    ✔     |

## 代码实现

- 参考实现：

    ```shell
    url=<upstream git url>
    commit_id=<pinned sha>
    ```

- 适配昇腾 AI 处理器的实现：

    ```shell
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/<ModelName>
    ```

# 准备训练环境
<!-- 模板说明（交付前删）：本节必须可复现：拿到文档的人在目标服务器上能按步骤装出环境。
     环境搭建的踩坑沉淀进 skill 的 references/flow/1-migrate.md 与该模型工作区的踩坑记录，不写进交付件。-->

## 安装昇腾环境

参考昇腾社区《PyTorch框架训练环境准备》。实测清单（至少给出 CANN 版本与 torch_npu 安装方式）：

```sh
# 1) CANN（root）: 装好 ascend-toolkit，记录版本（如 CANN 9.0.0）
ls /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2) conda env + CPU 版 torch（NPU 上不能装 GPU 版 torch）
conda create -n <env> python=3 -y && conda activate <env>
pip install torch==<version> --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==<版本配对>   # 版本配对表见 ../references/flow/1-migrate.md
# 3) 验证（source CANN 后）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
```

**表 1** 昇腾软件版本支持表

> 示例版本，**按实际验证环境填写**。注意两套版本号：`TorchNPU` 是文档/产品版本（如 26.0.0），
> 而 PyPI wheel 仍按 PyTorch 版本命名（如 torch_npu 2.7.1.postX）。
> 与表 2 的 PyTorch 2.7.1 配对，语料里最常见的组合就是下面这个（全仓 10+ 个模型）：
> TorchNPU 26.0.0 / CANN 9.0.0（较新的是 26.1.0 / 9.1.0）。
> 不要写成 TorchNPU 7.x：那是配 PyTorch 2.1 时代的版本号（如 Pi-0 的 7.1.0 / CANN 8.2.RC1）。

| 软件类型 | 首次支持版本 |
| :------: | :----------: |
| TorchNPU |    26.0.0    |
|   CANN   |    9.0.0     |

## 安装模型环境

**表 2** 版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| Python  |    3.x   |
| PyTorch |  2.7.1   |

0. 激活 CANN：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
1. `conda create -n <env> python=3 && conda activate <env>`
2. 准备源码（顺序**不能反**：先 `git apply` 打补丁，再覆盖交付文件；先覆盖会让 patch 打不干净）

    ```sh
    git clone <upstream> && cd <repo>
    git checkout <commit_id>
    cp -f ../<model>.patch ./                    # 只有 patch 本身先拷进来
    git apply --reject <model>.patch             # 1. 先在干净 checkout 上打补丁
    cp -rf ../test/ ./                           # 2. 再覆盖交付文件（会覆盖同名文件，故必须在 patch 之后）
    pip install -e . --no-build-isolation
    ```

3. 安装 Driving SDK（如需）
4. `pip install -r requirements.txt`

# 准备数据集

## 获取训练数据集

与「获取预训练权重」同等重要。以下三条必须填满（按实际可执行的命令/目录填，不要占位）：

1. 可执行的获取/构造方式：上游 release 链接 / 官方下载命令 / 合成脚本（给出命令或脚本相对路径与运行方式）；
2. 目录结构树：哪些子目录/文件必需（视频/图像原始文件、标注、切分文件、norm stats…）；
3. 数据量级与统计（N 条样本 / 时长），以及训练前是否要先跑 norm stats 之类全量扫描。

数据集路径通过 `test/*.sh` 的 `--dataset_path` 传入，示例里写出实测目录（VAE 等辅助权重也属于前置必须备齐的资产）。

## 获取预训练权重

同样是三条：① 来源（HF hub / 上游 release / 内网镜像，给出 repo id 或 URL）；② 校验（`sha256sum` 或下载命令本身）；③ 目录布局（`--model_path`/`--base_checkpoint_path` 指向哪个子目录，是与 transformers 兼容布局还是原生布局）。把「随机初始化（不加载权重）也能跑通训练」写清楚，方便别人先验证链路再验证权重。
注意点：`config.json` 里的嵌套绝对路径（如 `backbone` 字段）指向作者机器，README 要提醒使用者改成本地路径。

# 快速开始

本任务主要提供**单机 8 卡**的训练脚本。

## 开始训练

- 精度：`bash test/train_full_8p.sh --dataset_path=xxx --model_path=xxx`
- 性能：`bash test/train_performance_8p.sh --dataset_path=xxx --model_path=xxx`

## 训练结果

**表 3** 训练结果展示表

| 芯片 | 卡数 | global batch size | Precision | max steps | loss | FPS |
| :--: | :--: | :---------------: | :-------: | :-------: | :--: | :-: |
| 竞品A | 8p | 256 | Mixed | 1000 | 0.1873 | 73.12 |
| Atlas 800T A2 | 8p | 256 | Mixed | 1000 | 0.1876 | 56.14 |

# 变更说明

## 变更

2026.xx.xx：首次发布。

## FAQ

<!-- ============================================================
  以下为模板说明，只给写 README 的人看，**交付前请连同注释一起删除**
  ============================================================

约定：
- 表格全局编号：表 1 昇腾版本 / 表 2 三方库 / 表 3 结果
- 结果表行必须成对：`竞品A` + `Atlas 800T A2`（或 A3 / Atlas 950）
- 无 GPU 基线时注明「未做 GPU 对比」，不要伪造竞品数据
- `变更` 是 append-only 带日期的，包括后续性能优化里程碑
- 常见 FAQ：HF 离线镜像 + `HF_HOME`、权重目录结构树、`bc: command not found`、如何做开环评估

章节频率排序参考（截至 2026-09，会随仓演进）：`## 模型介绍` 51 / `## 代码实现` 46 / `# 简介` 45 / `## 目录` 44 / `## 支持任务列表` 39 / `# 变更说明` 37 / `## FAQ` 32。最佳范本：`SparseDrive/README.md`（最完整）、`Pi-0/README.md`（VLA 最干净）。

前面各节里凡是形如「本节必须可复现…」的行、以及指向 skill 内部文档的 `references/…` 路径，都是模板说明，**都不要带进交付件**。
============================================================ -->
