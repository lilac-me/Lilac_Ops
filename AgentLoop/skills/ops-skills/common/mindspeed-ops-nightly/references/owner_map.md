# 负责人映射（owner_map）说明

> **2026-09-08 脱敏改造**：为满足 skill 合入开源仓库的合规要求，负责人名称映射已从
> `scripts/summarize.py` 内嵌代码**外置**到运行环境的 NFS 目录，开源仓库内不再包含任何人名。

## 读取位置

- 文件：`/mnt/share/ops_nightly_agent/owner_map/owner_map.json`
- 即 `scripts/summarize.py` 头部 `OWNER_MAP_FILE` 指向的路径（可用环境变量 `OWNER_MAP_FILE` 覆盖）

## 格式

```json
{
  "chunk_bwd_dh": "owner-A",
  "wy_fast": "owner-B"
}
```

- key：算子名 / 单测文件基名（不含 `test_` 前缀、不含 `.py`）
- value：负责人标识（内部环境可填真实姓名，开源环境可填脱敏编号）
- 读取失败 / 文件缺失时 `summarize.py` 自动回退为空映射，报告中的 owner 显示为"未知"，不影响报告生成

## 维护

- 更新映射后无需改动任何 skill 代码，下次运行 `summarize.py` 即生效
- `python scripts/check_skill.py` 会校验：summarize.py 采用动态加载（无内嵌人名）且 NFS owner_map.json 可读

注：UT 失败用例按其测试文件名（`unit_tests/.../test_<op>.py`）匹配 `test_` 前缀后的算子名。
