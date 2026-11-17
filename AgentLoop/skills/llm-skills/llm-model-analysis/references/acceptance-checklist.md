# 验收清单

## 目录

- 结构分析验收
- Shape 验收
- NPU 验收
- FLOPs/MFU 验收
- 最终机器验收

## 结构分析验收

- [ ] 配置参数表完整；
- [ ] 层类型和层索引已展开；
- [ ] 参数量从模块到层到整网；
- [ ] Total 与 Active 参数区分；
- [ ] 参数量表的 Total 和 Active 均同时展示精确整数与十进制 B 值；
- [ ] B 值由精确值除以 `1_000_000_000` 生成，舍入和小数位符合结构参数规范；
- [ ] 创新点有基线对比；
- [ ] 训练与推理路径区分；
- [ ] 所有结论有源码依据。

## Shape 验收

- [ ] 整网主流程完整；
- [ ] 每个 Tensor 有名称和 Shape；
- [ ] 每个运算有独立节点；
- [ ] 权重有 Shape；
- [ ] Einsum/MatMul 缩并维清楚；
- [ ] Cache 和 Mask 有可视化；
- [ ] 复杂流程有实际小例子；
- [ ] 训练相关前向分支已覆盖；与训练实现存在实质差异的推理分支已说明；
- [ ] 至少完成一次静态或运行时校验。

## NPU 验收

- [ ] 非 Eager 路径无遗漏；
- [ ] 每个自定义 Kernel 有来源；
- [ ] Forward/Backward 分开判断；
- [ ] dtype 和 Shape 泛化性已检查；
- [ ] Fallback 已检查；
- [ ] 分布式算子已检查；
- [ ] 风险已分级；
- [ ] 每个缺口有验证或适配建议。

## FLOPs/MFU 验收

- [ ] FLOPs 口径明确；
- [ ] 各模块公式来自实际 Shape；
- [ ] MoE 使用 Active FLOPs；
- [ ] Sparse/Linear/Indexer 单独推导；
- [ ] MTP 已考虑，重计算口径已明确不计入理论 FLOPs；
- [ ] MFU 峰值精度明确；
- [ ] Step Time 口径明确；
- [ ] 至少一种交叉验证完成；
- [ ] 所有近似项已标注。

## 最终机器验收

最终交付前必须执行可自动化验收；任一强制项失败时继续修正，不得交付：

1. 输出目录恰好包含 `model_analysis.md` 与 `model_analysis.json` 两个最终文件；
2. Markdown 的二级标题恰好等于第 12 节规定的五个标题，名称和顺序完全一致；
3. JSON 可被标准 Parser 解析，顶层字段与第 11.1 节一致；
4. `layer_types` 主层 Count/Indices 与配置逐项一致，附件层单独验收；
5. JSON 参数总量、Active 多口径、附件参数与独立公式计算一致；
6. Markdown 中的参数精确值和 FLOPs 数值与 JSON 精确一致，所有参数 B 列均能从对应精确值重新换算；
7. 至少重算一个短序列和一个目标长度的模块/整网 FLOPs；
8. NPU `overall_as_is` 与 Matrix 中最严重的主路径阻塞项一致；
9. 没有运行环境时，Markdown 和 JSON 均明确写未运行，不得出现真实验证措辞；
10. 所有公开源码引用包含 Repository 与不可变 Revision，所有外部证据标明独立实现范围。

推荐使用临时脚本执行以下断言：

```text
exact_h2_headings == expected_five_headings
json_parse == pass
sum(main_layer_type_counts) == configured_main_layers
parameter_formula_values == json_parameter_values
markdown_parameter_billions == format_billions_by_level(json_parameter_values / 1_000_000_000)
flops_formula_values == json_flops_examples
markdown_key_values == json_key_values
deliverable_files == {model_analysis.md, model_analysis.json}
```

---
