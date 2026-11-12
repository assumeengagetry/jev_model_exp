# 源码存放位置与公开范围

[返回总览](README.md) · [人工复现](REPRODUCTION.md) · [公开源码清单](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)

**Git 只保存研究文档、表格和聚合证据，执行脚本保存在本机的库外目录，没有推送到 GitHub。** 其他文档中提到的 `.py` 文件都按本文定位；在其他机器上需要由资产持有者提供。

## 本机归档位置

研究库的相邻目录：

```text
../jev-辅助归档/2026-11-10方案说明与近期报告/
  draft-before-pull/
    reports/current-scheme-20261110/source/
      five_dimension/          第一段 prepare/label/train/coordinator 与 Engine
      joint/                   第二段数据、父参数控制、train、Engine、正式评测
      shared/                  renderer、TokenStore、原子组批、Teacher labeler、FA3 确定性包装
    reports/readout-study-20261108/source-20261110/
                               8 个源文件：完整题准备、readout、metrics、completion reader
    artifacts/inference-prefix-graphs-20261110/code/
                               fast_pair_engine / prefix_graph_runtime / run_public / compare_results
  extra-shared/student_flash_attention3.py
  PLACEMENT.json               97 份原材料与新位置的映射
```

原远端运行目录和 /tmp 下的采集副本没有修改或清理。

## FA3 由三项依赖组成

| 依赖 | 作用 |
|---|---|
| `student_flash_attention3.py` | 加载 FA3 kernel，并注册为 Transformers 的 attention 实现 |
| `student_flash_attention3_repro.py` | 确定性 forward/backward 包装、计数和检查；直接 import 上一项 |
| FA3 kernel 构建 | 需与 Torch 2.9、CUDA、Hopper 匹配；有脚本不代表 kernel 已安装 |

## 源码之外还需要的资产

- Teacher 发布权重旁的 `knowline_layout/knowline_server`，它们不属于本项目的自有源码。
- 数据构池、固定索引、token streams、effective targets、共同初始化和模型权重。

## 源码清单记录了什么

[SOURCE_MANIFEST.json](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json) 记录只读读取的执行源码文本及其字节对照，路径为采集时和本地归档的位置。它不含模型或语料的 hash，也不代表运行测试通过。配置 JSON 是字段摘录，路径已脱敏，需要适配路径后才能作为 config 使用。
