# 实际源码存放与公开边界

[返回总说明](README.md) · [人工复现手册](REPRODUCTION.md) · [公开源码清单](../../../03-证据与图表/06-五维训练与Readout/current-scheme-20261110/SOURCE_MANIFEST.json)

按已拉取的研究库整理约定，**Git只保存研究文档、表格和聚合证据；执行脚本不强行加回Git**。因此其它文档中的“库外源码”是实际审计依据的定位说明，不是一个承诺GitHub能点击下载的链接，也不是已公开的独立运行包。

## 本机归档位置

研究库的相邻目录：

```text
../jev-辅助归档/2026-11-10方案说明与近期报告/
  draft-before-pull/
    reports/current-scheme-20261110/source/
      five_dimension/          第一段prepare/label/train/coordinator与Engine
      joint/                   第二段数据/父参数控制/train/Engine/正式评测
      shared/                  renderer/TokenStore/原子组批/Teacher labeler/FA3确定性包装
    reports/readout-study-20261108/source-20261110/
                               8个完整题准备、readout、metrics与completion reader源文件
    artifacts/inference-prefix-graphs-20261110/code/
                               fast_pair_engine/prefix_graph_runtime/run_public/compare_results
  extra-shared/student_flash_attention3.py
  PLACEMENT.json               97份原材料与新位置映射，保全原始草稿
```

该目录只在当前本机存在，**未推送GitHub**。其它机器需由资产持有者提供相应源码；不能将研究库里的公式说明冒认为全部执行依赖已经公开。原远端运行目录和/tmp采集副本也未修改或清理。

## FA3是三项依赖，不是一份文件

1. `student_flash_attention3.py`：原FA3 kernel加载与Transformers attention注册层。
2. `student_flash_attention3_repro.py`：确定性forward/backward包装、计数和检查；直接import上一项。
3. 与Torch2.9/CUDA/Hopper匹配的实际FA3 kernel构建；脚本存在不等于kernel已安装或能在任意GPU运行。

Teacher还需要其发布权重旁的 `knowline_layout/knowline_server`；它们不是本次自有源码发布。数据构池、固定索引、token streams、effective targets、共同初始化和模型权重也须另外准备。

## 证据含义

源码清单记录只读读取的实际执行文本及字节对照，路径是当时采集／本地归档位置；**不是模型或语料hash，也不是运行测试通过证据**。JSON配置为字段摘录及路径脱敏，不适合作为未经路径适配即可执行的原config。
