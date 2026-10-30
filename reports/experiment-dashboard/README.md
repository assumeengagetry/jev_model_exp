# Language实验雷达图

直接打开`index.html`即可使用，不需要联网或安装前端依赖。

功能：95个历史完整Language参评终点，自动接入后续完整的专用模型/融合结果；九项Skill雷达、单次/seed均值、阶段/预算/rank搜索筛选、差值参照、SVG和CSV导出；附输入长度benchmark和实验路线。

更新数据：

```bash
~/assumeengage/temp_env/train/bin/python build_data.py
```

HTTP预览（从本目录启动）：

```bash
python -m http.server 8765 --bind 127.0.0.1
```

只读取既有结果，不运行模型。3M的batch/LR pilot在实验路线中单列，不填造九项Language雷达分数。新正负比例四组已从最新完整JSON接入。
