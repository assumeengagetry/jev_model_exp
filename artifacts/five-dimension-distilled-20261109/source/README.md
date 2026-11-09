# 源码角色

本目录保存本次实际执行的训练准备、语言Teacher包装器、严格Student trainer与队列控制脚本。它们仍依赖原训练缓存及既有共享runtime；这不是在任意机器从零重建全部训练语料的承诺。

正式推理用的 `decision_engine.py` 单独实现完整 pair 渲染与margin读出，不依赖训练缓存或Teacher。其审查和完整公共运行尚待完成，不能把仅有源码当成已验证可提交。

远端正式namespace：`/tmp/opencode/jev-five-dimension-distilled-20261109-v2`。禁止运行旧扩大量续训入口、已取消第十波入口或本轮被替代的初稿CPU准备队列。

用户明确禁止额外测试／烟测／hash，因此未执行这些步骤；真实执行证据单独存于上级材料和 `reports/five-dimension-distilled-submission-20261109/`。
