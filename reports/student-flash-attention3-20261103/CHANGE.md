# Student 显式 FlashAttention-3

用户要求：只管 Student 训练；明确使用 FlashAttention，不接受 SDPA 自动选择或静默回退。H800 优先采用针对 Hopper 优化的 FA3。

## 范围

- 当前 fresh-Qwen 100万 pair 三臂训练；Teacher 标注、Teacher 环境和正在进行的 BASE_QWEN 评测均不改、不重启。
- 保持原 Torch2.9.1+cu128、Transformers4.57.6、PEFT0.18.1；不安装依赖升级包或 CUDA toolkit。
- 采用 `kernels-community/flash-attn3` 中匹配 `torch29-cxx11-cu128-x86_64-linux` 的预编译 FA3（含 forward/backward/varlen），放在项目独立内核目录，不加入全局 Python site-packages。
- 通过 Transformers 的公开 AttentionInterface / mask 注册接口，使用显式实现名 `flash_attn3_hopper`。它是原生 FA3 内核的本地注册名，不是 SDPA，也不是不含实现的 PyPI 占位包。
- 训练构造器明文 `attn_implementation="flash_attn3_hopper"`；配置、日志及最终训练记录均写 FlashAttention-3，记录实际 provider。若内核缺失、ABI/硬件不匹配或模型没有采用指定实现，直接报错；没有 SDPA fallback。
- 不改变起点、100万样本、四科分布、seed、LoRA、学习率、batch、训练量或评测流程。不得为验收额外跑模型测试/烟测/哈希；导入原生扩展属于安装检查，真实前后向等正式训练开始后记录。

## 清单

- [x] 核对 H800、CUDA12.8、Torch2.9/CXX11 ABI、BF16、Qwen head_dim128、dropout0 与原生FA3接口兼容条件。
- [x] 获取并导入匹配版本的预编译 FA3。
- [x] Student 专用 FA3 注册与禁止回退守卫。
- [x] 原子更新当前 Student trainer/config，保留修改前备份。
- [x] 独立只读评审、核对实际部署与原 Teacher/基线进程未变。
- [ ] 正式 Student 训练首次前后向成功后，依据日志核实真实 FA3 算子执行；不在正式训练开始前冒报完成。

## 实际执行依据

- 首次集群直连传输超时，仅取得约1.39MB原生扩展片段；保留原片段，未回退或修改核心库。
- 改为本地下载同一公开资产版本并SCP，804,184,680字节原生扩展及配套Python文件完整传入；2026-11-03 07:06 UTC原生导入、dense/varlen及backward导出检查完成。没有构造模型或运行GPU前后向。
- 独立评审未发现阻断性源码问题；实际兼容性不以该评审代替。
- 07:11:59 UTC原子更新当前Student trainer和config；真实FA3 provider为kernels-community/flash-attn3，显式注册名flash_attn3_hopper。config/训练日志/完成记录均标FA3，禁止SDPAfallback。未安装PyPI同名占位包，未修改全局site-packages。
- 07:16:40 UTC再次核对：三份部署文件与准备内容逐字一致；100万pair、四科各25万、共同未训起点、seed42、LR1e-4、全局batch128、一轮、LoRA等均未变。
- Torch2.9.1+cu128、Transformers4.57.6、PEFT0.18.1版本未变；Teacher162355/162356/162357及基线父进程162358的PID/starttime均保持原值，未发停止或重启信号。
- 当前仍为Teacher标注＋BASE_QWEN评测，三组Student均尚未开始。因此这里只证明依赖、注册与配置完成，不宣称实际GPU训练或加速倍数已经验证。
- 成功凭据：installation-local-transport.log、deployment.log、verification.json；installation.log保留第一次超时历史，不能把它当成最终状态。

## 代理核对

- 2026-11-03实测：本地和集群的127.0.0.1:10808均监听，显式HTTP CONNECT访问HF均200；直接SOCKS5握手/CONNECT/TLS访问也均HTTP200。
- 当前工具进程与非交互SSH shell均未设置用户列出的六个代理环境变量。原手写socket直连下载函数也不自动读取这些环境变量。
- 因此之前超时不能归因于代理本身不可用；只说明该下载路径未显式应用用户给定的应用层代理。未为此改动Teacher进程环境或全局shell配置。
