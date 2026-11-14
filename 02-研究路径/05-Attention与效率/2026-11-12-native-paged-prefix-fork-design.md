# 原生FA3 paged KV的只读prefix fork候选

**状态：候选已实际运行；API组合可执行，但整体数值未准入。** 自写Triton prefix_block fork在真实case的原冻结margin合同未过；不修改失败结果、不放宽阈值。研究一个使用已装原生FA3、单联合softmax的替代执行器；训练仍用支持真实deterministic backward的原生varlen，绝不训练穿过forward-only cache API。

## 已实际读取的本机接口

远端`.../flash-attn3-torch29-cu128/flash_attn_interface.py`的`flash_attn_with_kvcache`明确提供`page_table`、`cu_seqlens_q`、`max_seqlen_q`、`cache_seqlens`和`num_splits`；cache的page_block_size可为1。只有传入k/v新值时才原地更新cache；本方案始终`k=None,v=None`，RoPE已由原模型施加，不再传rotary_cos/sin。

## 数据布局与不变量

- prefix预填只计算真实P个token，保存视图`K/V[P,8,128]`。
- 每层request-local KV backing为`[P+C,8,128]`，C取本group已知实际suffix波次的最大真实token数，不按ctx上限／Graph bucket向上取整；它是KV存储容量，不作为Embedding/Norm/MLP/QKV业务行数。
- 原prefix只复制一次到该backing后改由真实P视图持有，释放原临时prefix tensor；不按候选repeat/cat prefix KV。suffix实际T投影后只写`[P:P+T]`，绝不写`[:P]`。
- 原生paged cache使用page_block_size=1，真实page索引为前P个共享prefix pages＋本branch的suffix pages；page_table尾部仅metadata，不得读取超过cache_seqlens的页。
- q始终真实`[sumS,16,128]`，cu_q按本branch suffix长度；k/v cache为真实存储块视图`[P+C,1,8,128]`；no KV head repeat、no业务pad。每branch key长度`P+S_i`。
- causal分支native `causal=True`利用bottom-right对齐，prefix_block分支`causal=False`看本branch完整suffix；只一个P+S联合softmax，不独立归一化后相加。
- `num_splits=1`冻结为单KV归约策略；不传不存在的deterministic/dropout参数，不称cache API支持backward。cu_q packed和paged block1实际兼容性必须GPU验证，不凭doc宣称通过。
- prefix_block仍仅复用完整、完全相同的双向prefix；bidirectional禁止复用，完整P+S计算。

## 入口与所有权

独立源码`/tmp/opencode/jev-native-paged-fork-20261112-local/`；不改已冻结模型或正在运行的source。

`PagedPrefixOwner`绑定mode、model/core owner/参数版本、实际P、当前CUDA stream及C；每层prefixViews长度P。`fork(q,k,v,layer,cu,lengths,max_length)`在同一request stream上先写本T再调用原生cache，无其它CUDA stream并发复用。不静默退回SDPA／Triton／重复全pair。

## 验收边界

先readonly review，再用原真实macro中的完整多候选prefix group比较full-native、原Triton和新paged执行器的raw logits与margin，原容差不变。另核对prefix前后逐tensor不变、换序/换邻居/owner A→B→A、每层actual P/T、PageTable/cu/长度及内存容量报告。

已知BF16批形状/TF32/RoPE精度也在独立定位；不能将其修正收益全部归给paged kernel。即使新fork通过，也不是学习提分或clean SOTA结果，正式全量public和单卡延迟仍须真实执行。

## 真实执行回执

- 独立readonly review发现inference-mode构造／no-grad写入上下文不一致的P2；主代理核对后收紧整个owner生命周期必须inference_mode，入口在写入前拒绝no_grad。真实case已验证该拒绝行为，不以默认context碰巧相同掩盖接口漏洞。
- GPU3实际执行完整原多候选组，P333、S=[11,12]、C23、28层、单group全部2候选保留。packed3D Q＋cuQ＋page1实机可执行，每mode三波共84次cache API返回，末尾显式CUDA同步后才签execution_complete。计数仍是host API提交/返回，不谎称kernel launch/backward次数。
- backing每K或V形状[356,8,128]，全28层KV容量40828928bytes；P视图真实[333,8,128]。prefix每层K/V各copy一次，实际逐tensor复制准确／执行前后不变；换序逐位相同、同序重复逐位相同。
- causal full/paged raw相对RMS0.0071273、margin0.0060027，通过此组原合同。
- **prefix_block失败**：raw0.0556076、margin0.0490313（maxabs0.046875）；原Triton对该组仍raw0.0317146／margin0.0523001。没有放宽阈值或改写原失败，不用于该mode开发集加速。
- 本次3.3460s、峰值allocated1502660608bytes，含诊断snapshot与初始化；不是正式吞吐/稳定速度结论。不是整体serve准入，A→B→A、任意多组覆盖及完整public仍未执行。

结果及源码已回读到仓库`results/native-paged/`及源码snapshot。此方向先停止盲目换kernel；fullrow训练不依赖它，shape差异与真正内容依赖在独立精度研究中分别判定。
