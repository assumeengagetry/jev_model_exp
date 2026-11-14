# Fresh 四臂：冻结10M D_select 中间观察

**结论范围：** seed42、更新365、相同5840次完整case曝光、9988659逻辑pair token。本页不是Public Index，不与51.99作数值比较，不据本节点选赢家、停止训练或修改LR／曝光。主代理于2026-11-12T13:27:27UTC完成独立CPU回算；没有新增模型forward、optimizer step或eval。

## 四臂聚合

主指标先对case内全部supported-known field取`q[argmax margin]`均值，再case／域均值，最后五域等权。不是逐field打散统计；q不事后归一化。

| 臂 | Target mass ↑ | NLL ↓ | Brier sum ↓ | 完整onehot case exact（259子集） |
|---|---:|---:|---:|---:|
| C_D_BCE | 0.4754056282 | 1.1893012354 | 0.5864355014 | 114/259（44.02%） |
| C_P_BCE | 0.4968982522 | 1.2002228905 | 0.5840132656 | 121/259（46.72%） |
| C_D_CE | 0.5387655985 | 1.0677630039 | 0.5537759034 | 136/259（52.51%） |
| C_P_CE | 0.5208058597 | 1.0740614145 | 0.5574684269 | 129/259（49.81%） |

四臂都是375case（每域75）、615known field全支持、5190完整candidate、90unknown field原样保留但不forward，known-field coverage均1.0。没有截断、候选删除、短case替补、prefix-cache或按正确性重试。

### 每域target mass

| 臂 | Knowledge | Language | Retrieval | Tools | Arts |
|---|---:|---:|---:|---:|---:|
| C_D_BCE | 0.4689096973 | 0.4259195834 | 0.4800000000 | 0.4688655270 | 0.5333333333 |
| C_P_BCE | 0.5055763640 | 0.4456049256 | 0.5288888889 | 0.4644210825 | 0.5400000000 |
| C_D_CE | 0.5274282151 | 0.5087267656 | 0.5644444444 | 0.4865619006 | 0.6066666667 |
| C_P_CE | 0.4940948818 | 0.4685862501 | 0.5422222222 | 0.4924592776 | 0.6066666667 |

Tools的验证源主要procedural；这不是BFCL或真实工具泛化结论。

## 成对case分析：百分点，不是Public Index点

同375case对齐，整case保留内部field依赖；五域内分别有放回抽75case，共10000次共同draw、seed20261112。以下95%区间是**单训练seed、未作多重比较校正的描述性case-bootstrap**，不覆盖训练seed变异，也不证明两布局等价。

| 对照（前者−后者） | Target mass差（百分点） | 描述性95% case区间 |
|---|---:|---:|
| CE−BCE，dense | +6.3360 | [+2.3639, +10.3571] |
| CE−BCE，packed | +2.3908 | [−1.1402, +5.8235] |
| Packed−dense，BCE | +2.1493 | [−1.1556, +5.3704] |
| Packed−dense，CE | −1.7960 | [−4.3079, +0.6454] |
| CE−BCE，两布局等权 | +4.3634 | [+1.2903, +7.5392] |
| loss×layout交互 | −3.9452 | [−8.0171, +0.1908] |

解释：CE在两布局下NLL和Brier均低于同曝光BCE；target mass方向也均为正，但packed的case区间跨0。Packing的差异方向随loss改变，BCE有109field／97case的argmax差异，CE有58field／54case差异，**当前不能宣称packing本身带来稳定学习收益，也不能说长期训练轨迹数值等价**。这不推翻首macro已通过原冻结梯度／实际step合同，二者证据范围不同。

## 真实曝光与系统计数分开

- 四臂训练均完成19977318真实forward token（两pass）及9988659 backward-real token；首macro只消费一次。
- 训练tokenwise输入行数计数器（每算子的两pass累计）：dense26812992，packed19977318。它说明声明的packed执行不含padding，不是逐kernel profiling或吞吐测量。
- 四臂D_select均完整执行1554484真实token、469wave；dense输入行数1586320，packed1554484。节点ledger的单次dev wall分别47.75／47.44／46.95／48.30s；**不据这一轮或不同完成时刻宣称稳定加速**。

## 独立核验与归档

`PRIMARY_VERIFICATION_AND_PAIRED.json`绑定原始COMPLETE、SUMMARY、case-metrics及节点前后checkpoint stat。主代理直接在远端从原始select roster核对每case／field身份、完整候选数／顺序、未归一化target、保存的两类logits与margin，再用独立stdlib log-sum-exp回算argmax／NLL／Brier／case／域聚合；最大field指标差1.56e-15以内。节点前后checkpoint同cursor365，dev未修改训练计数。

- `C_*/SUMMARY.view.json`和`COMPLETE.view.json`是**显式删去重复scope／checkpoint_identity的归档视图**，各带原artifact指针；不是冒充原JSON逐字节副本。原始完整scope/source字节均保留远端，并与OWN_SCOPE／冻结config核对。
- 原始case-metrics、target／logits／私有案例／token及模型tensor未下载到Git；只归档聚合、身份指针和实际分析源码。
- 10M文件事件观察器已正常退出；新30M观察器只读inotify/pidfd等待下个冻结节点。原训练源／配置和整队列控制器没有修改或重启。
- D_confirm与public仍未执行，本节点不解除cached serving／完整public／正式单卡延迟门禁。
