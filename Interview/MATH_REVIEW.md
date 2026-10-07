# 数学校验与实现约定

校验日期：2026-10-07。覆盖现有 34 道题的公式、条件、推导和数值例子，以及 PPO/DPO/GRPO/OPD 模板相关注释。所有练习空白仍供学习者实现。

## 修正与补充

| 内容 | 校验结果与修正 |
|---|---|
| 自回归交叉熵 | 预测目标为 `y_t`，条件只含 `x,y_<t`；区分预训练的全序列 loss 与 response-only SFT。全 mask batch 要有明确处理。 |
| KL 两个方向 | 教师作为第一个参数是 forward KL，学生作为第一个参数是 reverse KL；补充支撑条件和无穷大情形。模式覆盖/集中不是无条件结论。 |
| 策略梯度 | 动作无关 baseline 的零期望证明要求匹配采样分布；权重 stop-gradient。包含自身样本的组内均值不能直接套用该无偏证明。 |
| GAE | 区分真实终止、时间截断、bootstrap mask 和 trace mask；先生成原始 returns，再标准化 policy advantage。总体标准差避免单 token NaN。 |
| PPO ratio | 重要性变换在固定状态的动作分布上精确；单 token ratio 不会校正整条轨迹的状态分布。old 与 reference 各有用途。 |
| PPO clip | 用优势正负分别写出一侧截断，提供完整数值表与梯度检查；不能把全部 ratio 无条件截断。 |
| DPO 最优策略 | 补齐拉格朗日求导、归一化常数和 KL 非负性验证。β>0 且参考策略支撑兼容；有限参数模型未必表示闭式解。 |
| DPO 偏好 loss | 同一 prompt 的 logZ 才能相消；原始序列 log probability 是有效 response token 的和。本模板的 token 均值会改变目标。 |
| GRPO 标准差 | 显式约定总体标准差 `correction=0`，并同时解释样本标准差。G=1 和相同奖励时优势为零；epsilon 不能修复已经产生的 NaN。 |
| GRPO k3 KL | 修正方向为 KL(current∥reference)。`expm1(u)-u` 非负；固定状态且动作来自当前策略时数值无偏，复用 old 样本未经校正时不成立。数值无偏也不保证梯度无偏。 |
| GRPO 平均方式 | 原论文先每条回答平均再回答间平均；模板按全 batch 有效 token 平均。两者在回答长度不同的时候赋权不同。 |
| OPD | 以 GKD 的原论文解释学生轨迹采样与不反传采样分布的约定；forward/reverse KL 均可用于这种数据采集方式。 |
| OPD 梯度 | sampled log-ratio 的无偏数值不代表朴素反传正确；完整词表 reverse KL 要对学生概率权重和 log probability 一起求导。 |
| Attention / RoPE | 点积分数缩放的方差推导明确独立分量假设；RoPE 相对位置等式针对固定内容及相同频率；补充全 mask 行与旋转布局的边界。 |
| KV Cache / GQA | 固定权重、因果性、位置和 padding 一致是复用条件；缓存同时包括 K/V，区分总生成复杂度和单步复杂度、MB 与 MiB。 |
| FlashAttention | 在线合并中的 o 明确是未归一化分子；减少 IO 与存储不等于将普通 dense attention 的算术复杂度降为线性。 |
| 累积 / AdamW | 不等大小 micro-batch 需按最终归约单位加权；补齐 bias correction、step 编号与 decoupled decay。 |
| 推测解码 / pass@k | 补齐拒绝残差归一化和 Z=0 边界；pass@k 解释无放回组合概率及 iid 样本下的估计条件。 |

## 一手依据

- [PPO](https://arxiv.org/abs/1707.06347)：clipped surrogate。
- [GAE](https://arxiv.org/abs/1506.02438)：加权 TD residual。
- [DPO](https://arxiv.org/abs/2305.18290)：最优策略、隐式奖励与偏好损失。
- [DeepSeekMath](https://arxiv.org/html/2402.03300v3)：§4.1、式 (3)/(4)，GRPO 的平均方式与 KL 方向。
- [On-policy Distillation / GKD](https://arxiv.org/html/2306.13649)：§3.1，学生轨迹、token divergence 与停止采样分布梯度。
- [MiniLLM](https://arxiv.org/abs/2306.08543)：reverse KL 蒸馏。
- 其余题卡保留对应 Attention、RoPE、LoRA、FlashAttention、GQA、AdamW、推测解码和 pass@k 原论文链接。

## 可复现检查

在项目根目录运行：

```bash
.venv/bin/python scripts/check_math.py
node scripts/check_interview.js
```

Python 检查关键恒等式、小例子和梯度，包含特意对比“KL 数值估计正确但朴素梯度错误”的反例。Node 检查每题完整性、跳转与全部本地 TeX 严格排版。数值检查用于发现符号、索引和实现错误，不能代替推导，也不表示尚未完成的训练 TODO 已通过。
