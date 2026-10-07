# LLM 面试手撕与推导

启动 `./start.sh` 后打开 `http://127.0.0.1:8765/interview`。本页全部内容随项目保存在本地，断网可看。

按现有模板的 EASY/HARD 思路使用：

1. **原理**：先看问题动机、符号定义、适用条件，明确哪个分布负责采样。
2. **数学推导**：EASY 默认展开核心公式、逐步推导和小例子；HARD 保留原理与符号，让你先闭卷推导，再展开核对。
3. **手撕代码**：点“理解了，开始手撕代码”，展开实现任务与自检点；有对应模板时跳到同难度的工作台。OPD 有独立的 EASY/HARD 代码模板。
4. **复盘**：完成面试追问再标记掌握，进度保存在当前浏览器。

## 目录

- 数学基础：交叉熵、KL、策略梯度、GAE
- 对齐算法：PPO、DPO、GRPO、OPD（此处指 On-Policy Distillation）
- Transformer：注意力、RoPE、KV Cache、RMSNorm、SwiGLU
- 训练与推理：LoRA、梯度累积、BPE、采样、精度、FlashAttention
- 工程追问：mask、log probabilities、padding、训练数据泄漏与评估

公式用本地 KaTeX 排版，JS、CSS、字体全部随项目保存。原论文链接放在各题卡片底部，只有主动访问论文才需要联网。

数学内容与代码提示在 2026-10-07 做了校验，详见 [校验记录](MATH_REVIEW.md)。区分论文目标与练习变体：DPO 模板用 token 均值，GRPO 模板用全 batch token 均值，OPD 固定学生采样前缀后反传完整词表 KL。

校验命令（不执行未完成的算法 TODO）：

```bash
.venv/bin/python scripts/check_math.py
node scripts/check_interview.js
```
