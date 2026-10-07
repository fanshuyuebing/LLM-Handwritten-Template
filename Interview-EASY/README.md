# OPD 手撕练习（EASY）

先看 [面试题库的 OPD 推导](../Interview/README.md)，再实现 `opd.py` 的三个 TODO：学生自采样、masked KL、单步训练。TODO 下方有注释答案。

```bash
.venv/bin/python Interview-EASY/opd.py
```

这是共享词表、随机初始化的 CPU Mock 练习，只检查数据流与梯度；loss 不应被当作模型能力指标。学生采样步骤不反传梯度，后续在固定的学生前缀上更新学生分布。
