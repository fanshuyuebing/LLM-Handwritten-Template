/* Offline lessons. TeX, definitions and examples are stored locally. */
window.INTERVIEW_QUESTIONS = [
  {
    "id": "ce",
    "group": "数学基础",
    "title": "从最大似然推导交叉熵",
    "kind": "推导",
    "level": "高频",
    "question": "给出一段 token 序列，写出自回归语言模型的似然、负对数似然与忽略 padding 的 loss。",
    "hint": "先用链式法则分解 p(x₁,…,xₜ)，再取负对数。",
    "formula": "p(y|x)=∏ₜ p(yₜ|x,y_<t)；L=−Σ m·log p / Σ m",
    "derivation": [
      "链式法则：先固定 prompt x，再逐步写出 response 的条件概率。",
      "对序列概率取 log：log p(y|x)=Σₜ log p(yₜ|x,y_<t)。最大化它等价于最小化负号后的和。",
      "真实标签是 one-hot 分布，其交叉熵正好等于真实 token 的负 log probability。",
      "除以有效 token 总数得到 token 平均；mask 只选择训练目标，不移除输入上下文。"
    ],
    "handwrite": "手写稳定 log_softmax + gather + loss_mask；用极大 logit 检查不会溢出。",
    "followup": "为什么不能先 softmax 再 log？如何处理所有位置都被 mask 的样本？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/1706.03762",
    "principle": "语言模型给每个真实的下一 token 分配概率。最大似然要求真实序列的概率尽可能大；取负对数后，连乘变成可逐位置相加的损失。",
    "math": [
      "p_\\theta(y\\mid x)=\\prod_{t=1}^{T}p_\\theta(y_t\\mid x,y_{<t})",
      "\\mathcal L_{\\mathrm{NLL}}=-\\frac{\\sum_{b,t}m_{b,t}\\log p_\\theta(y_{b,t}\\mid x_b,y_{b,<t})}{\\sum_{b,t}m_{b,t}}"
    ],
    "symbols": [
      {
        "symbol": "x",
        "meaning": "提示或上下文"
      },
      {
        "symbol": "y_t",
        "meaning": "第 t 个目标 token"
      },
      {
        "symbol": "m_{b,t}",
        "meaning": "位置是否计入损失，取 0 或 1"
      }
    ],
    "conditions": [
      "平均损失要求有效 token 总数大于 0；全 mask 的 batch 应跳过或明确定义为零。",
      "response-only SFT 才屏蔽 prompt；普通语言模型预训练通常也学习上下文中的有效 token。"
    ],
    "example": "两个有效 token 的目标概率为 0.5 和 0.25：序列概率 0.125，负对数似然总和约 2.0794，token 平均约 1.0397。",
    "checks": [
      "logits[:, :-1] 与 targets[:, 1:] 对齐。",
      "把 -100 换成安全索引后 gather，再应用 mask。",
      "极大 logits 不导致 NaN/Inf。"
    ],
    "stepMath": [
      "\\log p_\\theta(y\\mid x)=\\sum_t\\log p_\\theta(y_t\\mid x,y_{<t})",
      null,
      "H(e_y,p)=-\\sum_v(e_y)_v\\log p_v=-\\log p_y",
      null
    ]
  },
  {
    "id": "kl",
    "group": "数学基础",
    "title": "KL、交叉熵与两种方向",
    "kind": "推导",
    "level": "高频",
    "question": "写出 KL(p∥q) 与 KL(q∥p)，说明为什么两者不对称，以及蒸馏时前向/反向 KL 的不同倾向。",
    "hint": "KL(p∥q)=Eₚ[log p−log q]；展开为交叉熵减熵。",
    "formula": "KL(p∥q)=Σᵢ pᵢ log(pᵢ/qᵢ)=H(p,q)−H(p)\nKL(q∥p)=Σᵢ qᵢ log(qᵢ/pᵢ)",
    "derivation": [
      "优化 q 时，KL(p∥q) 中 H(p) 是常数，等价于教师分布上的交叉熵。",
      "KL(q∥p) 在学生分布 q 下取期望，学生较少访问的区域贡献小；常呈现更集中的模式选择倾向。",
      "不能把两者的采样分布、梯度估计或零概率行为混为一谈。"
    ],
    "handwrite": "给 p=(0.9,0.1), q=(0.5,0.5)，手算两个 KL 并比较。",
    "followup": "若 pᵢ=0 且 qᵢ>0，两个方向各有什么数值问题？",
    "source": "https://arxiv.org/abs/2306.08543",
    "principle": "KL 比较两种分布在同一事件空间上的差异。第一个参数决定期望由谁加权，所以交换参数会改变损失和梯度。",
    "math": [
      "D_{\\mathrm{KL}}(p\\Vert q)=\\sum_i p_i\\log\\frac{p_i}{q_i}=H(p,q)-H(p)",
      "D_{\\mathrm{KL}}(q\\Vert p)=\\sum_i q_i\\log\\frac{q_i}{p_i}"
    ],
    "symbols": [
      {
        "symbol": "p",
        "meaning": "固定的教师或目标分布"
      },
      {
        "symbol": "q",
        "meaning": "可训练的学生分布"
      }
    ],
    "conditions": [
      "约定 0·log(0/q)=0；若 p>0 而 q=0，则 KL(p∥q)=∞。",
      "模式覆盖/集中的说法是受模型容量等约束时的常见倾向，不是无条件定理；能精确表示 p 时两个方向都在 q=p 处取最小值 0。"
    ],
    "example": "p=(0.9,0.1)、q=(0.5,0.5)：KL(p∥q)≈0.3681，KL(q∥p)≈0.5108；自然对数单位为 nat。",
    "checks": [
      "相同分布时 KL≈0，两个方向通常不同。",
      "教师分布不接收梯度。",
      "明确区分精确词表求和、随机估计和实际用于反传的损失。"
    ]
  },
  {
    "id": "pg",
    "group": "数学基础",
    "title": "策略梯度与 baseline",
    "kind": "推导",
    "level": "高频",
    "question": "从 J(θ)=E_{y∼πθ}[R(y)] 推导 REINFORCE，解释 baseline 为什么不引入偏差。",
    "hint": "用 ∇π=π∇logπ；R 不必可微。",
    "formula": "∇θJ=E_{y∼πθ}[R(y)∇θ logπθ(y)]\nE[b(x)∇θlogπθ(y|x)] = b(x)∇θΣᵧπθ(y|x)=0",
    "derivation": [
      "对离散采样分布求导：∇Σᵧπθ(y)R(y)=Σᵧπθ(y)R(y)∇logπθ(y)。",
      "自回归序列的 logπ 是各 token log probability 的和。",
      "只依赖状态/提示、不依赖当前动作的 baseline 可降低方差而不改变期望梯度；经验估计仍可能有高方差。"
    ],
    "handwrite": "手写 logits → log_prob → (reward−baseline)·log_prob 的 batch loss。",
    "followup": "为什么把当前动作的 reward 当作 baseline 可能引入偏差？",
    "source": "https://arxiv.org/abs/1707.06347",
    "principle": "离散采样本身不易求导，但采样概率可以求导。用 log-derivative trick，把奖励变成提高或降低已采样动作概率的权重。",
    "math": [
      "J(\\theta)=\\sum_y\\pi_\\theta(y\\mid x)R(x,y)",
      "\\nabla J=\\mathbb E_{y\\sim\\pi_\\theta}[(R-b(x))\\nabla\\log\\pi_\\theta(y\\mid x)]",
      "\\mathbb E_{y\\sim\\pi_\\theta}[b(x)\\nabla\\log\\pi_\\theta(y\\mid x)]=b(x)\\nabla\\sum_y\\pi_\\theta(y\\mid x)=0"
    ],
    "symbols": [
      {
        "symbol": "R",
        "meaning": "与 θ 无直接依赖的奖励"
      },
      {
        "symbol": "b(x)",
        "meaning": "不依赖当前采样动作的 baseline"
      }
    ],
    "conditions": [
      "上述公式假设奖励本身不直接依赖 θ；若依赖，还要考虑相应导数。",
      "实现中 advantage/baseline 权重 stop-gradient。组内均值包含样本自身时，不可直接套用动作无关 baseline 的无偏证明。"
    ],
    "example": "二动作策略 p=(0.5,0.5)，奖励 (1,0)。期望奖励为 0.5；对 softmax logits 的梯度为 (0.25,−0.25)。加常数 baseline 后期望梯度不变。",
    "checks": [
      "用于最小化的 loss 有负号。",
      "reward/advantage.detach()，policy log_prob 保留梯度。",
      "逐 token log_prob 相加得到序列 log_prob。"
    ]
  },
  {
    "id": "gae",
    "group": "数学基础",
    "title": "GAE 的递推式",
    "kind": "推导",
    "level": "高频",
    "question": "从 TD residual 推导广义优势估计，写出从后向前的实现与终止 mask。",
    "hint": "δₜ=rₜ+γV(sₜ₊₁)−V(sₜ)。",
    "formula": "δₜ=rₜ+γ bₜV(sₜ₊₁)−V(sₜ)；Aₜ=δₜ+γλ cₜAₜ₊₁",
    "derivation": [
      "先计算一步 TD residual：奖励加可 bootstrap 的下一状态价值，再减当前价值。",
      "同一连续轨迹上，把未来 δ 按 (γλ) 的幂加权求和。",
      "提取第一项后得到从后向前的递推；bootstrap mask 和 trace mask 分别处理价值与轨迹边界。",
      "λ=0 得一步 TD；λ=1 时，对一段以真实终止结束的轨迹，得到折扣回报减当前 V。",
      "原始 A+V 给出价值训练目标。优势归一化只用于 policy loss，不能反过来改变该 target。"
    ],
    "handwrite": "给定 reward、value、done 三个长度为 4 的数组，不调用第三方 GAE 函数实现反向循环。",
    "followup": "λ=0 与 λ=1 分别接近什么？value bootstrap 在截断与真实终止时有何不同？",
    "link": "ppo",
    "source": "https://arxiv.org/abs/1506.02438",
    "principle": "GAE 把不同跨度的价值误差加权汇总。λ 控制更短的 TD 信号与更长的回报信号之间的权衡。",
    "math": [
      "\\delta_t=r_t+\\gamma b_t V(s_{t+1})-V(s_t)",
      "\\hat A_t=\\delta_t+\\gamma\\lambda c_t\\hat A_{t+1}",
      "\\hat R_t=\\hat A_t+V(s_t)"
    ],
    "symbols": [
      {
        "symbol": "b_t",
        "meaning": "bootstrap mask：真实终止取 0，时间截断可取 1"
      },
      {
        "symbol": "c_t",
        "meaning": "trace mask：本段轨迹内继续递推取 1，边界取 0"
      },
      {
        "symbol": "\\gamma,\\lambda",
        "meaning": "折扣系数与 GAE 加权系数"
      }
    ],
    "conditions": [
      "真实终止不 bootstrap；时间上限截断可以用终点 V，但不能把下一段 episode 的优势串进来。",
      "本仓库的固定长度练习将 response 末尾视为终止，取尾部 V=0；这不等于完整的时间截断处理。",
      "先用原始优势构造 value target，再单独标准化 policy advantage。"
    ],
    "example": "r=(0,0,1)、V=(0,0,0)、γ=1、λ=0.5，且末尾终止：δ=(0,0,1)，A=(0.25,0.5,1)。",
    "checks": [
      "反向循环结果与显式加权和一致。",
      "pad 和新 episode 不接入上一条轨迹。",
      "单个有效 token 的标准化不产生 NaN。"
    ],
    "stepMath": [
      null,
      "\\hat A_t=\\sum_{l\\ge0}(\\gamma\\lambda)^l\\left(\\prod_{j=0}^{l-1}c_{t+j}\\right)\\delta_{t+l}",
      null,
      null,
      null
    ]
  },
  {
    "id": "ppo-ratio",
    "group": "PPO",
    "title": "为什么 PPO 要用概率比",
    "kind": "推导",
    "level": "必考",
    "question": "从旧策略采集的数据出发，解释 rₜ(θ)=πθ(aₜ|sₜ)/πold(aₜ|sₜ) 为什么出现。",
    "hint": "旧策略采样，新策略优化；这是重要性采样。",
    "formula": "E_{a∼πθ}[A(a)] = E_{a∼πold}[(πθ(a)/πold(a))A(a)]\nrₜ(θ)=exp(logπθ(aₜ|sₜ)−logπold(aₜ|sₜ))",
    "derivation": [
      "固定状态 s，写出新策略动作期望 Σₐπθ(a|s)A(s,a)。",
      "逐项乘除 πold(a|s)，把权重移到 old 的动作分布里。",
      "实际样本的状态也来自 old；PPO 固定该状态分布构造局部代理目标。单 token ratio 不会校正全部状态访问概率。",
      "从 log_prob 差取 exp 得 ratio；多轮更新中保留采样时缓存的 old_log_prob。"
    ],
    "handwrite": "用 log_prob 计算 ratio，检查 θ=old 时 ratio≈1。",
    "followup": "old policy 与 reference policy 分别起什么作用？为什么不能混用？",
    "link": "ppo",
    "source": "https://arxiv.org/abs/1707.06347",
    "principle": "PPO 用旧策略生成样本，再用概率比衡量新策略给这些动作增加了多少权重。这个变换在固定状态上是精确的；沿整条轨迹的状态分布仍是旧策略的数据。",
    "math": [
      "\\mathbb E_{a\\sim\\pi_\\theta(\\cdot\\mid s)}[A(s,a)]=\\mathbb E_{a\\sim\\pi_{\\mathrm{old}}(\\cdot\\mid s)}\\left[\\frac{\\pi_\\theta(a\\mid s)}{\\pi_{\\mathrm{old}}(a\\mid s)}A(s,a)\\right]",
      "\\rho_t=\\exp(\\log\\pi_\\theta(a_t\\mid s_t)-\\log\\pi_{\\mathrm{old}}(a_t\\mid s_t))"
    ],
    "symbols": [
      {
        "symbol": "\\rho_t",
        "meaning": "当前动作的新旧策略概率比"
      },
      {
        "symbol": "d_{\\mathrm{old}}",
        "meaning": "旧策略诱导的状态分布"
      }
    ],
    "conditions": [
      "固定状态 s；目标动作的概率支撑必须包含在旧策略支撑中。",
      "PPO surrogate 使用旧状态分布和旧优势，不是对整个新策略回报的精确重要性采样。"
    ],
    "example": "旧策略给一个动作概率 0.2，新策略给 0.3，则 ratio=1.5。新旧策略完全相同且评分方式一致时 ratio=1。",
    "checks": [
      "old_log_prob 和 old advantage 固定且 detach。",
      "reference 用于正则，不能替代 ratio 分母中的 old。",
      "采样与打分的温度/过滤分布一致，否则 ratio=1 的检查失效。"
    ]
  },
  {
    "id": "ppo-clip",
    "group": "PPO",
    "title": "PPO Clipped Surrogate 分段解释",
    "kind": "推导",
    "level": "必考",
    "question": "写出 min(rA, clip(r,1−ε,1+ε)A)，按 A 正负解释什么时候梯度被截断。",
    "hint": "A>0 希望增大动作概率；A<0 希望减小。",
    "formula": "Lclip=E[min(rₜÂₜ, clip(rₜ,1−ε,1+ε)Âₜ)]\n训练时最小化 −Lclip",
    "derivation": [
      "A>0 时，r 超过 1+ε 后取截断项，继续增大概率不再增加代理目标。",
      "A<0 时，r 低于 1−ε 后取截断项，继续减小概率不再改善目标。",
      "clip 约束的是经验代理目标，不保证参数更新后所有状态的 KL 都严格受限；工程上仍监控 KL。"
    ],
    "handwrite": "手算 A=+2 与 A=−2、r∈{0.7,1.0,1.3}、ε=0.2 的每格目标。",
    "followup": "为什么不是简单地把 r 永远 clip 后再乘 A？",
    "link": "ppo",
    "source": "https://arxiv.org/abs/1707.06347",
    "principle": "裁剪限制已经朝有利方向走得太远的奖励，仍保留把不利更新拉回来的梯度。它是一侧裁剪的代理目标，不是把所有 ratio 都截断。",
    "math": [
      "f(\\rho,A)=\\min\\{\\rho A,\\operatorname{clip}(\\rho,1-\\epsilon,1+\\epsilon)A\\}",
      "f(\\rho,A)=\\begin{cases}A\\min(\\rho,1+\\epsilon),&A\\ge0\\\\A\\max(\\rho,1-\\epsilon),&A<0\\end{cases}",
      "\\mathcal L_{\\mathrm{policy}}=-\\mathbb E[f(\\rho_t,\\hat A_t)]"
    ],
    "symbols": [
      {
        "symbol": "A",
        "meaning": "固定的优势权重"
      },
      {
        "symbol": "\\epsilon",
        "meaning": "裁剪宽度，0<ε<1"
      }
    ],
    "conditions": [
      "讨论 f 对 ratio 的梯度时，A 是常量。",
      "裁剪不能保证最终策略 KL 严格不超阈值，仍需要监控。"
    ],
    "example": "ε=0.2、ratio=(0.7,1,1.3)：A=2 时 f=(1.4,2,2.4)；A=−2 时 f=(−1.6,−2,−2.6)。",
    "checks": [
      "A>0 且 ratio>1+ε 时该样本的 policy 梯度为零。",
      "A<0 且 ratio<1−ε 时梯度为零。",
      "计算 min 后取负号，与最大化目标的符号一致。"
    ]
  },
  {
    "id": "ppo-rlhf",
    "group": "PPO",
    "title": "RLHF 中的 token reward 与 value loss",
    "kind": "推导",
    "level": "高频",
    "question": "说明 response 最后位置 reward、逐 token KL 惩罚、value 预测如何拼成 PPO 训练信号。",
    "hint": "不要把 reward model、value model、reference model 看成同一个模型。",
    "formula": "rₜ = −β(logπold(yₜ|sₜ)−logπref(yₜ|sₜ)) + 1[t=T]RRM(x,y)\nLtotal=−Lclip + cᵥLvalue",
    "derivation": [
      "Reward Model 给完整回答一个终局分数；参考策略提供偏离惩罚。",
      "把终局分数加到最后有效 response token，其余位置保留逐 token KL shaped reward。",
      "value 网络估计状态价值供 GAE；对有效 response token 计算 policy/value loss，忽略 prompt 与 pad。",
      "上式中的采样 token 的 log-ratio 是 KL 的 Monte Carlo 估计，可逐点为负；真 KL 的期望非负。"
    ],
    "handwrite": "画出 prompt[Tp] + response[Tr] 的 log_prob、action_mask、reward、value shape。",
    "followup": "为什么 reference 冻结但 old policy 在 rollout 时保存？",
    "link": "ppo",
    "source": "https://arxiv.org/abs/2203.02155",
    "principle": "回答级奖励很稀疏。把终局奖励与逐 token 的参考策略惩罚拼在一起，再用 value/GAE 把这个信号传播到前面的动作。",
    "math": [
      "r_t=-\\beta\\log\\frac{\\pi_{\\mathrm{old}}(y_t\\mid s_t)}{\\pi_{\\mathrm{ref}}(y_t\\mid s_t)}+\\mathbf1[t=T]R_{\\mathrm{RM}}(x,y)",
      "\\mathcal L_{\\mathrm{total}}=\\mathcal L_{\\mathrm{policy}}+c_v\\mathcal L_{\\mathrm{value}}",
      "\\mathcal L_{\\mathrm{value}}=\\frac12\\operatorname{mean}_{m=1}\\max\\{(V_\\theta-\\hat R)^2,(V_{\\mathrm{clip}}-\\hat R)^2\\}"
    ],
    "symbols": [
      {
        "symbol": "T",
        "meaning": "最后有效 response token 的位置"
      },
      {
        "symbol": "R_{\\mathrm{RM}}",
        "meaning": "回答级奖励"
      },
      {
        "symbol": "V_{\\mathrm{clip}}",
        "meaning": "Vold+clip(Vθ−Vold,−εv,εv)"
      }
    ],
    "conditions": [
      "reward、returns、old values 在 policy/value 更新时固定。",
      "sampled log-ratio 是动作在 old 分布下的 KL(old∥ref) 估计，单个值允许为负。",
      "此处给出本仓库使用的 clipped value loss；其他 PPO 实现可以采用普通 MSE。"
    ],
    "example": "三个 response token、RM=2、β=0.1、三个 sampled log-ratio 均为 0.2，则 token reward=(-0.02,−0.02,1.98)。",
    "checks": [
      "终局 reward 放在最后有效 token，而不是 padding 最后一列。",
      "value 预测的是采取当前动作之前的状态。",
      "value target 用未标准化的 GAE+Vold。"
    ]
  },
  {
    "id": "dpo-opt",
    "group": "DPO",
    "title": "DPO：KL 约束目标到最优策略",
    "kind": "推导",
    "level": "必考",
    "question": "给定奖励 r(x,y) 和参考策略 πref，推导 KL 正则下的最优 π*。",
    "hint": "对每个 x 单独优化分布；加拉格朗日乘子保证 Σᵧπ(y|x)=1。",
    "formula": "maxπ E_{y∼π}[r(x,y)] − β KL(π(·|x)∥πref(·|x))\nπ*(y|x)=πref(y|x)exp(r(x,y)/β)/Z(x)",
    "derivation": [
      "固定 prompt x，加入约束 Σᵧπ(y|x)=1 的拉格朗日乘子 α。",
      "对每一个 π(y|x) 求导并令零，KL 的导数含 log(π/πref)+1。",
      "移项得到 log(π/πref)=r/β+α/β−1，指数化后所有回答共享一个归一化常数。",
      "用 Σπ=1 确定常数为 1/Z；还可把原目标写成 βlogZ−βKL(π∥π*)，用 KL 非负性验证最优解。"
    ],
    "handwrite": "在只有两个 response 的离散空间里，用具体 reward 和 πref 验证 π* 归一化。",
    "followup": "β 增大时 π* 如何变化？Z(x) 为什么不需要显式计算到最终 DPO loss？",
    "link": "dpo",
    "source": "https://arxiv.org/abs/2305.18290",
    "principle": "给定奖励时，希望把概率向高奖励回答移动，同时用 KL 防止离参考分布太远。对归一化概率分布求最优解，会得到指数重加权。",
    "math": [
      "\\max_\\pi\\sum_y\\pi(y\\mid x)r(x,y)-\\beta\\sum_y\\pi(y\\mid x)\\log\\frac{\\pi(y\\mid x)}{\\pi_{\\mathrm{ref}}(y\\mid x)}",
      "\\pi^*(y\\mid x)=\\frac{\\pi_{\\mathrm{ref}}(y\\mid x)e^{r(x,y)/\\beta}}{Z(x)}",
      "Z(x)=\\sum_y\\pi_{\\mathrm{ref}}(y\\mid x)e^{r(x,y)/\\beta}"
    ],
    "symbols": [
      {
        "symbol": "\\beta",
        "meaning": "正的 KL 正则系数"
      },
      {
        "symbol": "Z(x)",
        "meaning": "对同一 prompt 的全部回答做归一化"
      }
    ],
    "conditions": [
      "β>0；参考策略在待优化的回答上有支撑，且 Z 有限。",
      "推导是在所有归一化分布上求闭式解；有限容量模型未必能表示这个最优分布。"
    ],
    "example": "πref=(0.5,0.5)、r=(1,0)、β=1，则 π*=(e/(e+1),1/(e+1))≈(0.7311,0.2689)。",
    "checks": [
      "最优分布和为 1。",
      "奖励同时平移常数后 π* 不变。",
      "相同参考分布下 β→∞，π* 回到 πref。"
    ],
    "stepMath": [
      "\\mathcal F=\\sum_y\\pi_y[r_y-\\beta\\log(\\pi_y/\\pi_{\\mathrm{ref},y})]+\\alpha(\\sum_y\\pi_y-1)",
      "r_y-\\beta[\\log(\\pi_y/\\pi_{\\mathrm{ref},y})+1]+\\alpha=0",
      null,
      "J(\\pi)=\\beta\\log Z-\\beta D_{\\mathrm{KL}}(\\pi\\Vert\\pi^*)"
    ]
  },
  {
    "id": "dpo-loss",
    "group": "DPO",
    "title": "DPO：隐式奖励到分类损失",
    "kind": "推导",
    "level": "必考",
    "question": "将 π* 反解为奖励，再代入 Bradley–Terry 模型，得到 chosen/rejected 的 DPO loss。",
    "hint": "r(x,y)=βlog(π*(y|x)/πref(y|x))+βlogZ(x)。",
    "formula": "Δθ = [logπθ(y⁺|x)−logπref(y⁺|x)] − [logπθ(y⁻|x)−logπref(y⁻|x)]\nLDPO=−E log σ(βΔθ)",
    "derivation": [
      "先从最优策略的指数式反解奖励。",
      "Bradley–Terry 用 reward 差的 sigmoid 给出偏好概率。",
      "两回答共享 prompt，βlogZ 在相减时抵消；如果 prompt 不同，此步骤不成立。",
      "把 π* 参数化为 πθ，对观测到的偏好最大化似然，得到负 log-sigmoid 损失。",
      "对 Δ 求导得到负的梯度权重：尚未正确区分的偏好对会获得更强的更新信号。"
    ],
    "handwrite": "手写两个序列的有效 token log_prob 求和，再用 F.logsigmoid 算 loss。",
    "followup": "sequence log_prob 用 sum 与 mean 有什么差别？论文目标与仓库教学实现一致吗？",
    "link": "dpo",
    "source": "https://arxiv.org/abs/2305.18290",
    "principle": "偏好只告诉我们哪个回答更好。把奖励重写成 policy/reference 的 log-ratio，就能用同一 prompt 下的偏好对直接训练策略。",
    "math": [
      "r(x,y)=\\beta\\log\\frac{\\pi^*(y\\mid x)}{\\pi_{\\mathrm{ref}}(y\\mid x)}+\\beta\\log Z(x)",
      "\\Delta_\\theta=\\log\\frac{\\pi_\\theta(y^+\\mid x)}{\\pi_{\\mathrm{ref}}(y^+\\mid x)}-\\log\\frac{\\pi_\\theta(y^-\\mid x)}{\\pi_{\\mathrm{ref}}(y^-\\mid x)}",
      "\\mathcal L_{\\mathrm{DPO}}=-\\mathbb E_{\\mathcal D}\\log\\sigma(\\beta\\Delta_\\theta)",
      "\\frac{\\partial\\ell}{\\partial\\Delta_\\theta}=-\\beta\\sigma(-\\beta\\Delta_\\theta)"
    ],
    "symbols": [
      {
        "symbol": "y^+,y^-",
        "meaning": "同一个 prompt 的偏好/非偏好回答"
      },
      {
        "symbol": "\\Delta_\\theta",
        "meaning": "两回答的隐式 reward 差除以 β"
      }
    ],
    "conditions": [
      "采用 Bradley–Terry 偏好模型。",
      "πθ 替换理论 π* 做参数化拟合；不意味着一次更新就实现闭式最优策略。",
      "原始 DPO 的序列 log_prob 是 response token 的和。"
    ],
    "example": "πθ=πref 时 Δ=0，单对 loss=log2≈0.6931；若 β=0.1 且 Δ=2，loss≈0.5981。",
    "checks": [
      "同 prompt 的两个 βlogZ 项完全抵消。",
      "reference 无梯度，使用稳定 logsigmoid。",
      "β 固定时，增大 Δ 会减小单对 loss。"
    ]
  },
  {
    "id": "dpo-implementation",
    "group": "DPO",
    "title": "DPO 的 mask、长度偏差与 beta",
    "kind": "手撕",
    "level": "高频",
    "question": "一个样本包含 prompt/chosen/rejected，如何构造 labels，计算并聚合 response log probabilities？",
    "hint": "prompt 和 pad 都设 -100；logits 与 labels 要错位。",
    "formula": "labels=[−100 × prompt_length, response_token_ids, −100 × pad_length]\nlogp(y|x)=Σₜ response_maskₜ · logπ(yₜ|x,y_<t)",
    "derivation": [
      "把 prompt+response+EOS 拼成输入；只在有效 response 与 EOS 位置提供 labels，其余为 -100。",
      "logits[:, :-1] 对齐 labels[:, 1:]；先用安全 token id gather，再将被忽略位置清零。",
      "原始 DPO 对有效位置求和。仓库模板对有效 token 取均值，这是长度归一化变体，会改变 loss 与隐式 reward 的定义。",
      "β 同时参与理论正则关系和偏好 logit 的尺度；不能仅凭 β 增减推断某次有限步训练的实际 KL。"
    ],
    "handwrite": "写 chosen/rejected 拼接、右填充、labels(-100) 的 collate_fn。",
    "followup": "chosen 和 rejected 分开 forward 与拼成一次 forward 时，padding 约束分别是什么？",
    "link": "env",
    "source": "https://arxiv.org/abs/2305.18290",
    "principle": "代码必须先算对每个回答的条件概率，再组合偏好损失。padding 和 prompt mask 的错误会直接改变学到的偏好。",
    "math": [
      "\\ell_\\theta(y\\mid x)=\\sum_{t=1}^{T_y}\\log\\pi_\\theta(y_t\\mid x,y_{<t})",
      "\\bar\\ell_\\theta(y\\mid x)=\\frac{\\ell_\\theta(y\\mid x)}{T_y}\\quad\\text{(length-normalized variant)}"
    ],
    "symbols": [
      {
        "symbol": "T_y",
        "meaning": "有效 response token 数，通常包括 EOS"
      },
      {
        "symbol": "\\ell,\\bar\\ell",
        "meaning": "序列 log probability 与 token 平均 log probability"
      }
    ],
    "conditions": [
      "把 log_prob 除长度会改变原始 DPO 目标，并非通用的“消除所有长度偏差”。",
      "现有模板采用 token 均值，属于明确的教学变体。"
    ],
    "example": "两 token 回答各 token 概率 0.5：sum≈−1.3863、mean≈−0.6931。四 token 回答 sum≈−2.7726，而 mean 仍≈−0.6931。",
    "checks": [
      "labels=-100 的索引不能直接传给 gather。",
      "prompt 和 pad 不进入序列 log_prob。",
      "chosen/rejected 若拼接成同一张量，需对齐到共同最大长度。"
    ]
  },
  {
    "id": "grpo-adv",
    "group": "GRPO",
    "title": "GRPO 组内相对优势",
    "kind": "推导",
    "level": "必考",
    "question": "同一 prompt 采样 G 个 response，如何在没有 value model 的情况下构造 advantage？",
    "hint": "组内均值作 baseline；标准差避免 reward 尺度差异。",
    "formula": "μ=ΣR/G；σpop=√(Σ(R−μ)²/G)；A=(R−μ)/(σpop+ε)",
    "derivation": [
      "将 [B·G] rewards reshape 为 [B,G]，每行对应一个 prompt。",
      "在 G 维求均值并减去，得到该回答相对于同题候选的优势。",
      "明确标准差约定；本教学例用总体标准差，与 correction=0 一致。",
      "按标准差加 ε 缩放，再广播到本回答的有效 token。",
      "零方差没有相对优劣信号；G=1 并不能提供有意义的组内比较。"
    ],
    "handwrite": "给 rewards=[1,3,5] 手算 A；实现 [B,G] 张量版本并处理零方差。",
    "followup": "跨 prompt 混合算均值会出什么问题？",
    "link": "grpo",
    "source": "https://arxiv.org/abs/2402.03300",
    "principle": "同一个问题生成多个回答，用组内相对好坏代替单独训练的价值网络。先约定标准差定义，再实现标准化。",
    "math": [
      "\\mu=\\frac1G\\sum_{i=1}^{G}R_i,\\quad\\sigma_{\\mathrm{pop}}=\\sqrt{\\frac1G\\sum_i(R_i-\\mu)^2}",
      "\\hat A_i=\\frac{R_i-\\mu}{\\sigma_{\\mathrm{pop}}+\\varepsilon}"
    ],
    "symbols": [
      {
        "symbol": "G",
        "meaning": "每个 prompt 的采样回答数"
      },
      {
        "symbol": "\\sigma_{\\mathrm{pop}}",
        "meaning": "本教学示例使用总体标准差，对应 correction=0"
      }
    ],
    "conditions": [
      "论文写 std，具体实现可用总体或样本标准差；二者不能在手算与代码间混用。",
      "G=1 或全相同 reward 时，总体标准差为 0、优势为 0；加 ε 不能修复默认样本标准差已经产生的 NaN。",
      "包含当前样本的组内均值不是严格动作无关 baseline，因此不要照搬 REINFORCE baseline 的无偏证明。"
    ],
    "example": "reward=(1,3,5)：μ=3，σpop=√(8/3)≈1.633，A≈(−1.225,0,1.225)。若用样本标准差则 σ=2，A=(−1,0,1)。",
    "checks": [
      "只在同一 prompt 的 G 维做统计。",
      "明确设置 std(correction=0)。",
      "G=1 与常数组输入均返回有限的零优势。"
    ]
  },
  {
    "id": "grpo-loss",
    "group": "GRPO",
    "title": "GRPO 的策略目标与 KL",
    "kind": "推导",
    "level": "必考",
    "question": "在组内优势之外，写出 clipped ratio 与参考策略 KL 惩罚；区分 old、new、reference。",
    "hint": "ratio 使用 old policy；KL 使用 reference policy。",
    "formula": "ρᵢ,ₜ=πθ(yᵢ,ₜ|sᵢ,ₜ)/πold(yᵢ,ₜ|sᵢ,ₜ)\nJ=E[min(ρAᵢ,clip(ρ,1−ε,1+ε)Aᵢ)−β DKL(πθ∥πref)]",
    "derivation": [
      "旧策略为每个 prompt 采样 G 条 response，缓存 old log_prob。",
      "对每个有效 token 计算 ratio，与组内优势形成 clipped surrogate。",
      "定义 u=logπref−logπθ。因为 exp(u)−u−1≥0，所以每个样本的 k 非负。",
      "在固定状态且动作来自当前策略时，Eπθ[exp(u)]=Σπref=1，于是 E[k]=Eπθ[logπθ−logπref]。",
      "先按 Ti 平均再按 G 平均得到原论文目标。复用旧样本与全 batch token 平均都是需要说明的实现选择。"
    ],
    "handwrite": "手写 [B·G,Tr] ratio、mask、sequence advantage 广播及 f-divergence penalty。",
    "followup": "GRPO 相比 PPO 省掉了什么内存？又增加了哪项采样成本？",
    "link": "grpo",
    "source": "https://arxiv.org/abs/2402.03300",
    "principle": "GRPO 把回答级相对优势广播到 token，再沿每条回答平均。参考 KL 是独立的正则项；old 策略用于概率比，reference 用于限制偏离。",
    "math": [
      "\\rho_{i,t}=\\frac{\\pi_\\theta(y_{i,t}\\mid s_{i,t})}{\\pi_{\\mathrm{old}}(y_{i,t}\\mid s_{i,t})}",
      "J=\\mathbb E\\left[\\frac1G\\sum_i\\frac1{T_i}\\sum_t\\{f(\\rho_{i,t},\\hat A_i)-\\beta k_{i,t}\\}\\right]",
      "u=\\log\\pi_{\\mathrm{ref}}(a\\mid s)-\\log\\pi_\\theta(a\\mid s),\\quad k=e^u-u-1",
      "\\mathbb E_{a\\sim\\pi_\\theta}[k]=D_{\\mathrm{KL}}(\\pi_\\theta\\Vert\\pi_{\\mathrm{ref}})"
    ],
    "symbols": [
      {
        "symbol": "f",
        "meaning": "PPO 一侧裁剪目标"
      },
      {
        "symbol": "T_i",
        "meaning": "第 i 条回答的有效长度"
      },
      {
        "symbol": "k",
        "meaning": "非负的 sampled KL 估计值"
      }
    ],
    "conditions": [
      "上面的 KL 期望恒等式针对固定状态、a∼当前策略且两分布支撑兼容。",
      "多轮更新继续复用 old 样本时，未经重要性校正的 k 均值不再严格无偏估计当前策略 KL；更不等于自动微分获得无偏 KL 梯度。",
      "论文先在每条回答内平均，再在回答间平均；仓库按全 batch 有效 token 平均，会给予长回答更大权重。"
    ],
    "example": "πθ=(0.5,0.5)、πref=(0.9,0.1)：两个 k≈(0.2122,0.8094)，当前策略加权均值≈0.5108，等于 KL(πθ∥πref)。",
    "checks": [
      "KL 方向是 current∥reference。",
      "用 expm1(u)−u 可改善 u≈0 时的抵消误差。",
      "mask 后明确采用 sequence mean 还是 token mean。"
    ],
    "stepMath": [
      null,
      null,
      "e^u\\ge1+u\\ \\Longrightarrow\\ k\\ge0",
      "\\mathbb E_{\\pi_\\theta}[e^u-u-1]=1-\\mathbb E_{\\pi_\\theta}[u]-1=D_{\\mathrm{KL}}(\\pi_\\theta\\Vert\\pi_{\\mathrm{ref}})",
      null
    ]
  },
  {
    "id": "compare",
    "group": "对齐比较",
    "title": "PPO、DPO、GRPO、OPD 一分钟比较",
    "kind": "口述",
    "level": "必考",
    "question": "从数据来源、是否在线采样、监督信号、value/reward/teacher 需求四个维度比较。",
    "hint": "先说谁生成数据，再说每个样本如何给梯度。",
    "formula": "PPO：在线 rollout → RM/KL → value/GAE → clipped update\nDPO：离线偏好对 → 隐式 reward 差 → logistic loss\nGRPO：在线同题多采样 → 组内 reward → clipped update\nOPD：学生在线采样 → 教师逐 token 分布 → 蒸馏更新",
    "derivation": [
      "PPO/GRPO 的策略目标都常需旧策略概率比；DPO 直接学习偏好对，无需在线 rollout。",
      "PPO 用 value model 估计优势；GRPO 用同 prompt 组内奖励作相对优势。",
      "OPD 在学生访问的前缀上取得教师信号，常用分布级监督而不是仅有回答级 reward。"
    ],
    "handwrite": "闭卷画一个四列对照表；用一句话说明各方法最容易出错的张量位置。",
    "followup": "何时静态 DPO 数据可能不覆盖当前策略常访问的状态？",
    "source": "https://arxiv.org/abs/2306.13649",
    "principle": "先辨认数据由谁生成、监督信号来自哪里，再讨论损失。算法名字相似，不代表模型需求与梯度路径相同。",
    "math": [
      "\\mathrm{PPO}:\\quad R,\\ V,\\ \\hat A_t,\\ \\rho_t",
      "\\mathrm{DPO}:\\quad (y^+,y^-),\\ -\\log\\sigma(\\beta\\Delta)",
      "\\mathrm{GRPO}:\\quad (R_1,\\ldots,R_G),\\ \\hat A_i,\\ \\rho_{i,t}",
      "\\mathrm{OPD}:\\quad y\\sim\\pi_S,\\ D(\\pi_T(\\cdot\\mid s_t),\\pi_S(\\cdot\\mid s_t))"
    ],
    "symbols": [
      {
        "symbol": "R,V",
        "meaning": "奖励与价值估计"
      },
      {
        "symbol": "\\pi_T,\\pi_S",
        "meaning": "教师与学生"
      }
    ],
    "conditions": [
      "此处讨论标准离线 DPO、在线 PPO/GRPO，以及学生轨迹上的 token 级 OPD。",
      "GRPO 的奖励可来自规则或模型，不要求一定训练一个 RM。"
    ],
    "example": "数学判题：GRPO 可以用答案是否正确的规则奖励；偏好对数据可供 DPO；有教师 token 分布时可做 OPD。",
    "checks": [
      "分别列出采样器、reward、value、reference、teacher 的作用。",
      "能口述训练信号如何传给 policy logits。"
    ]
  },
  {
    "id": "opd-objective",
    "group": "OPD",
    "title": "OPD：学生轨迹上的教师监督",
    "kind": "推导",
    "level": "必考",
    "question": "OPD 此处指 On-Policy Distillation。写出 student rollout、teacher 条件分布、token-level KL 目标。",
    "hint": "前缀 sₜ=(x,y_<t) 由学生生成，教师只在这些前缀上给分布。",
    "formula": "y∼πS(·|x), sₜ=(x,y_<t)\nLforward=E_{y∼πS} Σₜ KL(πT(·|sₜ) ∥ πS(·|sₜ))",
    "derivation": [
      "本轮先由学生策略采样 response；每个动作前的状态是 prompt 加已生成 token。",
      "教师在这些学生状态上提供整个词表的下一 token 分布，不要求教师生成完整答案。",
      "固定前缀后计算 forward KL；教师熵与学生 θ 无关，学生梯度等价于教师软标签的交叉熵梯度。",
      "GKD 风格有意不反传采样分布。若定义随 θ 变化的完整期望，除局部 KL 导数外还会出现轨迹概率的 score-function 项。",
      "选择 reverse KL 会改变局部损失和梯度，但学生轨迹采样这一 on-policy 特征仍然成立。"
    ],
    "handwrite": "用两个 mock logits 张量 [B,T,V] 写 masked KL，teacher logits stop-gradient。",
    "followup": "与教师生成完整答案再做 SFT 相比，解决了什么状态分布问题？",
    "source": "https://arxiv.org/abs/2306.13649",
    "link": "opd",
    "principle": "让学生在自己可能出错的前缀上接受教师纠正，减轻只学习教师轨迹带来的训练与推理状态分布差异。OPD 表示数据采集方式，并不指定唯一的 KL 方向。",
    "math": [
      "y\\sim\\pi_{S,\\mathrm{old}}(\\cdot\\mid x),\\quad s_t=(x,y_{<t})",
      "\\mathcal L_F=\\mathbb E_{x,y}\\left[\\frac1{T_y}\\sum_tD_{\\mathrm{KL}}(\\pi_T(\\cdot\\mid s_t)\\Vert\\pi_{S,\\theta}(\\cdot\\mid s_t))\\right]",
      "\\nabla_{z_S}\\operatorname{CE}(\\pi_T,\\pi_S)=\\pi_S-\\pi_T"
    ],
    "symbols": [
      {
        "symbol": "\\pi_{S,\\mathrm{old}}",
        "meaning": "本轮采样时的学生策略"
      },
      {
        "symbol": "\\pi_T",
        "meaning": "冻结教师"
      },
      {
        "symbol": "z_S",
        "meaning": "学生 logits"
      }
    ],
    "conditions": [
      "此处定义 GKD 风格：固定采样前缀，不对采样分布反传。",
      "教师与学生必须有可直接对齐的 token 事件空间。",
      "题目展示前向 KL，仓库 train_step 用反向 KL；两者都可在学生轨迹上学习。",
      "本仓库 mask 是全 batch token 平均；变长回答与逐序列平均的权重不同。"
    ],
    "example": "固定前缀上 teacher=(0.8,0.2)、student=(0.5,0.5)，前向 KL≈0.1927；对 student logits 的梯度是 (−0.3,0.3)。",
    "checks": [
      "采样使用 no_grad，教师打分使用 no_grad。",
      "第一个 response token 对齐 prompt 最后一列 logits。",
      "loss 只更新 student，teacher.grad 始终为空。"
    ],
    "stepMath": [
      null,
      null,
      "D_{\\mathrm{KL}}(T\\Vert S)=H(T,S)-H(T)",
      "\\nabla_\\theta\\mathbb E_{y\\sim\\pi_\\theta}[\\ell_\\theta(y)]=\\mathbb E[\\nabla_\\theta\\ell_\\theta(y)+\\ell_\\theta(y)\\nabla_\\theta\\log\\pi_\\theta(y)]",
      null
    ]
  },
  {
    "id": "opd-reverse",
    "group": "OPD",
    "title": "OPD：反向 KL 与 sampled-token 信号",
    "kind": "推导",
    "level": "高频",
    "question": "给定学生前缀，写出反向 KL 的精确形式与采样 token 的估计形式，说明两者差别。",
    "hint": "在固定状态 s 下，KL(πS∥πT)=E_{a∼πS}[logπS(a|s)−logπT(a|s)]。",
    "formula": "KL(πS∥πT | s)=ΣₐπS(a|s)[logπS(a|s)−logπT(a|s)]\nĝ(s,a)=logπS(a|s)−logπT(a|s), a∼πS",
    "derivation": [
      "在固定前缀上按学生分布展开 reverse KL。",
      "从学生分布采一个动作，其 log-ratio 的期望就是 KL 数值。",
      "对精确求和求导时，学生概率权重与 logπS 都依赖 θ，因此出现 (log-ratio+1) 的 score-function 系数。",
      "朴素的 sampled log-ratio.backward() 只微分选中的 logπS，漏掉概率权重导数；这一般不是正确的 KL 梯度。",
      "小词表练习直接对完整分布求和并反传，便于检查方向、mask 与梯度。"
    ],
    "handwrite": "写 exact_vocab_reverse_kl(student_logits, teacher_logits, mask)，再实现采样 log-ratio 估计并比较。",
    "followup": "为什么 token log-ratio 可以为负，而真实 KL 不能？",
    "source": "https://arxiv.org/abs/2306.08543",
    "link": "opd",
    "principle": "反向 KL 在固定前缀上用学生概率加权。它的数值可以用 sampled log-ratio 估计，但估计值的朴素反传不等于真实 KL 的梯度。",
    "math": [
      "K(s)=\\sum_a\\pi_S(a\\mid s)\\log\\frac{\\pi_S(a\\mid s)}{\\pi_T(a\\mid s)}",
      "\\hat K=\\log\\pi_S(a\\mid s)-\\log\\pi_T(a\\mid s),\\quad a\\sim\\pi_S",
      "\\nabla K=\\mathbb E_{a\\sim\\pi_S}\\left[(\\log(\\pi_S/\\pi_T)+1)\\nabla\\log\\pi_S\\right]"
    ],
    "symbols": [
      {
        "symbol": "K(s)",
        "meaning": "固定状态上的精确 reverse KL"
      },
      {
        "symbol": "\\hat K",
        "meaning": "只采一个 token 的数值估计"
      }
    ],
    "conditions": [
      "教师固定、支撑覆盖学生；否则 KL 可以无穷大。",
      "公式中的 +1 项期望为零，可作为常数 baseline 去掉，但仍要考虑采样分布的梯度。",
      "仓库用完整词表 KL 自动微分，只固定前缀分布，不固定学生词表概率权重。"
    ],
    "example": "student=(0.5,0.5)、teacher=(0.8,0.2)：sampled log-ratio≈(−0.4700,0.9163)，均值≈0.2231。固定采样动作后直接反传 logπS，其期望梯度为 0，但真实 KL 梯度通常不为 0。",
    "checks": [
      "全词表 KL 数值非负，单 token log-ratio 可以为负。",
      "student.softmax 权重保留梯度。",
      "teacher logits detach，mask 分母仅计有效位置。"
    ]
  },
  {
    "id": "opd-fail",
    "group": "OPD",
    "title": "OPD 的覆盖、兼容性与失败模式",
    "kind": "口述",
    "level": "进阶",
    "question": "为何“教师分数更高”也不一定蒸馏成功？哪些轨迹/词表条件会使监督变弱？",
    "hint": "看学生能到达的状态、教师在这些状态的分布，以及 tokenizer 是否兼容。",
    "formula": "学习信号发生在 sₜ∼学生诱导的状态分布 dπS，而非教师状态分布 dπT。",
    "derivation": [
      "学生采不到的状态不会直接得到 on-policy 信号；教师强在其他轨迹上的能力未必能传过来。",
      "教师和学生推理模式差异太大时，在学生前缀上的教师反馈可能与目标轨迹不一致。",
      "全词表蒸馏通常需要可对齐的 token 空间；不同 tokenizer 不能直接逐 token 比 KL。",
      "可以用冷启动、提示选择、受限词表或序列级反馈缓解，具体选择取决于实现与资源。"
    ],
    "handwrite": "构造两步 toy 采样：一版用硬过滤令路径概率为 0，一版用小概率，比较收集到的教师监督状态。",
    "followup": "OPD 与在线 RL 的密集信号、采样成本、教师调用成本分别如何比较？",
    "source": "https://arxiv.org/abs/2306.13649",
    "principle": "在线监督只发生在学生访问到的前缀上。覆盖不足、token 事件空间不一致或教师对学生前缀不可靠，都会限制蒸馏效果。",
    "math": [
      "s_t\\sim d_{\\pi_S},\\quad\\mathcal L=\\mathbb E_{s_t\\sim d_{\\pi_S}}D(\\pi_T(\\cdot\\mid s_t),\\pi_S(\\cdot\\mid s_t))"
    ],
    "symbols": [
      {
        "symbol": "d_{\\pi_S}",
        "meaning": "学生诱导的状态访问分布"
      }
    ],
    "conditions": [
      "状态访问少意味着监督少；“永远到不了”只适用于严格零支撑或硬采样约束。",
      "未过滤的有限 softmax logits 通常处处有正概率，罕见路径是有限采样覆盖问题。"
    ],
    "example": "如果 top-k 永久过滤掉通向某状态的 token，该状态不会出现在 rollout；若只是概率极小，则仍可能访问，但短时间内样本很少。",
    "checks": [
      "区分有限采样缺失与真正零支撑。",
      "不同 tokenizer 不能直接按 token 下标算 KL。"
    ]
  },
  {
    "id": "attention",
    "group": "Transformer",
    "title": "Scaled Dot-Product Attention 手撕",
    "kind": "手撕",
    "level": "必考",
    "question": "给 Q,K,V 的形状 [B,H,T,Dh]，写出 causal attention 全流程。",
    "hint": "scores 的最后两个维度是 [Tq,Tk]，softmax 沿 key 维。",
    "formula": "Attention(Q,K,V)=softmax(QKᵀ/√Dh + M)V",
    "derivation": [
      "QKᵀ 得 [B,H,Tq,Tk]；除以 √Dh 避免维度增大时点积方差增大。",
      "在 softmax 前把不可见位置设为 −∞；沿 Tk 做稳定 softmax。",
      "权重乘 V 得 [B,H,Tq,Dh]；转置并 concat 多头后投影。"
    ],
    "handwrite": "手写 safe_softmax、causal mask、多头 reshape；验证未来 token 变化不影响过去输出。",
    "followup": "prefill 带缓存时 mask 如何偏移？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/1706.03762",
    "principle": "点积衡量 query 与 key 的匹配，softmax 把分数变成权重，再对 value 求加权和。因果 mask 决定每个位置能看见哪些 key。",
    "math": [
      "S=\\frac{QK^\\top}{\\sqrt{d_h}}+M,\\quad P=\\operatorname{softmax}_{\\mathrm{key}}(S),\\quad O=PV",
      "\\operatorname{Var}\\left(\\sum_{j=1}^{d_h}q_jk_j\\right)=d_h\\quad\\Longrightarrow\\quad\\operatorname{Var}\\left(\\frac{q^\\top k}{\\sqrt{d_h}}\\right)=1"
    ],
    "symbols": [
      {
        "symbol": "Q,K,V",
        "meaning": "形状 [B,H,T,Dh]"
      },
      {
        "symbol": "M",
        "meaning": "允许位置为 0，禁止位置为 −∞"
      }
    ],
    "conditions": [
      "方差推导假设各分量独立、零均值、单位方差；真实训练分布并不完全满足。",
      "全遮蔽行不能直接做 softmax(-∞,…,-∞)；应避免或明确定义零输出。"
    ],
    "example": "q=k=(1,1)，dh=2：未缩放点积为 2，缩放后为 √2。单个允许 key 的 attention 权重为 1。",
    "checks": [
      "softmax 沿最后的 key 维。",
      "未来 token 改动不影响过去输出。",
      "cache 场景的 causal mask 使用绝对位置。"
    ]
  },
  {
    "id": "rope",
    "group": "Transformer",
    "title": "RoPE 为什么编码相对位置",
    "kind": "推导",
    "level": "高频",
    "question": "写 2D 旋转矩阵，证明同一频率下 R(m)q 与 R(n)k 的内积只依赖 n−m。",
    "hint": "旋转矩阵正交且 R(m)ᵀR(n)=R(n−m)。",
    "formula": "R(m)=[[cos(mω),−sin(mω)],[sin(mω),cos(mω)]]\n(R(m)q)ᵀ(R(n)k)=qᵀR(n−m)k",
    "derivation": [
      "把 q、k 每两个维度组成一对，各自按位置角度旋转。",
      "利用 R(m)ᵀ=R(−m) 以及旋转矩阵合成关系，内积出现位置差。",
      "Q/K 旋转而 V 不旋转；decode 时位置索引必须接着历史 KV 长度。"
    ],
    "handwrite": "实现 rotate_half、cos/sin 缓存，并比较相同相对距离的点积。",
    "followup": "模型超出预计算最大位置时如何处理？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/2104.09864",
    "principle": "对 Q/K 用位置相关的正交旋转后，内积中的两个绝对位置合成相对位移。结论针对旋转部分，不意味着完整内容无关。",
    "math": [
      "R(m)=\\begin{pmatrix}\\cos(m\\omega)&-\\sin(m\\omega)\\\\\\sin(m\\omega)&\\cos(m\\omega)\\end{pmatrix}",
      "(R(m)q)^\\top R(n)k=q^\\top R(m)^\\top R(n)k=q^\\top R(n-m)k"
    ],
    "symbols": [
      {
        "symbol": "\\omega",
        "meaning": "某对维度的角频率"
      },
      {
        "symbol": "m,n",
        "meaning": "query/key 的绝对位置"
      }
    ],
    "conditions": [
      "对应维度使用相同频率和旋转约定；点积仍取决于 q、k 的内容。",
      "相邻维度配对与 split-half 配对是不同布局，不能混用 cos/sin 和 rotate_half。"
    ],
    "example": "固定 q=(1,0)、k=(1,0)、ω=0.1。位置 (2,5) 与 (8,11) 的旋转点积都是 cos(0.3)。",
    "checks": [
      "旋转保持向量范数。",
      "平移两者位置不改变固定 q/k 的旋转点积。",
      "decode 的位置偏移等于历史 cache 长度。"
    ]
  },
  {
    "id": "kvcache",
    "group": "Transformer",
    "title": "KV Cache 的形状、复杂度和因果性",
    "kind": "手撕",
    "level": "必考",
    "question": "为什么自回归 decode 缓存 K/V，而不用缓存每步的 Q？写出 prefill 与单 token decode 的形状。",
    "hint": "新查询只需和全部历史 key 做匹配。",
    "formula": "Kcache,Vcache:[B,H,Tpast,Dh]\nKnew,Vnew:[B,H,1,Dh] → concat(dim=2)",
    "derivation": [
      "历史 token 的 K/V 在固定权重与因果注意力下不变，缓存避免每步重算历史层。",
      "当前 Q 只用于当前 token 查询，之后可丢弃；历史 K/V 供未来每步使用。",
      "每步新 Q 对 Tpast+1 个 key 算注意力；总生成算量仍随长度增长，缓存的显存也按层数和上下文长度线性增长。"
    ],
    "handwrite": "比较整段 forward 与逐 token 携 KV cache 的最后位置 logits。",
    "followup": "GQA/MQA 为什么能降低 KV cache 内存？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/2305.13245",
    "principle": "因果模型的历史表示不受未来 token 影响。权重固定时，可重用历史 K/V，只为新 token 计算新的 Q/K/V。",
    "math": [
      "K_{\\mathrm{cache}},V_{\\mathrm{cache}}\\in\\mathbb R^{B\\times H\\times T_{\\mathrm{past}}\\times d_h}",
      "Q_{\\mathrm{new}}K_{\\mathrm{cache+new}}^\\top\\in\\mathbb R^{B\\times H\\times1\\times(T_{\\mathrm{past}}+1)}"
    ],
    "symbols": [
      {
        "symbol": "T_{\\mathrm{past}}",
        "meaning": "已缓存 token 数"
      },
      {
        "symbol": "d_h",
        "meaning": "每个 head 的维度"
      }
    ],
    "conditions": [
      "模型权重固定、eval 模式、position_ids 一致；缓存不适用于不加调整的双向注意力。",
      "普通完整注意力下，生成 N 个 token 的累计注意力算量仍为 O(N²)，不是整个生成 O(N)。"
    ],
    "example": "历史长度 8，新增 1 token：Q 为 [B,H,1,Dh]，更新后的 K/V 为 [B,H,9,Dh]，scores 为 [B,H,1,9]。",
    "checks": [
      "整段 forward 与逐 token cache forward 的 logits 接近。",
      "K/V 沿序列维拼接。",
      "batch/padding 与位置索引一致。"
    ]
  },
  {
    "id": "norm-ffn",
    "group": "Transformer",
    "title": "RMSNorm 与 SwiGLU",
    "kind": "手撕",
    "level": "高频",
    "question": "写出 RMSNorm 和 SwiGLU 的前向，并解释 Pre-Norm 残差顺序。",
    "hint": "RMSNorm 不减均值；SwiGLU 有 gate/up/down 三个投影。",
    "formula": "RMSNorm(x)=x·rsqrt(mean(x²)+ε)·w\nSwiGLU(x)=Wdown(SiLU(Wgate x) ⊙ Wup x)",
    "derivation": [
      "RMSNorm 用平方均值归一化最后一维，不做中心化；可学习向量做逐通道缩放。",
      "SwiGLU 的门控分支与值分支逐元素相乘，再投影回模型维度。",
      "Pre-Norm block：h=x+Attention(Norm(x))；out=h+FFN(Norm(h))。"
    ],
    "handwrite": "实现 RMSNorm 和 SwiGLU，对比 PyTorch 参考运算的 shape 与梯度。",
    "followup": "为什么不要把参数 weight 初始化为全零？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/1910.07467",
    "principle": "RMSNorm 用均方根控制尺度，不减均值；SwiGLU 用一条非线性门控制另一条投影的值。Pre-Norm 在两个子层前分别归一化。",
    "math": [
      "\\operatorname{RMSNorm}(x)=\\frac{x}{\\sqrt{\\frac1d\\sum_jx_j^2+\\varepsilon}}\\odot w",
      "\\operatorname{SwiGLU}(x)=W_d[\\operatorname{SiLU}(W_gx)\\odot W_ux]",
      "h=x+\\operatorname{Attn}(\\operatorname{Norm}(x)),\\quad z=h+\\operatorname{FFN}(\\operatorname{Norm}(h))"
    ],
    "symbols": [
      {
        "symbol": "w",
        "meaning": "每通道可训练缩放"
      },
      {
        "symbol": "W_g,W_u,W_d",
        "meaning": "gate、up、down 投影"
      }
    ],
    "conditions": [
      "RMSNorm 沿隐藏维；epsilon 放在平方均值内。",
      "省略偏置的 Llama 风格 FFN；其他架构可以选择不同偏置约定。"
    ],
    "example": "x=(3,4)，ε=0、w=(1,1)：均方根=√12.5，输出约 (0.8485,1.1314)，均值并不为 0。",
    "checks": [
      "最后一维 shape 不变。",
      "w 通常初始化为全 1。",
      "半精度平方和可以在 float32 中归约。"
    ]
  },
  {
    "id": "mask",
    "group": "工程细节",
    "title": "四种 mask 不要混淆",
    "kind": "口述",
    "level": "必考",
    "question": "分别说清 causal mask、attention mask、loss mask、action mask 的位置和语义。",
    "hint": "一个决定能看谁，一个决定哪些输入有效，两个决定哪些目标计入更新。",
    "formula": "causal: scores[...,t,k>t]=−∞\nattention: pad 不参与注意力\nloss/action: 无效目标 token 不计入目标函数",
    "derivation": [
      "causal mask 避免看见未来 token。",
      "attention mask 标识输入 pad；具体如何作用于 Q/K 由模型实现决定。",
      "loss mask 用于监督损失，如 DPO prompt/pad 不算 response log_prob；action mask 用于 PPO/GRPO response 有效 token。",
      "mask 往往要与 shift 后的 log_prob [B,T−1] 对齐，最常见 bug 是错一个 token。"
    ],
    "handwrite": "画一条 [BOS,p1,p2,r1,r2,EOS,PAD] 的四类 mask。",
    "followup": "left padding 时 position_ids 与生成末位怎么处理？",
    "link": "env",
    "principle": "四类 mask 对应四种问题：能看谁、哪些输入有效、哪些标签参与监督、哪些动作参与策略优化。它们不能因 shape 相同就互换。",
    "math": [
      "M_{t,k}=\\begin{cases}0,&k\\le t\\text{ and key valid}\\\\-\\infty,&\\text{otherwise}\\end{cases}",
      "\\mathcal L=\\frac{\\sum_t m_t\\ell_t}{\\sum_t m_t}"
    ],
    "symbols": [
      {
        "symbol": "M",
        "meaning": "作用于 attention scores 的加性 mask"
      },
      {
        "symbol": "m_t",
        "meaning": "作用于目标 token 的 loss/action mask"
      }
    ],
    "conditions": [
      "输入 mask 与目标 mask 先错位再匹配 logits。",
      "EOS 通常计入动作或监督，EOS 之后才屏蔽。"
    ],
    "example": "[BOS,p1,p2,r1,r2,EOS,PAD] 的 response label mask=(0,0,0,1,1,1,0)；shift 后使用第 2 个到最后一个标签的位置。",
    "checks": [
      "prompt 仍可被 response 关注。",
      "EOS 计入损失，之后的 token 不计。",
      "padding 不污染统计分母。"
    ]
  },
  {
    "id": "lora",
    "group": "训练与推理",
    "title": "LoRA 参数量与缩放",
    "kind": "推导",
    "level": "高频",
    "question": "W∈R^{dout×din}，LoRA 低秩 r 的参数量是多少？前向如何合并？",
    "hint": "ΔW=BA，B∈R^{dout×r}, A∈R^{r×din}。",
    "formula": "y=Wx + (α/r)BAx\ntrainable=r(din+dout),  full=dout·din",
    "derivation": [
      "冻结原权重 W，只优化 A/B。",
      "低秩参数量是 r·din+r·dout；当 r 远小于两个维度时节省优化器状态与可训练参数。",
      "推理时可将 (α/r)BA 加回 W；若多个 adapter 动态切换则可分开计算。"
    ],
    "handwrite": "实现低秩线性层，验证 merge 前后输出一致。",
    "followup": "LoRA 减少了哪些内存，哪些激活/基础权重内存并不会自动消失？",
    "source": "https://arxiv.org/abs/2106.09685",
    "principle": "低秩增量把大矩阵更新限制在两个较小矩阵的乘积中。原权重被冻结，训练参数与优化器状态主要来自低秩分支。",
    "math": [
      "\\Delta W=\\frac\\alpha rBA,\\quad A\\in\\mathbb R^{r\\times d_{\\mathrm{in}}},\\ B\\in\\mathbb R^{d_{\\mathrm{out}}\\times r}",
      "y=Wx+\\Delta Wx,\\quad N_{\\mathrm{LoRA}}=r(d_{\\mathrm{in}}+d_{\\mathrm{out}})"
    ],
    "symbols": [
      {
        "symbol": "r",
        "meaning": "低秩维度"
      },
      {
        "symbol": "\\alpha/r",
        "meaning": "原始 LoRA 的缩放约定"
      }
    ],
    "conditions": [
      "只计算 A/B 参数，未计可选 bias 或其他可训练模块。",
      "merge 对 eval 模式且关闭 dropout 的同一 adapter 等价。"
    ],
    "example": "din=dout=4096、r=8：LoRA 65,536 参数，全量矩阵 16,777,216 参数，比例为 1/256。",
    "checks": [
      "基础 W 不接收梯度。",
      "关闭 dropout 后 merge 前后输出一致。",
      "常见初始化是一矩阵随机、另一矩阵为零，初始增量为零。"
    ]
  },
  {
    "id": "flash",
    "group": "训练与推理",
    "title": "FlashAttention 的 IO 核心",
    "kind": "口述",
    "level": "高频",
    "question": "为什么标准 attention 会产生 O(T²) 的显存中间量？在线 softmax 如何分块合并？",
    "hint": "先分块计算 max、exp sum 和加权 value 累积。",
    "formula": "m= max(m₁,m₂)\nℓ=e^(m₁−m)ℓ₁+e^(m₂−m)ℓ₂\no=e^(m₁−m)o₁+e^(m₂−m)o₂",
    "derivation": [
      "普通实现可能显式写出 T×T scores/probabilities，读写显存昂贵。",
      "分块流式维护每行最大值 m、指数和 ℓ、未归一化的加权和 o；新块到来时按新 max 重缩放旧统计。",
      "最后输出 o/ℓ，数学上等价稳定 softmax attention；实际实现仍要正确处理因果/padding mask。"
    ],
    "handwrite": "用小矩阵手算两块的 m/ℓ/o 合并，和一次性 softmax 比较。",
    "followup": "它降低的是 FLOPs、HBM IO，还是两者？",
    "source": "https://arxiv.org/abs/2205.14135",
    "principle": "用分块和在线 softmax 避免把整个注意力矩阵反复写入显存。合并时必须在共同最大值下重新缩放分母和未归一化分子。",
    "math": [
      "m=\\max(m_1,m_2)",
      "\\ell=e^{m_1-m}\\ell_1+e^{m_2-m}\\ell_2",
      "o=e^{m_1-m}o_1+e^{m_2-m}o_2,\\quad O=o/\\ell"
    ],
    "symbols": [
      {
        "symbol": "m_i",
        "meaning": "块内某一行的最大 score"
      },
      {
        "symbol": "\\ell_i",
        "meaning": "该行 Σ exp(score−mi)"
      },
      {
        "symbol": "o_i",
        "meaning": "该行 Σ exp(score−mi)·value，尚未归一化"
      }
    ],
    "conditions": [
      "o_i 是未归一化分子，不是已经除过 ℓ_i 的 attention 输出。",
      "空块/全 mask 块要显式处理，不能计算 −∞−(−∞)。"
    ],
    "example": "scores=(0,log2)、values=(1,3)：softmax 权重=(1/3,2/3)，输出=7/3。分成两块并重缩放后结果相同。",
    "checks": [
      "分块合并与完整 softmax 输出一致。",
      "普通 dense attention 仍是二次算术复杂度，主要减少 HBM IO 和中间存储。",
      "极大 scores 和 mask 均稳定。"
    ]
  },
  {
    "id": "gqa",
    "group": "训练与推理",
    "title": "MHA、MQA、GQA 的 KV 缓存",
    "kind": "推导",
    "level": "高频",
    "question": "给 nQ 个 query heads、nKV 个 KV heads，推导每层 KV cache 大小和 GQA 的内存收益。",
    "hint": "每 token 同时存 K 与 V。",
    "formula": "KV cache elements/layer = 2·B·T·nKV·Dh\nMHA: nKV=nQ; MQA: nKV=1; GQA: 1<nKV<nQ",
    "derivation": [
      "Q heads 可共享更少的 KV heads；每一 KV head 服务一组 Q heads。",
      "缓存与 nKV 成正比，GQA 把 MHA 的 KV 内存按 nKV/nQ 比例缩小（同 head_dim 前提）。",
      "这主要改善 decode 时缓存容量与带宽压力；注意质量、计算实现与分组方式的权衡。"
    ],
    "handwrite": "给 B=2,T=4096,nQ=32,nKV=8,Dh=128,bf16(2 bytes) 计算每层字节数。",
    "followup": "为何 MQA 的共享更强但可能牺牲表达能力？",
    "source": "https://arxiv.org/abs/2305.13245",
    "principle": "多个 query head 共享更少的 KV heads，缓存大小由 KV head 数决定。计算时要正确把每组 query 映射到相应 KV。",
    "math": [
      "N_{\\mathrm{KV/layer}}=2BTn_{\\mathrm{KV}}d_h",
      "\\mathrm{bytes}=2LBTn_{\\mathrm{KV}}d_h\\cdot\\mathrm{bytesPerElement}"
    ],
    "symbols": [
      {
        "symbol": "L",
        "meaning": "层数"
      },
      {
        "symbol": "n_{\\mathrm{KV}}",
        "meaning": "KV head 数"
      },
      {
        "symbol": "n_Q",
        "meaning": "query head 数"
      }
    ],
    "conditions": [
      "常规 GQA 要求 nQ 可被 nKV 整除；同一 head_dim 和 dtype 下比较。",
      "公式未包含缓存管理、分页和其他激活内存开销。"
    ],
    "example": "B=2、T=4096、nKV=8、Dh=128、bf16：每层 33,554,432 字节=32 MiB；nQ=32 的 MHA 为 128 MiB。",
    "checks": [
      "包括 K 和 V 两份缓存。",
      "区分 MB 与 MiB。",
      "按 nKV/nQ 比例比较 cache 内存。"
    ]
  },
  {
    "id": "bpe",
    "group": "训练与推理",
    "title": "BPE 合并与编码",
    "kind": "手撕",
    "level": "高频",
    "question": "从 UTF-8 字节序列训练一个最小 BPE，说明合并规则与 encode 的顺序。",
    "hint": "统计相邻 pair 频次，最高频 pair 合成新 token。",
    "formula": "vocab₀={0,…,255};  merge:(a,b)→new_id",
    "derivation": [
      "字节级初始词表保证任意 UTF-8 文本可表示。",
      "每轮统计当前 token 序列的相邻 pair，选最高频 pair 并替换出现位置；记录合并顺序。",
      "编码时按训练的合并规则/优先级应用，解码时把 token bytes 拼接再按 UTF-8 解码。"
    ],
    "handwrite": "用短语料训练 5 个 merge，验证 decode(encode(text))=text。",
    "followup": "多字节中文被切开时如何保证最终解码正确？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/1508.07909",
    "principle": "从可表示任意文本的初始符号出发，反复把频繁相邻符号合成新 token。编码要复用训练的合并优先级，而不是重新按待编码文本的频率学习。",
    "math": [
      "\\mathcal V_0=\\{0,\\ldots,255\\},\\quad (a,b)\\longrightarrow v_{\\mathrm{new}}"
    ],
    "symbols": [
      {
        "symbol": "\\mathcal V_0",
        "meaning": "字节级初始词表"
      }
    ],
    "conditions": [
      "这里实现教学用 byte-level BPE；实际 tokenizer 还含预切分、特殊 token 与合并边界规则。"
    ],
    "example": "序列 [a,b,a,b] 中 pair (a,b) 最频繁；合成 c 后得到 [c,c]，解码将 c 展开为 a,b。",
    "checks": [
      "encode 应按已学习 rank，而非新文本的 pair 频次。",
      "先拼全部 bytes 再 UTF-8 decode。"
    ]
  },
  {
    "id": "sampling",
    "group": "训练与推理",
    "title": "Temperature、Top-K、Top-P",
    "kind": "手撕",
    "level": "高频",
    "question": "给 logits，写出三种采样控制的顺序和数值稳定处理。",
    "hint": "先除温度，再裁候选集合，最后 softmax/multinomial。",
    "formula": "pᵢ=softmax(logitᵢ/T)\nTop-P: 保留按概率降序累积刚好达到 p 的最小前缀",
    "derivation": [
      "温度 T<1 使分布更尖，T>1 更平；T→0 需单独处理或安全下界。",
      "Top-K 只留概率最高的 K 个候选。",
      "Top-P 按降序累积概率，保留越过阈值的那个 token；不能把首个越界项也删掉。",
      "全部过滤后 softmax 无定义，应始终保留至少一个候选。"
    ],
    "handwrite": "手写批量 top-p mask，测试 p 很小、K>词表大小、重复 logits。",
    "followup": "为何 top-p 的集合大小会随上下文变化？",
    "link": "transformer",
    "source": "https://arxiv.org/abs/1904.09751",
    "principle": "温度改变候选概率的相对差距，Top-K/Top-P 改变候选集合。最终必须对保留候选重新归一化后再采样。",
    "math": [
      "p_i=\\frac{e^{z_i/\\tau}}{\\sum_j e^{z_j/\\tau}},\\quad\\tau>0",
      "j^*=\\min\\{j:\\sum_{i=1}^{j}p_{(i)}\\ge p_{\\mathrm{cut}}\\}"
    ],
    "symbols": [
      {
        "symbol": "\\tau",
        "meaning": "采样温度"
      },
      {
        "symbol": "p_{(i)}",
        "meaning": "按概率从大到小排序"
      },
      {
        "symbol": "p_{\\mathrm{cut}}",
        "meaning": "Top-P 阈值，0<p≤1"
      }
    ],
    "conditions": [
      "τ=0 使用显式 greedy 分支，不能直接除以 0。",
      "同时用 Top-K 和 Top-P 时，顺序及重新归一化方式决定最终分布，需要明确定义。"
    ],
    "example": "已排序概率=(0.6,0.25,0.15)，Top-P=0.7，保留前两个 token，再归一化为约 (0.7059,0.2941)。",
    "checks": [
      "保留第一个让累计概率达到阈值的 token。",
      "始终至少一个候选，K 限制在 [1,V]。",
      "重复 logits 的 tie 规则明确。"
    ]
  },
  {
    "id": "grad-acc",
    "group": "训练与推理",
    "title": "梯度累积与大 batch 等价条件",
    "kind": "推导",
    "level": "高频",
    "question": "将有效 batch 拆成 K 个 micro-batch，loss 应怎样缩放？什么时候不等价？",
    "hint": "若每个 micro-batch 大小相同，把各自 mean loss 除 K。",
    "formula": "Lbatch=(1/K)Σₖ Lmicro,k  （等大小、同一归约方式）",
    "derivation": [
      "每个 micro-batch backward 前把 mean loss 除 K；K 次后只做一次 optimizer.step()。",
      "大小或有效 token 数不同时，应按实际样本/有效 token 权重归约，而不是机械除 K。",
      "dropout 随机性、BatchNorm、梯度裁剪时机、优化器 step/scheduler 频率都可能破坏严格等价。"
    ],
    "handwrite": "比较单大 batch 与累积 K 次的小 batch 参数梯度。",
    "followup": "梯度裁剪应放在每个 micro-batch 还是累积完成之后？",
    "principle": "梯度对相加的损失是线性的。因此先累积各 micro-batch 的正确权重，再做一次优化器更新，可模拟同一损失定义下的大 batch。",
    "math": [
      "\\mathcal L=\\sum_{k=1}^{K}\\frac{n_k}{\\sum_jn_j}\\mathcal L_k,\\quad\\nabla\\mathcal L=\\sum_k\\frac{n_k}{\\sum_jn_j}\\nabla\\mathcal L_k"
    ],
    "symbols": [
      {
        "symbol": "n_k",
        "meaning": "样本均值时为样本数，token 均值时为有效 token 数"
      },
      {
        "symbol": "K",
        "meaning": "micro-batch 数"
      }
    ],
    "conditions": [
      "权重必须匹配最终目标的归约单位；样本均值不能随意改成 token 均值。",
      "严格梯度比较要关闭随机性和 batch 依赖层，并只在累积完成后 clip/step。"
    ],
    "example": "两个 micro-batch 有效 token 数为 2 和 6，token mean loss 分别为 1 和 3：全 batch loss=0.25×1+0.75×3=2.5，不是 2。",
    "checks": [
      "一个大 batch 与加权累积的梯度相近。",
      "最后一次 micro-batch 后只做一次 step。",
      "梯度裁剪与 scheduler 时机一致。"
    ]
  },
  {
    "id": "precision",
    "group": "工程细节",
    "title": "混合精度与数值稳定",
    "kind": "口述",
    "level": "高频",
    "question": "bf16、fp16 在指数范围上的差异，以及 softmax/logsumexp 的稳定写法。",
    "hint": "logsumexp 先减 max；fp16 更容易需要 loss scaling。",
    "formula": "logΣᵢexp(xᵢ)=m+logΣᵢexp(xᵢ−m), m=maxᵢxᵢ",
    "derivation": [
      "减去最大值避免 exp 上溢；在 log 概率路径上直接用 log_softmax 更稳定。",
      "bf16 指数位更宽、尾数较短；fp16 指数范围较窄，梯度下溢更值得警惕。",
      "参数、优化器状态与归约常需要更高精度，具体策略取决于硬件和训练框架。"
    ],
    "handwrite": "构造 [1000,1001] 与 [−1000,−1001]，比较朴素 softmax 和稳定版本。",
    "followup": "什么时候要在 float32 中做 softmax/reduction？",
    "principle": "指数变换放大极端数值。减去最大值不改变 softmax，却能避免正方向上溢；不同浮点格式还会影响范围与舍入精度。",
    "math": [
      "\\operatorname{LSE}(x)=m+\\log\\sum_i e^{x_i-m},\\quad m=\\max_i x_i",
      "\\log\\operatorname{softmax}(x)_i=x_i-\\operatorname{LSE}(x)"
    ],
    "symbols": [
      {
        "symbol": "m",
        "meaning": "当前归约维的最大值"
      }
    ],
    "conditions": [
      "至少一个有限 logit；全 −∞ 行不能仅靠减 max 修复。",
      "稳定实现不表示不会下溢，小概率仍可能在低精度下变成 0。"
    ],
    "example": "logits=(1000,1001) 的稳定 softmax 约为 (0.2689,0.7311)。bf16 有 8 位指数、7 位尾数；fp16 有 5 位指数、10 位尾数（均不含符号位）。",
    "checks": [
      "优先直接 log_softmax，而不是先 softmax 再 log。",
      "归约必要时转 float32。",
      "全 mask 行独立处理。"
    ]
  },
  {
    "id": "evaluation",
    "group": "工程细节",
    "title": "离线评估、数据泄漏与长度偏差",
    "kind": "口述",
    "level": "高频",
    "question": "如果你声称 DPO/GRPO 训练后效果变好，如何设计可信的比较？",
    "hint": "固定模型、数据划分、采样预算和答案判分规则。",
    "formula": "pass@k / accuracy / preference win-rate 等指标应注明采样温度、k、测试集与置信区间。",
    "derivation": [
      "确保训练题、验证题、测试题按题目或来源去重，避免同题改写泄漏。",
      "记录基础模型、训练 token/采样成本、reward model 或 judge 的偏差。",
      "长度不同会影响序列 log probability 的和与均值；评价 win-rate 时也可能有回答长度偏好。",
      "同时查看质量、格式、失败样本和成本，不能只报一个汇总分。"
    ],
    "handwrite": "设计一个 50 道题的小型盲测表，列出记录字段与对照组。",
    "followup": "为什么用同一个 reward model 训练和评估会高估进步？",
    "principle": "可信的评估来自受控比较：数据、预算、采样规则和判分保持一致，并报告有限样本的不确定性。",
    "math": [
      "\\hat p=\\frac cn,\\quad\\mathrm{SE}(\\hat p)\\approx\\sqrt{\\frac{\\hat p(1-\\hat p)}n}"
    ],
    "symbols": [
      {
        "symbol": "c,n",
        "meaning": "独立二项试验的成功数与总数"
      }
    ],
    "conditions": [
      "标准误近似要求独立、相同成功概率，且样本量足够；同题重复生成不能简单当作独立题目扩大 n。",
      "小样本或边界成功率更适合精确/Wilson 区间，成对模型比较应按题做配对统计。"
    ],
    "example": "50 道独立题答对 40 道，accuracy=0.8；二项标准误近似 √(0.8×0.2/50)≈0.0566。",
    "checks": [
      "题目层面去重与训练/测试隔离。",
      "同采样预算、同判分与模型版本。",
      "记录长度、失败类型和成本。"
    ]
  },
  {
    "id": "reward-bt",
    "group": "对齐比较",
    "title": "奖励模型与 Bradley–Terry 偏好概率",
    "kind": "推导",
    "level": "高频",
    "question": "给 chosen/rejected 成对偏好数据，如何训练标量 reward model？",
    "hint": "只看 reward 差，不需要绝对分数的零点。",
    "formula": "P(y⁺≻y⁻|x)=σ(rφ(x,y⁺)−rφ(x,y⁻))\nLRM=−E logσ(rφ(x,y⁺)−rφ(x,y⁻))",
    "derivation": [
      "Reward Model 对完整 prompt+response 输出标量分数。",
      "Bradley–Terry 假设偏好概率由两回答的 reward 差经 sigmoid 给出。",
      "对正确偏好取负对数似然；对两分数同时加同一个常数，loss 不变，因此绝对 reward 零点不可辨识。"
    ],
    "handwrite": "用两个标量张量写 reward loss；测试 chosen=rejected 时 loss=log 2。",
    "followup": "Reward Model 被策略持续优化时，为什么可能出现 reward hacking？",
    "source": "https://arxiv.org/abs/2203.02155",
    "principle": "两个回答的相对奖励决定偏好概率，因此只有奖励差可以被数据识别，绝对零点不影响偏好损失。",
    "math": [
      "P(y^+\\succ y^-\\mid x)=\\frac{e^{r^+}}{e^{r^+}+e^{r^-}}=\\sigma(r^+-r^-)",
      "\\mathcal L_{\\mathrm{RM}}=-\\mathbb E\\log\\sigma(r^+-r^-)"
    ],
    "symbols": [
      {
        "symbol": "r^+,r^-",
        "meaning": "标量奖励"
      }
    ],
    "conditions": [
      "Bradley–Terry 假设；奖励加同一个只依赖 prompt 的常数也不改变同题偏好。"
    ],
    "example": "r+=(1)、r−=(0)，偏好概率≈0.7311、loss≈0.3133；都加 10 后结果相同。",
    "checks": [
      "分数相同时 loss=log2。",
      "reward 差越大，chosen 的 loss 越小。",
      "稳定 logsigmoid，梯度正负正确。"
    ]
  },
  {
    "id": "sft-mask",
    "group": "训练与推理",
    "title": "SFT 的 response-only loss",
    "kind": "手撕",
    "level": "高频",
    "question": "指令微调时 prompt、response、padding 如何拼接并参与损失？",
    "hint": "模型看到 prompt，但可以只对 response token 计算 loss。",
    "formula": "input=[BOS,prompt,response,EOS,PAD]\nlabels=[−100 × prompt,response,EOS,−100 × pad]",
    "derivation": [
      "整个 prompt 作为上下文进入 causal LM，但在 response-only 训练设 label=-100。",
      "logits 和 labels 错位：位置 t 的 logits 预测 t+1 的 token。",
      "具体 chat template、system/user/assistant 边界要和推理保持一致，避免训练与推理格式漂移。"
    ],
    "handwrite": "给两条长短不同的对话，拼 batch 并生成 input_ids、attention_mask、labels。",
    "followup": "什么时候会选择训练全序列 loss？",
    "link": "env",
    "principle": "prompt 提供预测上下文，response 提供训练目标。response-only 训练只屏蔽目标损失，不屏蔽模型读取 prompt。",
    "math": [
      "\\mathcal L_{\\mathrm{SFT}}=-\\frac{\\sum_t m_t\\log\\pi_\\theta(y_t\\mid x,y_{<t})}{\\sum_t m_t}"
    ],
    "symbols": [
      {
        "symbol": "m_t",
        "meaning": "response 和 EOS 取 1，prompt/pad 取 0"
      }
    ],
    "conditions": [
      "labels 的位置与 input token 对齐，再统一 shift；BOS 应屏蔽。",
      "所有样本均无有效 response 时应跳过 batch。"
    ],
    "example": "输入 [BOS,p1,r1,EOS,PAD]，labels=[−100,−100,r1,EOS,−100]；shift 后只训练预测 r1 和 EOS 的两列。",
    "checks": [
      "第一个 response 由 prompt 最后一列 logits 预测。",
      "EOS 有监督，PAD 无监督。",
      "训练推理 chat template 一致。"
    ]
  },
  {
    "id": "adamw",
    "group": "训练与推理",
    "title": "AdamW 与权重衰减",
    "kind": "推导",
    "level": "高频",
    "question": "写出 Adam 的一阶/二阶矩更新，解释 decoupled weight decay 与把 λθ 加到梯度的区别。",
    "hint": "AdamW 的衰减直接作用在参数更新上。",
    "formula": "mₜ=β₁mₜ₋₁+(1−β₁)gₜ; vₜ=β₂vₜ₋₁+(1−β₂)gₜ²\nθₜ₊₁=θₜ−η m̂ₜ/(√v̂ₜ+ε)−ηλθₜ",
    "derivation": [
      "Adam 用梯度的指数滑动一阶矩和平方梯度二阶矩调整每个参数的步长。",
      "把 λθ 加进梯度会经过自适应缩放；AdamW 将 −ηλθ 从梯度更新中分离出来。",
      "实践中常跳过 bias 和 norm 参数的 weight decay；必须区分优化器权重衰减和 RLHF 里的 KL 惩罚。"
    ],
    "handwrite": "手算一维参数在前两步的 m/v、bias correction 和 AdamW 更新。",
    "followup": "为什么梯度累积改变 optimizer.step() 的频率会影响学习率调度？",
    "source": "https://arxiv.org/abs/1711.05101",
    "principle": "Adam 用历史梯度统计自适应调整步长；AdamW 把参数收缩从这些梯度统计中分离，避免把 L2 正则项一并进行自适应缩放。",
    "math": [
      "m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t,\\quad v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2",
      "\\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\quad\\hat v_t=\\frac{v_t}{1-\\beta_2^t}",
      "\\theta_t=(1-\\eta_t\\lambda)\\theta_{t-1}-\\eta_t\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\varepsilon}"
    ],
    "symbols": [
      {
        "symbol": "m_t,v_t",
        "meaning": "一阶/二阶矩"
      },
      {
        "symbol": "\\lambda",
        "meaning": "decoupled weight decay 系数"
      }
    ],
    "conditions": [
      "从 m0=v0=0、t=1 开始，明确 epsilon 的放置。",
      "假设该参数参与这次 optimizer step；不能把 grad=None 与有效零梯度混为一谈。"
    ],
    "example": "θ0=1、g1=2、η=0.1、λ=0.01，ε忽略：bias correction 后 m̂=2、v̂=4，θ1=0.999−0.1=0.899。",
    "checks": [
      "bias correction 幂指数从 1 开始。",
      "weight decay 不进入 m/v。",
      "和相同设置的 torch.optim.AdamW 比较一步结果。"
    ]
  },
  {
    "id": "speculative",
    "group": "训练与推理",
    "title": "推测解码为什么不改变目标分布",
    "kind": "推导",
    "level": "进阶",
    "question": "draft model 先提出 token，target model 并行验证；写出单 token 的接受率和拒绝后的修正分布。",
    "hint": "目标是保持 target 分布 p；draft 分布记 q。",
    "formula": "a(x)=min(1,p(x)/q(x)), x∼q\n拒绝后从 r(x)∝max(0,p(x)−q(x)) 采样",
    "derivation": [
      "draft 按 q 提议候选 x；target 给出 p(x)，按 min(1,p/q) 接受。",
      "被接受的候选质量为 min(p,q)；剩余目标质量为 p−min(p,q)=max(0,p−q)。",
      "拒绝时从剩余质量归一化采样，合起来恰好是目标分布 p。",
      "加速来自 target 一次验证多个 draft token；接受率低时收益下降。"
    ],
    "handwrite": "给 p=(0.7,0.3)、q=(0.4,0.6)，手算接受与残差分布。",
    "followup": "为什么不能拒绝后直接从 p 重新采样？",
    "source": "https://arxiv.org/abs/2211.17192",
    "principle": "draft 提议分布 q 不等于 target 分布 p。接受概率保留双方重叠的概率质量，再用残差分布补足 target 的其余质量。",
    "math": [
      "a(v)=\\min\\left(1,\\frac{p(v)}{q(v)}\\right),\\quad v\\sim q",
      "Z=\\sum_v[p(v)-q(v)]_+,\\quad r(v)=\\frac{[p(v)-q(v)]_+}{Z}",
      "q(v)a(v)+Zr(v)=\\min(p(v),q(v))+[p(v)-q(v)]_+=p(v)"
    ],
    "symbols": [
      {
        "symbol": "p,q",
        "meaning": "同一前缀上的 target/draft 有效采样分布"
      },
      {
        "symbol": "[u]_+",
        "meaning": "max(u,0)"
      }
    ],
    "conditions": [
      "若用温度/Top-P，p/q 必须取实际处理后的采样分布，不可混用原始 softmax。",
      "q(v)=0 的候选不会从 draft 被采出；p=q 时 Z=0 且永不进入拒绝分支。"
    ],
    "example": "p=(0.7,0.3)、q=(0.4,0.6)：接受率为 (1,0.5)，总接受概率 0.7；拒绝残差 r=(1,0)，补回 0.3，最终正好是 p。",
    "checks": [
      "按解析概率混合后精确重建 p。",
      "Z=0 时不做除法。",
      "接受的每一步在相应前缀下更新条件分布。"
    ]
  },
  {
    "id": "passk",
    "group": "工程细节",
    "title": "pass@k 与采样预算",
    "kind": "推导",
    "level": "高频",
    "question": "同一题生成 n 个样本，其中 c 个正确；无放回选 k 个，pass@k 的估计式是什么？",
    "hint": "先求 k 个全错的概率。",
    "formula": "pass@k = 1 − C(n−c,k)/C(n,k), n≥k",
    "derivation": [
      "n 个样本中有 n−c 个错误答案。",
      "无放回抽 k 个全错的组合数是 C(n−c,k)，总组合数是 C(n,k)。",
      "至少一个正确的概率等于 1 减全错概率；比较模型时 n、k、采样温度与预算应一致。"
    ],
    "handwrite": "给 n=10,c=2,k=1/2/5 手算 pass@k。",
    "followup": "为什么只比较 pass@1 可能掩盖模型在多样采样下的能力？",
    "source": "https://arxiv.org/abs/2107.03374",
    "principle": "从 n 个已生成候选中无放回选 k 个，至少一个正确的概率可用“1−全部错误”计算。作为模型指标估计量还需要说明生成样本的随机假设。",
    "math": [
      "\\widehat{\\mathrm{pass@}k}=1-\\frac{\\binom{n-c}{k}}{\\binom nk},\\quad1\\le k\\le n"
    ],
    "symbols": [
      {
        "symbol": "n,c,k",
        "meaning": "生成数、正确数、抽取数"
      }
    ],
    "conditions": [
      "n−c<k 时错误组合数为 0，因此估计值为 1。",
      "在独立同分布候选假设下，可作为 1−(1−p)^k 的无偏估计；beam search 或相关样本不能自动沿用该结论。"
    ],
    "example": "n=10、c=2：pass@1=0.2，pass@2=1−28/45≈0.3778，pass@5=1−56/252≈0.7778。",
    "checks": [
      "c=0 得 0，c=n 得 1。",
      "边界 n−c<k 得 1。",
      "比较时保持题集、n、k 和温度一致。"
    ]
  }
];
