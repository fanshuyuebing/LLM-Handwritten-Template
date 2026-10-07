"""OPD / On-Policy Distillation 手撕练习 (EASY).

学生生成自己的续写；冻结教师在学生到达的前缀上输出 token 分布；
固定这些采样前缀，最小化逐位置的 KL。本文件用共享词表的微型 Mock 模型，
不下载模型权重，也不代表真实训练质量。
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OPDConfig:
    vocab_size: int = 32
    hidden_size: int = 48
    max_new_tokens: int = 6
    lr: float = 1e-3


class TinyPolicy(nn.Module):
    """教学用自回归策略；Embedding + GRU + LM Head。"""
    def __init__(self, config: OPDConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.gru(self.embed(input_ids))
        return self.lm_head(hidden)  # [B, T, V]


# TODO-1: On-policy rollout。只返回生成的 response token；本练习固定长度，无 EOS。
@torch.no_grad()
def rollout_student(student: TinyPolicy, prompt_ids: torch.Tensor,
                    max_new_tokens: int) -> torch.Tensor:
    """输入 [B,Tp]，返回学生采样的 [B,Tr]；采样不反传梯度。"""
    # 答案：full_ids = prompt_ids
    #       generated = []
    #       for _ in range(max_new_tokens):
    #           next_logits = student(full_ids)[:, -1, :]
    #           next_id = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
    #           generated.append(next_id)
    #           full_ids = torch.cat([full_ids, next_id], dim=1)
    #       return torch.cat(generated, dim=1)
    raise NotImplementedError("TODO-1: 实现学生自采样 rollout")


# TODO-2: 同一批学生前缀上的教师/学生分布 KL。
def masked_token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    response_mask: torch.Tensor, direction: str = "reverse") -> torch.Tensor:
    """logits:[B,Tr,V], mask:[B,Tr]；返回有效位置的平均 KL。"""
    assert direction in ("forward", "reverse")
    # 答案：s_logp = F.log_softmax(student_logits.float(), dim=-1)
    #       t_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    #       if direction == "reverse":
    #           per_token = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
    #       else:
    #           per_token = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
    #       return (per_token * response_mask).sum() / response_mask.sum().clamp_min(1)
    raise NotImplementedError("TODO-2: 实现 masked forward/reverse KL")


# TODO-3: 固定学生采样轨迹，对有效 response 位置做蒸馏更新。
def train_step(student: TinyPolicy, teacher: TinyPolicy,
               prompt_ids: torch.Tensor, optimizer: torch.optim.Optimizer,
               max_new_tokens: int) -> torch.Tensor:
    """返回本次 KL loss（已脱离计算图）。"""
    # 答案：response = rollout_student(student, prompt_ids, max_new_tokens)
    #       full_ids = torch.cat([prompt_ids, response], dim=1)
    #       first = prompt_ids.shape[1] - 1
    #       with torch.no_grad():
    #           teacher_logits = teacher(full_ids)[:, first:-1, :]
    #       student_logits = student(full_ids)[:, first:-1, :]
    #       mask = torch.ones_like(response, dtype=torch.float32)
    #       loss = masked_token_kl(student_logits, teacher_logits, mask, "reverse")
    #       optimizer.zero_grad()
    #       loss.backward()
    #       optimizer.step()
    #       return loss.detach()
    raise NotImplementedError("TODO-3: 实现 OPD 单步更新")


def main() -> None:
    torch.manual_seed(7)
    config = OPDConfig()
    student, teacher = TinyPolicy(config), TinyPolicy(config)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    prompts = torch.randint(0, config.vocab_size, (2, 4))
    loss = train_step(student, teacher, prompts, optimizer, config.max_new_tokens)
    assert torch.isfinite(loss), "KL loss 不能是 NaN/Inf"
    assert all(parameter.grad is None for parameter in teacher.parameters())
    print(f"OPD 一步完成：masked reverse KL = {loss.item():.4f}")
    print("提示：模型随机初始化，本练习只验证数据流和梯度。")


if __name__ == "__main__":
    main()
