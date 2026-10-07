"""Offline numerical and gradient checks for interview mathematics.

This validates mathematical identities and worked examples, not unfinished TODOs.
Run: .venv/bin/python scripts/check_math.py
"""
from math import comb, log
import unittest
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
DTYPE = torch.float64


def tensor(values, **kwargs):
    return torch.tensor(values, dtype=DTYPE, **kwargs)


def kl(p, q):
    return (p * (p.log() - q.log())).sum()


class InterviewMath(unittest.TestCase):
    def close(self, actual, expected):
        torch.testing.assert_close(torch.as_tensor(actual, dtype=DTYPE),
                                   torch.as_tensor(expected, dtype=DTYPE), rtol=1e-7, atol=1e-8)

    def test_nll_and_shift_mask(self):
        logits = tensor([[[0., 1.], [1000., 1001.], [2., 0.]]])
        labels = torch.tensor([[-100, 1, 0]])
        shifted = labels[:, 1:]
        mask = shifted != -100
        logp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, shifted.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        self.close(-(logp * mask).sum() / mask.sum(),
                   F.cross_entropy(logits[:, :-1].reshape(-1, 2), shifted.reshape(-1)))
        self.close(-tensor([0.5, 0.25]).log().sum(), -log(0.125))

    def test_kl_directions_and_zero(self):
        p, q = tensor([.9, .1]), tensor([.5, .5])
        self.close(kl(p, p), 0.)
        self.close(kl(p, q), .3680642071684971)
        self.close(kl(q, p), .5108256237659907)

    def test_policy_gradient_baseline(self):
        logits = tensor([0., 0.], requires_grad=True)
        p = logits.softmax(-1)
        rewards = tensor([1., 0.])
        exact = torch.autograd.grad((p * rewards).sum(), logits, retain_graph=True)[0]
        # Enumerate score-function samples with fixed sampling weights and baseline.
        surrogate = (p.detach() * (rewards - .3) * p.log()).sum()
        estimate = torch.autograd.grad(surrogate, logits)[0]
        self.close(exact, [.25, -.25])
        self.close(estimate, exact)

    def test_fixed_state_importance_ratio(self):
        old, new, advantage = tensor([.2, .8]), tensor([.3, .7]), tensor([2., -1.])
        self.close((old * (new / old) * advantage).sum(), (new * advantage).sum())

    def test_ppo_clip_values_and_gradients(self):
        expected = {2.: [1.4, 2., 2.4], -2.: [-1.6, -2., -2.6]}
        for advantage in expected:
            ratio = tensor([.7, 1., 1.3], requires_grad=True)
            target = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            self.close(target, expected[advantage])
            grad = torch.autograd.grad(target.sum(), ratio)[0]
            self.close(grad, [2., 2., 0.] if advantage > 0 else [0., -2., -2.])

    def test_gae_terminal_and_truncation(self):
        reward, values = tensor([0., 0., 1.]), tensor([0., 0., 0.])
        delta = reward - values
        gae = torch.zeros_like(delta)
        accumulator = tensor(0.)
        for t in reversed(range(3)):
            accumulator = delta[t] + .5 * accumulator
            gae[t] = accumulator
        self.close(gae, [.25, .5, 1.])
        # Same reward, different boundary: true terminal has no bootstrap;
        # truncation bootstraps final V but ends the trace into another episode.
        r, gamma, v, final_v = 1., .9, 2., 3.
        self.close(r + gamma * 0 * final_v - v, -1.)
        self.close(r + gamma * 1 * final_v - v, 1.7)

    def test_dpo_optimal_policy_and_constant_reward_shift(self):
        reference, reward, beta = tensor([.5, .5]), tensor([1., 0.]), 1.
        best = (reference.log() + reward / beta).softmax(-1)
        self.close(best, [.7310585786300049, .2689414213699951])
        self.close(best, (reference.log() + (reward + 5.) / beta).softmax(-1))
        log_z = (reference.log() + reward / beta).logsumexp(-1)
        for probability in [.1, .4, .6, .9]:
            p = tensor([probability, 1-probability])
            objective = (p * reward).sum() - beta * kl(p, reference)
            self.close(objective, beta * log_z - beta * kl(p, best))

    def test_dpo_loss_gradient(self):
        delta = tensor(2., requires_grad=True)
        beta = .1
        loss = -F.logsigmoid(beta * delta)
        self.close(loss, .5981388693815918)
        self.close(torch.autograd.grad(loss, delta)[0], -beta * torch.sigmoid(-beta * delta.detach()))
        self.close(-F.logsigmoid(tensor(0.)), log(2.))

    def test_grpo_std_convention_and_single_sample(self):
        rewards = tensor([[1., 3., 5.], [2., 2., 2.]])
        advantage = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True, correction=0) + 1e-8)
        self.close(advantage[0], tensor([-1., 0., 1.]) * (1.5 ** .5))
        self.close(advantage[1], [0., 0., 0.])
        single = tensor([[3.]])
        single_adv = (single - single.mean(-1, keepdim=True)) / (single.std(-1, keepdim=True, correction=0) + 1e-8)
        self.close(single_adv, [[0.]])

    def test_k3_direction_sampling_and_gradient_caveat(self):
        current, reference = tensor([.5, .5]), tensor([.9, .1])
        u = reference.log() - current.log()
        k3 = torch.expm1(u) - u
        self.assertTrue(bool((k3 >= 0).all()))
        self.close((current * k3).sum(), kl(current, reference))
        # Old-sample expectation without importance weighting is different.
        old = tensor([.8, .2])
        self.assertGreater(abs(float((old * k3).sum() - kl(current, reference))), .1)
        self.close((old * (current/old) * k3).sum(), kl(current, reference))
        # Demonstrate that unbiased value estimates do not guarantee correct
        # gradients when sampling weights are detached.
        logits = tensor([0., 0.], requires_grad=True)
        p = logits.softmax(-1)
        u = reference.log() - p.log()
        naive = (p.detach() * (torch.expm1(u)-u)).sum()
        exact_grad = torch.autograd.grad(kl(p, reference), logits, retain_graph=True)[0]
        naive_grad = torch.autograd.grad(naive, logits)[0]
        self.assertGreater(float((exact_grad-naive_grad).abs().max()), .01)

    def test_opd_forward_and_reverse_gradients(self):
        teacher = tensor([.8, .2])
        logits = tensor([0., 0.], requires_grad=True)
        student = logits.softmax(-1)
        forward = kl(teacher, student)
        self.close(forward, .19274475702175753)
        self.close(torch.autograd.grad(forward, logits, retain_graph=True)[0], [-.3, .3])
        reverse = kl(student, teacher)
        self.close(reverse, .22314355131420976)
        exact_grad = torch.autograd.grad(reverse, logits, retain_graph=True)[0]
        expected_score_surrogate = (student.detach() * (student.log()-teacher.log()+1).detach() * student.log()).sum()
        self.close(torch.autograd.grad(expected_score_surrogate, logits, retain_graph=True)[0], exact_grad)
        naive_sample = (student.detach() * (student.log()-teacher.log())).sum()
        self.close(torch.autograd.grad(naive_sample, logits)[0], [0., 0.])
        self.assertGreater(float(exact_grad.abs().max()), .01)

    def test_rope_relative_position(self):
        def rotate(x, position):
            angle = tensor(.1 * position)
            c, s = angle.cos(), angle.sin()
            return torch.stack((c*x[0]-s*x[1], s*x[0]+c*x[1]))
        q, k = tensor([1., 0.]), tensor([1., 0.])
        self.close(rotate(q, 2) @ rotate(k, 5), rotate(q, 8) @ rotate(k, 11))
        self.close(rotate(q, 2).norm(), q.norm())

    def test_flash_online_softmax(self):
        scores, value = tensor([0., log(2.)]), tensor([1., 3.])
        maximum = scores.max()
        denominator = sum(torch.exp(s-maximum) for s in scores)
        numerator = sum(torch.exp(s-maximum)*v for s, v in zip(scores, value))
        self.close(numerator/denominator, scores.softmax(-1) @ value)
        self.close(numerator/denominator, 7/3)

    def test_normalization_and_gradient_accumulation_examples(self):
        self.close(tensor([3., 4.]) / tensor([3., 4.]).square().mean().sqrt(), [.848528137423857, 1.131370849898476])
        self.assertEqual(2*2*4096*8*128*2, 32*1024**2)
        w = tensor(2., requires_grad=True)
        data = tensor([1., 2., 3., 4.])
        full_grad = torch.autograd.grad((w*data).square().mean(), w)[0]
        w = tensor(2., requires_grad=True)
        for micro in [data[:1], data[1:]]:
            ((w*micro).square().mean() * micro.numel()/data.numel()).backward()
        self.close(w.grad, full_grad)

    def test_adamw_example(self):
        parameter = torch.nn.Parameter(tensor(1.))
        optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=.01, eps=0.)
        parameter.grad = tensor(2.)
        optimizer.step()
        self.close(parameter.detach(), .899)

    def test_speculative_distribution_and_passk(self):
        p, q = tensor([.7, .3]), tensor([.4, .6])
        acceptance = torch.minimum(torch.ones_like(p), p/q)
        residual = (p-q).clamp_min(0)
        rejection_probability = 1 - (q*acceptance).sum()
        self.close(q*acceptance + rejection_probability*residual/residual.sum(), p)
        for k, expected in [(1,.2),(2,17/45),(5,7/9)]:
            self.close(1-comb(8,k)/comb(10,k), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
