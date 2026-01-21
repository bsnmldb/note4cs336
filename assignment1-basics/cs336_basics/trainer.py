from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

def cross_entropy_loss(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True).values
    stabilized_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(stabilized_logits), dim=-1)) + max_logits
    loss = log_sum_exp - torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = state.get("v", torch.zeros_like(p.data))
                v = beta2 * v + (1 - beta2) * grad**2
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
    
def lr_consine_schedule(current_step: int, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return lr_max * current_step / warmup_steps
    elif current_step > total_steps:
        return lr_min
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine_decay
    
def grad_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
