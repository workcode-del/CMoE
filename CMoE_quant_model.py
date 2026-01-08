import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Optional, Tuple, List
from compressed_tensors.linear.compressed_linear import CompressedLinear


class quantLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu"
    ):
        super().__init__()
        self.gate_proj = CompressedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = CompressedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = CompressedLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)

        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output

class quantRouter(nn.Module):
    def __init__(self, hidden_size, n_experts, n_activated, bias_speed = 0.001):
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = CompressedLinear(n_experts, hidden_size, bias=False).to(torch.bfloat16)
        self.classifier = CompressedLinear(n_experts, hidden_size, bias=False).to(torch.bfloat16)
    
    def update_bias(self, counts):
        mean_load = counts.mean()
        # Decrease bias for overloaded experts, increase for underloaded
        overloaded = counts > mean_load
        underloaded = counts < mean_load

        self.extra_bias.data[overloaded] -= self.bias_update_speed
        self.extra_bias.data[underloaded] += self.bias_update_speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.classifier is None:
            # scores = self.act_fn(self.gate(x)).abs()
            scores = self.gate(x)
        else:
            scores = (self.classifier(x) * self.act_fn(self.gate(x))).abs() 
            # scores = self.gate(x)

        # print(scores.shape)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        scores = scores + self.extra_bias.to(x.device)[None, :]

        weights, indices = torch.topk(scores, self.topk, dim=-1)

        # original_scores = 1 + original_scores*self.extra_scale.to(x.device)
        # weights = original_scores.gather(1, indices)

        return weights.type_as(x), indices
