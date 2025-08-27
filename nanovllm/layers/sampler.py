import torch
from torch import nn

from nanovllm.triton_operators.softmax_online import triton_softmax_online


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        # probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs = triton_softmax_online(logits)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1) + epsilon
        ).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
