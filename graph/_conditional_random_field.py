import torch
import time
from torch import nn
import numpy as np

torch.manual_seed(int(time.time()))


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(scores: torch.Tensor) -> torch.Tensor:
    max_score, _ = scores.max(dim=0)
    return (scores - max_score).exp().sum(dim=0).log() + max_score


class CRF(nn.Module):
    """Implementation of CRF model (with minor midification)."""

    def __init__(self, tags: list):
        super(CRF, self).__init__()
        self._idx2tag = tags
        self._tag2idx = {tag: index for index, tag in
                         enumerate(self._idx2tag)}
        self._n_tags = len(self._idx2tag)
        # Matrix of transition parameters.
        # Entry i,j is the score of transitioning from i to j.
        self._transitions = nn.Parameter(torch.randn(self._n_tags, self._n_tags,
                                                     dtype=torch.float32),
                                         requires_grad=True)
        # Weight of emission probability, used to balance the influence of
        # inbalanced data.
        self._tag_weight = nn.Parameter(torch.ones(self._n_tags,
                                                   dtype=torch.float32),
                                        requires_grad=True)

    def neg_log_likelihood(self, emit_scores: np.ndarray,
                           tags: np.ndarray) -> torch.Tensor:
        # Calculate log sum exp of all possible paths using dynamic programming
        emit_scores = torch.tensor(emit_scores, dtype=torch.float32)
        emit_scores *= self._tag_weight
        prev = emit_scores[0]
        # Iterate through the sentence
        for obs in emit_scores[1:]:
            prev_exp = prev.expand(self._n_tags, self._n_tags).t()
            obs_exp = obs.expand(self._n_tags, self._n_tags)
            scores = prev_exp + obs_exp + self._transitions
            # Calculate log sum exp of scores
            prev = log_sum_exp(scores)
        total_score = log_sum_exp(prev)

        # Calculate score for given tag sequence
        idx_list = [self._tag2idx[tag] for tag in tags]
        curr_score = emit_scores[0][idx_list[0]]
        for i in range(1, len(idx_list)):
            curr_score += self._transitions[idx_list[i - 1], idx_list[i]]
            curr_score += emit_scores[i, idx_list[i]]
        return total_score - curr_score

    def forward(self, emit_scores: np.ndarray) -> list:
        backpointers = []
        # Viterbi
        emit_scores = torch.tensor(emit_scores, dtype=torch.float32)
        emit_scores *= self._tag_weight
        prev = emit_scores[0]
        for obs in emit_scores[1:]:
            prev_exp = prev.expand(self._n_tags, self._n_tags).t()
            obs_exp = obs.expand(self._n_tags, self._n_tags)
            scores = prev_exp + obs_exp + self._transitions
            prev, max_prev_idx = scores.max(dim=0)
            backpointers.append(max_prev_idx)
        _, idx = prev.max(dim=0)
        # Backtrack path
        path = [idx]
        for max_prev_idx in backpointers[::-1]:
            curr = path[-1]
            path.append(max_prev_idx[curr].item())
        path = [self._idx2tag[idx] for idx in path[::-1]]
        return path
