from abc import ABC, abstractmethod
from diffusion import Diffusion
import torch


class ScoringStrategy(ABC):
    @abstractmethod
    def compute_scores(self, batch_ids: torch.Tensor, batch_mask: torch.Tensor):
        pass

    @abstractmethod
    def compute_raw(self, batch_ids: torch.Tensor, batch_mask: torch.Tensor):
        pass


class PseudoELBOScoring(ScoringStrategy):
    """Baseline Scoring Strategy"""

    def __init__(
        self,
        model: Diffusion,
        *,
        batch_size: int,
        normalize_by: str = "seq_length",  # also possible num_masked
    ):
        self.batch_size = batch_size
        self.model = model
        self.normalize_by = normalize_by
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def compute_raw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hyp_log_probs = []
        hyp_effective_lengths = []
        with torch.no_grad():
            for j in range(0, len(input_ids), self.batch_size):
                batch_ids = input_ids[j : j + self.batch_size]
                batch_mask = attention_mask[j : j + self.batch_size]

                loss_output = self.model._loss(batch_ids, batch_mask)

                if self.normalize_by == "seq_length":
                    lengths = batch_mask.sum(dim=-1)
                else:
                    non_zero_mask = (loss_output.nlls != 0.0).float()
                    lengths = (batch_mask * non_zero_mask).sum(dim=-1).clamp(min=1)

                batch_log_prob = -loss_output.nlls.sum(dim=-1)
                hyp_log_probs.append(batch_log_prob)
                hyp_effective_lengths.append(lengths)

        all_log_probs = torch.cat(hyp_log_probs)
        all_effective_lengths = torch.cat(hyp_effective_lengths)

        return all_log_probs, all_effective_lengths

    def compute_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        raw_scores, lengths = self.compute_raw(input_ids, attention_mask)
        return raw_scores / lengths.clamp(min=1)


class MonteCarloScoring(ScoringStrategy):
    """Monte Carlo Scoring Strategy"""

    def __init__(self, batch_size: int, num_sampling: int, aggregation: str, inner_strategy: ScoringStrategy):
        self.batch_size = batch_size
        self.num_sampling = num_sampling
        self.inner_strategy = inner_strategy
        self.aggregation = aggregation  # can be mean or sum_then_normalize or exclude_zeros

    def compute_raw(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        all_log_probs = []
        all_effective_lengths = []
        for i in range(self.num_sampling):
            seed = 42 + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            scores, length = self.inner_strategy.compute_raw(input_ids, attention_mask)
            all_log_probs.append(scores)
            all_effective_lengths.append(length)
        all_log_probs = torch.stack(all_log_probs)
        all_effective_lengths = torch.stack(all_effective_lengths)

        return all_log_probs, all_effective_lengths

    def compute_scores(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        all_samples, all_lengths = self.compute_raw(input_ids, attention_mask)
        return self._aggregate(all_samples, all_lengths)

    def _aggregate(self, all_samples: torch.Tensor, all_lengths: torch.Tensor):
        if self.aggregation == "mean":
            normalized = all_samples / all_lengths.clamp(min=1)
            return normalized.mean(dim=0)

        elif self.aggregation == "sum_then_normalize":
            total_scores = all_samples.sum(dim=0)
            total_lengths = all_lengths.sum(dim=0)
            return total_scores / total_lengths

        elif self.aggregation == "exclude_zeros":
            mask = all_samples != 0
            return (all_samples * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
