from abc import ABC, abstractmethod
from diffusion import Diffusion
import torch


def sample_bernoulli_mask(mask_prob: float, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = torch.rand(batch_size, seq_len, device=attention_mask.device) < mask_prob
    mask = mask & attention_mask.bool()
    return mask


def sample_t_bernoulli_mask(
    attention_mask: torch.Tensor,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    t = torch.rand(batch_size, device=attention_mask.device) * (t_max - t_min) + t_min
    mask = torch.rand(batch_size, seq_len, device=attention_mask.device) < t[:, None]
    mask = mask & attention_mask.bool()
    return mask


def sample_multinomial_mask(num_masks: int, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    weights = attention_mask.float()
    min_len = attention_mask.sum(dim=-1).min().item()
    n = min(num_masks, int(min_len))

    if n == 0:
        return torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    indices = torch.multinomial(weights, num_samples=n, replacement=False)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)
    return mask


def indices_to_mask(indices: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=attention_mask.device)
    mask.scatter_(1, indices, True)
    mask = mask & attention_mask.bool()
    return mask


class ScoringStrategy(ABC):
    @abstractmethod
    def compute_scores(
        self,
        batch_ids: torch.Tensor,
        batch_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        pass

    @abstractmethod
    def compute_raw(
        self,
        batch_ids: torch.Tensor,
        batch_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        pass


class SingleMaskScoring(ScoringStrategy):
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

    def compute_raw(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        mask_fn_kwargs = mask_fn_kwargs or {}
        hyp_log_probs = []
        hyp_effective_lengths = []
        with torch.no_grad():
            for j in range(0, len(input_ids), self.batch_size):
                batch_ids = input_ids[j : j + self.batch_size]
                batch_mask = attention_mask[j : j + self.batch_size]
                mask = mask_fn(attention_mask=batch_mask, **mask_fn_kwargs)
                loss_output = self.model._loss(batch_ids, batch_mask, diffusion_mask=mask)
                if self.normalize_by == "seq_length":
                    lengths = batch_mask.sum(dim=-1)
                else:
                    lengths = loss_output.num_masked.clamp(min=1)

                batch_log_prob = -loss_output.nlls.sum(dim=-1)
                hyp_log_probs.append(batch_log_prob)
                hyp_effective_lengths.append(lengths)

        all_log_probs = torch.cat(hyp_log_probs)
        all_effective_lengths = torch.cat(hyp_effective_lengths)

        return all_log_probs, all_effective_lengths

    def compute_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        raw_scores, lengths = self.compute_raw(input_ids, attention_mask, mask_fn, mask_fn_kwargs)
        return raw_scores / lengths.clamp(min=1)


class MonteCarloScoring(ScoringStrategy):
    """Monte Carlo Scoring Strategy"""

    def __init__(self, num_sampling: int, aggregation: str, inner_strategy: ScoringStrategy):
        self.num_sampling = num_sampling
        self.inner_strategy = inner_strategy
        self.aggregation = aggregation  # can be mean or sum_then_normalize or exclude_zeros

    def compute_raw(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        all_log_probs = []
        all_effective_lengths = []
        for i in range(self.num_sampling):
            seed = 42 + i
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            scores, length = self.inner_strategy.compute_raw(input_ids, attention_mask, mask_fn, mask_fn_kwargs)
            all_log_probs.append(scores)
            all_effective_lengths.append(length)
        all_log_probs = torch.stack(all_log_probs)
        all_effective_lengths = torch.stack(all_effective_lengths)

        return all_log_probs, all_effective_lengths

    def compute_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        all_samples, all_lengths = self.compute_raw(input_ids, attention_mask, mask_fn, mask_fn_kwargs)
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


class CouplingMaskStrategy(ScoringStrategy):
    def __init__(
        self,
        model: Diffusion,
        *,
        batch_size: int,
    ):
        self.batch_size = batch_size
        self.model = model
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def compute_raw(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        mask_fn_kwargs = mask_fn_kwargs or {}
        hyp_log_probs = []
        hyp_effective_lengths = []
        with torch.no_grad():
            for j in range(0, len(input_ids), self.batch_size):
                batch_ids = input_ids[j : j + self.batch_size]
                batch_mask = attention_mask[j : j + self.batch_size]
                mask_1 = mask_fn(attention_mask=batch_mask, **mask_fn_kwargs)
                loss_output_1 = self.model._loss(batch_ids, batch_mask, diffusion_mask=mask_1)
                mask_2 = (~mask_1) & batch_mask.bool()
                loss_output_2 = self.model._loss(batch_ids, batch_mask, diffusion_mask=mask_2)
                lengths = batch_mask.sum(dim=-1)
                batch_log_prob = -loss_output_1.nlls.sum(dim=-1) + -loss_output_2.nlls.sum(dim=-1)
                hyp_log_probs.append(batch_log_prob)
                hyp_effective_lengths.append(lengths)

        all_log_probs = torch.cat(hyp_log_probs)
        all_effective_lengths = torch.cat(hyp_effective_lengths)

        return all_log_probs, all_effective_lengths

    def compute_scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_fn: callable = sample_bernoulli_mask,
        mask_fn_kwargs: dict = None,
    ):
        raw_scores, lengths = self.compute_raw(input_ids, attention_mask, mask_fn, mask_fn_kwargs)
        return raw_scores / lengths.clamp(min=1)
