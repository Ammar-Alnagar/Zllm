# sampling.py - Token sampling strategies for Mini-vLLM

import numpy as np
import torch
from typing import List, Optional, Tuple
from .config import SamplingParams


class Sampler:
    """Token sampling strategies for text generation"""

    def __init__(self, vocab_size: int = 152064):
        """Initialize sampler

        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size

    def sample(self, logits: np.ndarray, params: SamplingParams) -> int:
        """Sample next token from logits

        Args:
            logits: Raw logits from model [vocab_size]
            params: Sampling parameters

        Returns:
            Selected token ID
        """
        # Convert to numpy if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        # Apply temperature
        if params.temperature != 1.0:
            logits = logits / params.temperature

        # Apply top-k filtering
        if params.use_top_k:
            logits = self._apply_top_k(logits, params.top_k)

        # Apply top-p (nucleus) filtering
        if params.use_top_p:
            logits = self._apply_top_p(logits, params.top_p)

        # Convert to probabilities
        # Subtract max for numerical stability
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Sample from distribution
        token_id = np.random.choice(len(probs), p=probs)

        return int(token_id)

    def _apply_top_k(self, logits: np.ndarray, top_k: int) -> np.ndarray:
        """Apply top-k filtering

        Args:
            logits: Input logits
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits with low-probability tokens masked
        """
        # Find top-k indices
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]

        # Create mask
        mask = np.full_like(logits, -float("inf"))
        mask[top_k_indices] = logits[top_k_indices]

        return mask

    def _apply_top_p(self, logits: np.ndarray, top_p: float) -> np.ndarray:
        """Apply top-p (nucleus) filtering

        Args:
            logits: Input logits
            top_p: Cumulative probability threshold

        Returns:
            Filtered logits
        """
        # Convert to probabilities
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cutoff point
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumulative_probs, top_p, side="right")

        # If no tokens meet threshold, keep at least one
        cutoff_idx = max(1, cutoff_idx)

        # Create mask
        mask = np.full_like(logits, -float("inf"))

        # Get indices of tokens to keep
        keep_indices = sorted_indices[:cutoff_idx]
        mask[keep_indices] = logits[keep_indices]

        return mask

    def sample_batch(
        self, logits_batch: np.ndarray, params_list: List[SamplingParams]
    ) -> List[int]:
        """Sample tokens for a batch

        Args:
            logits_batch: Batch of logits [batch_size, vocab_size]
            params_list: List of sampling parameters

        Returns:
            List of sampled token IDs
        """
        assert len(logits_batch) == len(params_list), "Batch size mismatch"

        tokens = []
        for logits, params in zip(logits_batch, params_list):
            token = self.sample(logits, params)
            tokens.append(token)

        return tokens


class GreedySampler(Sampler):
    """Greedy sampling (always pick highest probability token)"""

    def sample(self, logits: np.ndarray, params: SamplingParams) -> int:
        """Greedy sampling - always pick highest probability"""
        return int(np.argmax(logits))


class RandomSampler(Sampler):
    """Pure random sampling (ignores logits)"""

    def sample(self, logits: np.ndarray, params: SamplingParams) -> int:
        """Random sampling - uniform distribution"""
        return np.random.randint(0, self.vocab_size)


# Factory function
def create_sampler(
    sampling_type: str = "multinomial", vocab_size: int = 152064
) -> Sampler:
    """Create sampler instance

    Args:
        sampling_type: Type of sampler ("multinomial", "greedy", "random")
        vocab_size: Vocabulary size

    Returns:
        Sampler instance
    """
    if sampling_type == "multinomial":
        return Sampler(vocab_size)
    elif sampling_type == "greedy":
        return GreedySampler(vocab_size)
    elif sampling_type == "random":
        return RandomSampler(vocab_size)
    else:
        raise ValueError(f"Unknown sampling type: {sampling_type}")
