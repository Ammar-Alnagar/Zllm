# tokenizer.py - Tokenizer wrapper for Mini-vLLM

import tiktoken
from typing import List, Optional


class Tokenizer:
    """Tokenizer wrapper using tiktoken (used by GPT models)"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize tokenizer

        Args:
            model_name: tiktoken model name (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Common special tokens
        self.special_tokens = {
            "<|endoftext|>": 100257,  # Common EOS token
            "<|im_start|>": 100264,
            "<|im_end|>": 100265,
        }

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID"""
        return self.special_tokens["<|endoftext|>"]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""
        return self.encoding.n_vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        if add_special_tokens:
            # Add BOS token at the beginning for some models
            tokens = [self.special_tokens["<|im_start|>"]]
            tokens.extend(self.encoding.encode(text))
        else:
            tokens = self.encoding.encode(text)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special tokens
            filtered_ids = []
            for token_id in token_ids:
                if token_id not in self.special_tokens.values():
                    filtered_ids.append(token_id)
            token_ids = filtered_ids

        return self.encoding.decode(token_ids)

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        """Encode a batch of texts

        Args:
            texts: List of input texts
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token ID lists
        """
        return [self.encode(text, add_special_tokens) for text in texts]

    def decode_batch(
        self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token ID lists

        Args:
            token_ids_batch: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return [
            self.decode(token_ids, skip_special_tokens) for token_ids in token_ids_batch
        ]

    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encode(text, add_special_tokens=False))
