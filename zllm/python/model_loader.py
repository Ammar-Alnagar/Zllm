# model_loader.py - Model Weight Loading for Mini-vLLM

import os
import torch
import json
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from .config import ModelConfig


class ModelLoader:
    """Model weight loader for Mini-vLLM"""

    def __init__(self, model_path: str):
        """Initialize model loader

        Args:
            model_path: Path to model directory or HuggingFace model name
        """
        self.model_path = model_path
        self.config: Optional[ModelConfig] = None
        self.state_dict: Optional[Dict[str, torch.Tensor]] = None

    def load_config(self) -> ModelConfig:
        """Load model configuration"""
        config_path = os.path.join(self.model_path, "config.json")

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
        else:
            # Use default Qwen3 configuration
            config_data = {
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 152064,
                "max_position_embeddings": 8192,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
            }

        # Convert to ModelConfig
        self.config = ModelConfig(
            hidden_size=config_data.get("hidden_size", 4096),
            intermediate_size=config_data.get("intermediate_size", 11008),
            num_hidden_layers=config_data.get("num_hidden_layers", 32),
            num_attention_heads=config_data.get("num_attention_heads", 32),
            num_key_value_heads=config_data.get("num_key_value_heads", 8),
            head_dim=config_data.get("head_dim", 128),
            vocab_size=config_data.get("vocab_size", 152064),
            max_position_embeddings=config_data.get("max_position_embeddings", 8192),
            rope_theta=config_data.get("rope_theta", 1000000.0),
            rms_norm_eps=config_data.get("rms_norm_eps", 1e-6),
            dtype=torch.float16,  # Default to FP16
        )

        return self.config

    def load_weights(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Load model weights

        Args:
            device: Device to load weights to

        Returns:
            Dictionary of model weights
        """
        if self.config is None:
            self.load_config()

        # Check for safetensors format first (preferred)
        safetensors_files = list(Path(self.model_path).glob("*.safetensors"))
        if safetensors_files:
            return self._load_safetensors(safetensors_files, device)

        # Fall back to PyTorch format
        pytorch_files = list(Path(self.model_path).glob("*.bin")) + list(
            Path(self.model_path).glob("pytorch_model*.bin")
        )

        if pytorch_files:
            return self._load_pytorch(pytorch_files, device)

        raise FileNotFoundError(f"No model weights found in {self.model_path}")

    def _load_safetensors(self, files: list, device: str) -> Dict[str, torch.Tensor]:
        """Load SafeTensors format weights"""
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors package required for SafeTensors format")

        state_dict = {}

        for file_path in files:
            with safe_open(file_path, framework="pt", device=device) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

        self.state_dict = state_dict
        return state_dict

    def _load_pytorch(self, files: list, device: str) -> Dict[str, torch.Tensor]:
        """Load PyTorch format weights"""
        state_dict = {}

        for file_path in files:
            checkpoint = torch.load(file_path, map_location=device, weights_only=True)
            state_dict.update(checkpoint)

        self.state_dict = state_dict
        return state_dict

    def get_embedding_weights(self) -> Optional[torch.Tensor]:
        """Get token embedding weights"""
        if self.state_dict is None:
            return None

        # Try common embedding weight names
        for name in [
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
        ]:
            if name in self.state_dict:
                return self.state_dict[name]

        return None

    def get_output_weights(self) -> Optional[torch.Tensor]:
        """Get output projection weights"""
        if self.state_dict is None:
            return None

        # Try common output weight names
        for name in ["lm_head.weight", "model.lm_head.weight"]:
            if name in self.state_dict:
                return self.state_dict[name]

        return None

    def get_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Get weights for a specific transformer layer

        Args:
            layer_idx: Layer index

        Returns:
            Dictionary of layer weights
        """
        if self.state_dict is None:
            return {}

        layer_weights = {}
        layer_prefixes = [
            f"layers.{layer_idx}",
            f"model.layers.{layer_idx}",
            f"transformer.h.{layer_idx}",
        ]

        for prefix in layer_prefixes:
            for key, tensor in self.state_dict.items():
                if key.startswith(prefix):
                    # Remove prefix to get relative key
                    relative_key = key[len(prefix) + 1 :]  # +1 for dot
                    layer_weights[relative_key] = tensor

        return layer_weights

    def get_attention_weights(self, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        """Get attention weights for a layer

        Returns:
            Tuple of (wq, wk, wv, wo) weights
        """
        layer_weights = self.get_layer_weights(layer_idx)

        # Try different naming conventions
        weight_names = [
            (
                "attention.wq.weight",
                "attention.wk.weight",
                "attention.wv.weight",
                "attention.wo.weight",
            ),
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
            ),
            (
                "attn.q_proj.weight",
                "attn.k_proj.weight",
                "attn.v_proj.weight",
                "attn.o_proj.weight",
            ),
        ]

        for wq_name, wk_name, wv_name, wo_name in weight_names:
            if all(
                name in layer_weights for name in [wq_name, wk_name, wv_name, wo_name]
            ):
                return (
                    layer_weights[wq_name],
                    layer_weights[wk_name],
                    layer_weights[wv_name],
                    layer_weights[wo_name],
                )

        raise KeyError(f"Attention weights not found for layer {layer_idx}")

    def get_mlp_weights(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get MLP weights for a layer

        Returns:
            Tuple of (w1, w2, w3) weights for SwiGLU
        """
        layer_weights = self.get_layer_weights(layer_idx)

        # Try different naming conventions
        weight_names = [
            (
                "feed_forward.w1.weight",
                "feed_forward.w2.weight",
                "feed_forward.w3.weight",
            ),
            ("mlp.gate_proj.weight", "mlp.down_proj.weight", "mlp.up_proj.weight"),
            (
                "feed_forward.gate_proj.weight",
                "feed_forward.down_proj.weight",
                "feed_forward.up_proj.weight",
            ),
        ]

        for w1_name, w2_name, w3_name in weight_names:
            if all(name in layer_weights for name in [w1_name, w2_name, w3_name]):
                return (
                    layer_weights[w1_name],
                    layer_weights[w2_name],
                    layer_weights[w3_name],
                )

        raise KeyError(f"MLP weights not found for layer {layer_idx}")

    def get_norm_weights(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalization weights for a layer

        Returns:
            Tuple of (attn_norm, mlp_norm) weights
        """
        layer_weights = self.get_layer_weights(layer_idx)

        # Try different naming conventions
        weight_names = [
            ("attention_norm.weight", "feed_forward_norm.weight"),
            ("input_layernorm.weight", "post_attention_layernorm.weight"),
            ("norm1.weight", "norm2.weight"),
        ]

        for attn_norm_name, mlp_norm_name in weight_names:
            if all(name in layer_weights for name in [attn_norm_name, mlp_norm_name]):
                return (layer_weights[attn_norm_name], layer_weights[mlp_norm_name])

        raise KeyError(f"Norm weights not found for layer {layer_idx}")

    def get_final_norm_weight(self) -> Optional[torch.Tensor]:
        """Get final layer norm weight"""
        if self.state_dict is None:
            return None

        # Try common final norm names
        for name in ["norm.weight", "model.norm.weight", "ln_f.weight"]:
            if name in self.state_dict:
                return self.state_dict[name]

        return None

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for the model

        Returns:
            Dictionary with memory estimates in GB
        """
        if self.config is None:
            self.load_config()

        # Calculate parameter count
        embedding_params = self.config.vocab_size * self.config.hidden_size
        output_params = self.config.vocab_size * self.config.hidden_size

        per_layer_params = (
            # Attention: 4 projections
            4 * self.config.hidden_size * self.config.hidden_size
            +
            # MLP: 3 projections
            3 * self.config.hidden_size * self.config.intermediate_size
            +
            # Norms: 2
            2 * self.config.hidden_size
        )

        total_params = (
            embedding_params
            + output_params
            + self.config.num_hidden_layers * per_layer_params
            + self.config.hidden_size  # Final norm
        )

        # Memory in bytes (FP16 = 2 bytes per param)
        memory_bytes = total_params * 2

        return {
            "total_parameters": total_params,
            "memory_gb_fp16": memory_bytes / (1024**3),
            "memory_gb_fp32": (total_params * 4) / (1024**3),
            "embedding_memory_gb": (embedding_params * 2) / (1024**3),
            "per_layer_memory_gb": (per_layer_params * 2) / (1024**3),
        }
