"""The ZLM decoder and its optimizers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 49_152
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_layers: int = 38
    num_heads: int = 16
    num_kv_heads: int = 4
    context_length: int = 4096
    extended_context_length: int = 16_384
    rope_theta: float = 100_000.0
    rms_norm_eps: float = 1e-6
    init_std: float = 0.02
    seed: int = 137



class RMSNorm(nn.Module):
    def __init__(self, dimension, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return (values * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.active_context = config.context_length
        self.yarn = False
        dimension = config.hidden_size // config.num_heads
        self.register_buffer('cos', torch.empty(config.context_length, dimension // 2, dtype=torch.float32), persistent=False)
        self.register_buffer('sin', torch.empty_like(self.cos), persistent=False)

    @torch.no_grad()
    def set_phase(self, yarn):
        """Switch between the base and YaRN phases; compile again after switching."""
        config = self.config
        dimension = config.hidden_size // config.num_heads
        device = self.cos.device
        self.yarn = yarn
        self.active_context = config.extended_context_length if yarn else config.context_length
        with torch.autocast(device_type=device.type, enabled=False):
            inv_freq = config.rope_theta ** (-torch.arange(0, dimension, 2, device=device, dtype=torch.float32) / dimension)
            scale = 1.0
            if yarn:
                factor = config.extended_context_length / config.context_length
                low = math.floor(dimension * math.log(config.context_length / (32 * 2 * math.pi)) / (2 * math.log(config.rope_theta)))
                high = math.ceil(dimension * math.log(config.context_length / (2 * math.pi)) / (2 * math.log(config.rope_theta)))
                low, high = max(low, 0), min(high, dimension - 1)
                ramp = ((torch.arange(dimension // 2, device=device, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
                inv_freq = inv_freq * (1 - ramp) + (inv_freq / factor) * ramp
                scale = 1 + 0.1 * math.log(factor)
            angles = torch.outer(torch.arange(self.active_context, device=device, dtype=torch.float32), inv_freq)
            self.cos = angles.cos() * scale
            self.sin = angles.sin() * scale

    def forward(self, length, start=0):
        positions = slice(start, start + length)
        return self.cos[positions][None, None], self.sin[positions][None, None]


def apply_rotary(x, cos, sin):
    dtype = x.dtype
    pairs = x.float().unflatten(-1, (-1, 2))
    first, second = pairs.unbind(-1)
    return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2).to(dtype)


class KVCache:
    """Request-local storage. Start a new cache for each prompt and RoPE phase."""

    def __init__(self, model, batch_size, capacity):
        config = model.config
        shape = (config.num_layers, batch_size, config.num_kv_heads, capacity,
                 config.hidden_size // config.num_heads)
        weight = model.embed_tokens.weight
        self.keys = weight.new_empty(shape)
        self.values = weight.new_empty(shape)
        self.length = 0


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        kv_size = self.num_kv_heads * self.head_dim
        # Separate matrices preserve the approved Muon update geometry.
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=torch.float32)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False, dtype=torch.float32)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False, dtype=torch.float32)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=torch.float32)

    def forward(self, x, cos, sin, cache=None, start=0):
        batch, length, _ = x.shape
        query = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)
        if cache is not None:
            end = start + length
            keys, values = cache
            keys[:, :, start:end].copy_(key)
            values[:, :, start:end].copy_(value)
            key, value = keys[:, :, :end], values[:, :, :end]
        # A single decode query can attend to the entire populated prefix.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                                  is_causal=False if start else True, scale=self.scale, enable_gqa=True)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=torch.float32)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=torch.float32)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=torch.float32)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, cache=None, start=0):
        x = x + self.attention(self.attention_norm(x), cos, sin, cache, start)
        return x + self.mlp(self.mlp_norm(x))


class ZLM(nn.Module):
    def __init__(self, config=ModelConfig(), *, device='cpu'):
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        # Construct on meta to avoid initializing every matrix twice.
        with torch.device('meta'):
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float32)
            self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.rope = RotaryEmbedding(config)
        if torch.device(device).type != 'meta':
            self.to_empty(device=device)
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        generator = torch.Generator(device=self.embed_tokens.weight.device).manual_seed(self.config.seed)
        residual_std = self.config.init_std / math.sqrt(2 * self.config.num_layers)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = residual_std if name.endswith(('o_proj', 'down_proj')) else self.config.init_std
                nn.init.normal_(module.weight, mean=0, std=std, generator=generator)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        self.rope.set_phase(False)

    def hidden_states(self, input_ids):
        cos, sin = self.rope(input_ids.shape[1])
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, cos, sin, use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = layer(hidden, cos, sin)
        return self.norm(hidden)

    @torch.no_grad()
    def forward_cached(self, input_ids, cache):
        """Prefill an empty cache or append one token. Return last-token logits [B, V]."""
        start, length = cache.length, input_ids.shape[1]
        cos, sin = self.rope(length, start)
        hidden = self.embed_tokens(input_ids)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, (cache.keys[index], cache.values[index]), start)
        logits = F.linear(self.norm(hidden[:, -1]), self.embed_tokens.weight)
        cache.length = start + length
        return logits

    def forward(self, input_ids):
        # The output projection uses the embedding parameter directly, exactly once.
        return F.linear(self.hidden_states(input_ids), self.embed_tokens.weight)


def build_optimizers(model, *, muon_lr, adamw_lr):
    """Muon for linear matrices; AdamW for the tied embedding and norms."""
    matrices = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
    matrix_ids = {id(parameter) for parameter in matrices}
    other = [parameter for parameter in model.parameters() if id(parameter) not in matrix_ids]
    muon = torch.optim.Muon(matrices, lr=muon_lr, weight_decay=0.1, momentum=0.95,
                            nesterov=True, ns_coefficients=(3.4445, -4.7750, 2.0315),
                            eps=1e-7, ns_steps=5, adjust_lr_fn='match_rms_adamw')
    adamw = torch.optim.AdamW(other, lr=adamw_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                              foreach=False, fused=model.embed_tokens.weight.is_cuda)
    return muon, adamw
