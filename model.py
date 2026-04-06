from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_next: torch.Tensor | None = None
    loss_next2: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scaling_factor: float = 1.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cached_seq_len: int = 0

    def _build_cache(self, seq_len: int, device, dtype):
        if (
            self._cos_cached is not None
            and self._cached_seq_len >= seq_len
            and self._cos_cached.device == device
            and self._cos_cached.dtype == dtype
        ):
            return (
                self._cos_cached[:, :, :seq_len, :],
                self._sin_cached[:, :, :seq_len, :],
            )
        t = torch.arange(seq_len, device=device, dtype=dtype) / self.scaling_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()[None, None, :, :]
        self._sin_cached = emb.sin()[None, None, :, :]
        self._cached_seq_len = seq_len
        return self._cos_cached, self._sin_cached

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos, sin = self._build_cache(seq_len, q.device, q.dtype)
        q = (q * cos[:, :, : q.size(2), :]) + (
            self._rotate_half(q) * sin[:, :, : q.size(2), :]
        )
        k = (k * cos[:, :, : k.size(2), :]) + (
            self._rotate_half(k) * sin[:, :, : k.size(2), :]
        )
        return q, k


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rotary_base: int,
        rotary_scaling_factor: float,
    ):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim,
            base=rotary_base,
            scaling_factor=rotary_scaling_factor,
            max_seq_len=max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.rope.apply(q, k, t)

        # Expand K/V for GQA without repeat_interleave to save a tiny bit of
        # intermediate memory; expand is a zero-copy view until written to.
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(b, self.n_kv_heads, self.n_rep, t, self.head_dim)
            k = k.reshape(b, self.n_heads, t, self.head_dim)
            v = v.unsqueeze(2).expand(b, self.n_kv_heads, self.n_rep, t, self.head_dim)
            v = v.reshape(b, self.n_heads, t, self.head_dim)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, self.dim)
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        rotary_base: int,
        rotary_scaling_factor: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim, n_heads, n_kv_heads, max_seq_len, rotary_base, rotary_scaling_factor
        )
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rotary_base: int = 10000,
        rotary_scaling_factor: float = 1.0,
        tie_embeddings: bool = True,
        pad_token_id: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.tie_embeddings = tie_embeddings
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    d_ff,
                    dropout,
                    max_seq_len,
                    rotary_base,
                    rotary_scaling_factor,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not self.tok_embeddings.weight:
                    nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        mtp_weight_1: float = 1.0,
        mtp_weight_2: float = 0.5,
    ) -> ModelOutput:
        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = ckpt.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = loss_next = loss_next2 = None
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

            # next-token loss
            shift_logits_1 = logits[:, :-1, :].contiguous()
            shift_labels_1 = labels[:, :-1].contiguous()
            loss_next = loss_fct(
                shift_logits_1.view(-1, self.vocab_size), shift_labels_1.view(-1)
            )

            # next-next-token auxiliary loss
            shift_logits_2 = (
                logits[:, :-2, :].contiguous()
                if logits.size(1) > 2
                else logits[:, :0, :]
            )
            shift_labels_2 = (
                labels[:, 1:-1].contiguous() if labels.size(1) > 2 else labels[:, :0]
            )
            if shift_logits_2.numel() > 0 and shift_labels_2.numel() > 0:
                loss_next2 = loss_fct(
                    shift_logits_2.view(-1, self.vocab_size), shift_labels_2.view(-1)
                )
                loss = mtp_weight_1 * loss_next + mtp_weight_2 * loss_next2
            else:
                loss = loss_next

        return ModelOutput(
            logits=logits, loss=loss, loss_next=loss_next, loss_next2=loss_next2
        )