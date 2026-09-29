"""Reference temporal encoder and two distinct time series/LLM fusion paths."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class TemporalPatchEncoder(nn.Module):
    """Patchify a T×C signal and contextualize its valid patches."""

    def __init__(self, channels: int, hidden_size: int, patch_size: int = 16, layers: int = 2, heads: int = 4):
        super().__init__()
        if channels < 1 or patch_size < 1 or hidden_size % heads:
            raise ValueError("channels/patch_size must be positive and hidden_size divisible by heads")
        self.channels = channels
        self.patch_size = patch_size
        self.patch_projection = nn.Linear(channels * patch_size, hidden_size)
        block = nn.TransformerEncoderLayer(hidden_size, heads, hidden_size * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def position_encoding(length: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(length, hidden_size, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: hidden_size // 2])
        return encoding.to(dtype)

    def forward(self, series: torch.Tensor, series_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if series.ndim != 3 or series.shape[-1] != self.channels:
            raise ValueError(f"Expected series [batch, time, {self.channels}]")
        if series_mask.shape != series.shape[:2]:
            raise ValueError("series_mask must have shape [batch, time]")
        if not torch.all(series_mask.any(dim=1)):
            raise ValueError("Every series needs at least one valid time step")
        series = series.to(self.patch_projection.weight.dtype) * series_mask.unsqueeze(-1)
        pad = (-series.shape[1]) % self.patch_size
        if pad:
            series = F.pad(series, (0, 0, 0, pad))
            series_mask = F.pad(series_mask, (0, pad), value=False)
        batch, time, channels = series.shape
        patches = series.reshape(batch, time // self.patch_size, self.patch_size * channels)
        patch_mask = series_mask.reshape(batch, -1, self.patch_size).any(dim=-1)
        hidden = self.patch_projection(patches)
        hidden = hidden + self.position_encoding(hidden.shape[1], hidden.shape[2], hidden.device, hidden.dtype)
        hidden = self.encoder(hidden, src_key_padding_mask=~patch_mask)
        return self.norm(hidden) * patch_mask.unsqueeze(-1), patch_mask


class PreAlignModel(nn.Module):
    """Project temporal tokens into the LLM input embedding space."""

    def __init__(self, llm: nn.Module, encoder: TemporalPatchEncoder, encoder_size: int):
        super().__init__()
        self.llm = llm
        self.temporal_encoder = encoder
        self.aligner = nn.Sequential(nn.Linear(encoder_size, llm.config.hidden_size), nn.LayerNorm(llm.config.hidden_size))

    def set_stage(self, stage: str, train_encoder: bool | None = None) -> None:
        if train_encoder is None:
            train_encoder = stage == "perceive"
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        for parameter in self.temporal_encoder.parameters():
            parameter.requires_grad_(train_encoder)
        for parameter in self.aligner.parameters():
            parameter.requires_grad_(True)
        self.llm.eval()

    def train(self, mode: bool = True) -> "PreAlignModel":
        super().train(mode)
        self.llm.eval()  # Keep the frozen backbone's dropout disabled.
        if not any(p.requires_grad for p in self.temporal_encoder.parameters()):
            self.temporal_encoder.eval()
        return self

    def forward(self, series: torch.Tensor, series_mask: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        temporal, patch_mask = self.temporal_encoder(series, series_mask)
        temporal = self.aligner(temporal)
        text = self.llm.get_input_embeddings()(input_ids)
        embeddings = torch.cat([temporal, text], dim=1)
        full_mask = torch.cat([patch_mask.to(attention_mask.dtype), attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(patch_mask.shape, -100, device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([prefix_labels, labels], dim=1)
        outputs = self.llm(inputs_embeds=embeddings, attention_mask=full_mask, labels=full_labels, return_dict=True)
        return {"loss": outputs.loss, "logits": outputs.logits}


class PostAlignModel(nn.Module):
    """Let text hidden states attend to encoded temporal patches after the LLM."""

    def __init__(self, llm: nn.Module, encoder: TemporalPatchEncoder, encoder_size: int, heads: int = 8):
        super().__init__()
        hidden_size = llm.config.hidden_size
        if hidden_size % heads:
            raise ValueError("LLM hidden size must be divisible by post-align attention heads")
        self.llm = llm
        self.temporal_encoder = encoder
        self.temporal_projection = nn.Linear(encoder_size, hidden_size)
        self.cross_attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def set_stage(self, stage: str, train_encoder: bool | None = None) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def forward(self, series: torch.Tensor, series_mask: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        temporal, patch_mask = self.temporal_encoder(series, series_mask)
        temporal = self.temporal_projection(temporal)
        outputs = self.llm.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                      return_dict=True, use_cache=False)
        text = outputs.last_hidden_state
        attended, _ = self.cross_attention(text, temporal, temporal, key_padding_mask=~patch_mask, need_weights=False)
        hidden = self.norm(text + attended)
        logits = self.llm.get_output_embeddings()(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].contiguous().reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}


def build_model(config: dict[str, Any], llm: nn.Module) -> nn.Module:
    model_config = config["model"]
    encoder_size = int(model_config.get("encoder_size", 512))
    encoder = TemporalPatchEncoder(
        channels=int(model_config["channels"]), hidden_size=encoder_size,
        patch_size=int(model_config.get("patch_size", 16)),
        layers=int(model_config.get("encoder_layers", 2)),
        heads=int(model_config.get("encoder_heads", 8)),
    )
    architecture = model_config["architecture"]
    if architecture == "pre_align":
        return PreAlignModel(llm, encoder, encoder_size)
    if architecture == "post_align":
        return PostAlignModel(llm, encoder, encoder_size, int(model_config.get("cross_attention_heads", 8)))
    raise ValueError(f"Unknown architecture: {architecture}")
