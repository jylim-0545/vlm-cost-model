"""Config loader: model dims + cloud unit prices as typed objects.

Single source of truth for byte math and prices (CLAUDE.md Section 8: no magic
numbers inline). Everything downstream imports from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
KB = 1024
GB = 1024 ** 3


@dataclass(frozen=True)
class ModelSpec:
    """One target VLM. Byte sizes are derived, never stored."""
    key: str
    repo_id: str
    llm_backbone: str
    hidden_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    vision_encoder: str
    per_frame_tokens: int | str
    trust_remote_code: bool

    def vision_token_bytes(self, dtype_bytes: int) -> int:
        """Bytes for ONE stored vision token = hidden_size * dtype."""
        return self.hidden_size * dtype_bytes

    def kv_token_bytes(self, dtype_bytes: int) -> int:
        """Bytes for ONE token's KV cache = 2(k+v) * layers * kv_heads * head_dim * dtype."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes


@dataclass(frozen=True)
class StorageTier:
    """One storage tier: storage rate + bandwidth + egress. resource_price (the cost/s
    of the resource that stalls during retrieval, i.e. the H100) is NOT a tier knob —
    it comes from prices.yaml and is passed into network_cost_usd by the caller."""
    name: str
    usd_per_gb_month: float
    bandwidth_gbps: float
    egress_price_usd_per_gb: float

    def retrieval_time_s(self, read_bytes: int) -> float:
        return read_bytes / (self.bandwidth_gbps * 1e9)

    def network_cost_usd(self, read_bytes: int, resource_price_usd_per_s: float,
                         include_egress: bool = True) -> float:
        """Per-access network cost = egress (data transfer) + stalled-resource time.
        resource_price_usd_per_s is the $/s of the resource idled during retrieval
        (read from prices.yaml). S3 Standard has NO per-GB retrieval fee; egress is
        INTERNET-only (~0 same-region). include_egress=False drops the (volatile) egress."""
        egress = (read_bytes / 1e9) * self.egress_price_usd_per_gb if include_egress else 0.0
        return egress + self.retrieval_time_s(read_bytes) * resource_price_usd_per_s

    def storage_cost_usd(self, total_bytes: int, retention_days: float) -> float:
        return total_bytes / 1e9 * self.usd_per_gb_month * (retention_days / 30.0)


@dataclass(frozen=True)
class ModelConfig:
    dtype_name: str
    dtype_bytes: int
    models: dict[str, ModelSpec]

    def vision_bytes(self, model_key: str, n_vision_tokens: int) -> int:
        return self.models[model_key].vision_token_bytes(self.dtype_bytes) * n_vision_tokens

    def kv_bytes(self, model_key: str, n_vision_tokens: int) -> int:
        return self.models[model_key].kv_token_bytes(self.dtype_bytes) * n_vision_tokens


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_models() -> ModelConfig:
    raw = _load_yaml("models.yaml")
    dtype = raw["dtype"]
    models = {
        key: ModelSpec(key=key, **spec)
        for key, spec in raw["models"].items()
    }
    return ModelConfig(dtype_name=dtype["name"], dtype_bytes=int(dtype["bytes"]), models=models)


@lru_cache(maxsize=1)
def load_prices() -> dict[str, Any]:
    """Compute unit prices + run defaults (nested dict). See prices.yaml.
    Per-tier storage/network params live in storage_tiers.yaml (load_storage_tiers)."""
    return _load_yaml("prices.yaml")


@lru_cache(maxsize=1)
def load_storage_tiers() -> dict[str, StorageTier]:
    """Per-tier storage rate + network/retrieval params, swept by the price model."""
    raw = _load_yaml("storage_tiers.yaml")
    return {name: StorageTier(name=name, **params) for name, params in raw["tiers"].items()}
