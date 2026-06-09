"""LMCache Encoder-Cache connector for vLLM — vendored from vLLM PR #38668
("[ECConnector] LMCache EC connector entrypoint", open/unmerged as of 2026-06-09).

Kept in THIS repo (not vLLM's site-packages) on purpose: vLLM's ECConnectorFactory
falls back to `ec_connector_module_path` for any connector name not in its registry
(see factory.get_connector_class), and that path may be ANY importable module. So we
point vLLM at this module instead of patching vLLM:

    ec_transfer_config = ECTransferConfig(
        ec_connector="LMCacheECConnector",
        ec_role="ec_both",
        ec_connector_module_path="measure.lmcache_ec_connector",
    )

This thin shim subclasses vLLM's ECConnectorBase and delegates to lmcache 0.4.6's
`LMCacheECConnectorImpl` (interface verified compatible: __init__(vllm_config, role,
parent) + start_load_caches/save_caches/has_cache_item/update_state_after_alloc/
build_connector_meta). Requires the worker process to have the repo root on PYTHONPATH.
"""
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request


class LMCacheECConnector(ECConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        try:
            from lmcache.integration.vllm.vllm_ec_adapter import (  # type: ignore[import-not-found]
                LMCacheECConnectorImpl,
            )
        except ImportError as e:
            raise ImportError(
                "LMCacheECConnector requires lmcache to be installed."
            ) from e
        self._impl = LMCacheECConnectorImpl(
            vllm_config=vllm_config,
            role=role,
            parent=self,
        )

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs: Any
    ) -> None:
        return self._impl.start_load_caches(encoder_cache, **kwargs)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs: Any
    ) -> None:
        return self._impl.save_caches(encoder_cache, mm_hash, **kwargs)

    def has_cache_item(self, identifier: str) -> bool:
        return self._impl.has_cache_item(identifier)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        return self._impl.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        return self._impl.build_connector_meta(scheduler_output)
