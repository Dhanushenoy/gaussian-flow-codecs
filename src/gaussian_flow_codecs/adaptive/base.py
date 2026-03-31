from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KernelInitialization:
    """Initial kernel parameters before any trainable reparameterization."""

    centers: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class PlacementConfig:
    """Configuration shared by kernel placement strategies."""

    init_mode: str = "adaptive"
    adaptive_fraction: float = 0.5
    adaptive_power: float = 1.0
    global_sigma_scale: float = 1.0
    local_sigma_scale: float = 0.5
    blend_mode: str = "normalized"
    error_warmup_steps: int = 100
    error_warmup_batch_size: int = 10000
    error_eval_chunk_size: int = 10000
    error_warmup_lr: float = 1e-2
    multires_fine_fraction: float = 0.5
    multires_coarse_sigma_scale: float = 1.5
    multires_fine_sigma_scale: float = 0.5
    use_multires_overlay: bool = False
