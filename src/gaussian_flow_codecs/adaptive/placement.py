from .base import KernelInitialization, PlacementConfig
from .error_based import build_error_based_initialization
from .grid import build_uniform_kernel_layout
from .multiresolution import apply_multiresolution_overlay, build_multiresolution_initialization
from .omega_weighted import build_omega_weighted_initialization


def build_kernel_initialization(
    snapshot,
    n_gaussians: int,
    device,
    config: PlacementConfig,
) -> KernelInitialization:
    """
    Build the initial kernel set for the selected placement strategy.

    Current modes:
    - grid: uniform kernels over the full domain
    - adaptive: mixture of global grid kernels and omega-weighted local kernels
    - error_based: mixture of global kernels and error-targeted local kernels
    - multires: coarse and fine uniform kernel sets with different widths
    """

    if config.init_mode == "grid":
        kernel_init = build_uniform_kernel_layout(n_gaussians, device)
    elif config.init_mode == "adaptive":
        kernel_init = build_omega_weighted_initialization(snapshot, n_gaussians, device, config)
    elif config.init_mode == "error_based":
        kernel_init = build_error_based_initialization(snapshot, n_gaussians, device, config)
    elif config.init_mode == "multires":
        kernel_init = build_multiresolution_initialization(snapshot, n_gaussians, device, config)
    else:
        raise ValueError(f"Unsupported init_mode: {config.init_mode}")

    if config.use_multires_overlay and config.init_mode != "multires":
        return apply_multiresolution_overlay(kernel_init, config)
    return kernel_init
