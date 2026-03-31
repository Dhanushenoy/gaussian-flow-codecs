import numpy as np
import torch

from gaussian_flow_codecs.metrics import compute_vorticity_np

from .base import KernelInitialization, PlacementConfig
from .grid import build_uniform_kernel_layout, select_spread_subset


def sample_vorticity_weighted_centers(snapshot, n_samples: int, adaptive_power: float) -> torch.Tensor:
    """
    Sample centers from the true field with probability biased by |omega|.

    This keeps support everywhere through the additive floor while allocating
    more kernels to rotationally active regions.
    """

    spacing = tuple(1.0 / (dim - 1) for dim in snapshot.dimensions)
    vorticity = compute_vorticity_np(snapshot.velocity, spacing)
    vorticity_mag = np.linalg.norm(vorticity, axis=-1)

    weights = np.power(vorticity_mag + 1e-6, adaptive_power)
    probabilities = weights.reshape(-1)
    probabilities /= probabilities.sum()

    replace = n_samples > probabilities.size
    indices = np.random.choice(probabilities.size, size=n_samples, replace=replace, p=probabilities)
    coords = snapshot.coords.reshape(-1, 3)[indices]
    return torch.tensor(coords, dtype=torch.float32)


def build_omega_weighted_initialization(
    snapshot,
    n_gaussians: int,
    device: str | torch.device,
    config: PlacementConfig,
) -> KernelInitialization:
    """
    Build a mixed global/local initialization using vorticity-weighted samples.

    A fraction of the kernels stays on the regular grid to preserve global
    coverage, while the remainder is concentrated in rotationally active
    regions and given a smaller initial sigma.
    """

    uniform = build_uniform_kernel_layout(n_gaussians, device)
    n_total = uniform.centers.shape[0]

    adaptive_fraction = min(max(config.adaptive_fraction, 0.0), 1.0)
    n_local = int(round(n_total * adaptive_fraction))
    n_global = n_total - n_local

    global_centers = select_spread_subset(uniform.centers, n_global)
    global_sigma = select_spread_subset(uniform.sigma, n_global) * config.global_sigma_scale

    if n_local == 0:
        return KernelInitialization(centers=global_centers, sigma=global_sigma)

    local_centers = sample_vorticity_weighted_centers(
        snapshot=snapshot,
        n_samples=n_local,
        adaptive_power=config.adaptive_power,
    ).to(device)
    local_sigma = select_spread_subset(uniform.sigma, n_local) * config.local_sigma_scale

    centers = torch.cat([global_centers, local_centers], dim=0)
    sigma = torch.cat([global_sigma, local_sigma], dim=0)
    return KernelInitialization(centers=centers, sigma=sigma)
