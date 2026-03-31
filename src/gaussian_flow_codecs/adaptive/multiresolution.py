import torch

from .base import KernelInitialization, PlacementConfig
from .grid import build_uniform_kernel_layout


def build_multiresolution_initialization(
    snapshot,
    n_gaussians: int,
    device: str | torch.device,
    config: PlacementConfig,
) -> KernelInitialization:
    """
    Build a two-level kernel set with coarse and fine uniform coverage.

    Both levels span the full domain, but they use different kernel widths and
    different kernel counts. This is a simple multi-scale baseline that matches
    the intuition that turbulent fields contain both broad structure and sharp
    localized detail.
    """

    del snapshot  # The initial multi-resolution strategy uses domain coverage only.

    base = build_uniform_kernel_layout(n_gaussians, device)
    n_total = base.centers.shape[0]

    fine_fraction = min(max(config.multires_fine_fraction, 0.0), 1.0)
    n_fine = int(round(n_total * fine_fraction))
    n_coarse = n_total - n_fine

    parts_centers = []
    parts_sigma = []

    if n_coarse > 0:
        coarse = build_uniform_kernel_layout(n_coarse, device)
        parts_centers.append(coarse.centers)
        parts_sigma.append(coarse.sigma * config.multires_coarse_sigma_scale)

    if n_fine > 0:
        fine = build_uniform_kernel_layout(n_fine, device)
        parts_centers.append(fine.centers)
        parts_sigma.append(fine.sigma * config.multires_fine_sigma_scale)

    centers = torch.cat(parts_centers, dim=0)
    sigma = torch.cat(parts_sigma, dim=0)
    return KernelInitialization(centers=centers, sigma=sigma)


def apply_multiresolution_overlay(
    kernel_init: KernelInitialization,
    config: PlacementConfig,
) -> KernelInitialization:
    """
    Apply a coarse/fine sigma split on top of an existing placement.

    The incoming centers are kept unchanged. Only the per-kernel initial sigma
    is repartitioned into broader and narrower subsets.
    """

    n_total = kernel_init.centers.shape[0]
    fine_fraction = min(max(config.multires_fine_fraction, 0.0), 1.0)
    n_fine = int(round(n_total * fine_fraction))
    n_coarse = n_total - n_fine

    sigma = kernel_init.sigma.clone()
    if n_coarse > 0:
        sigma[:n_coarse] *= config.multires_coarse_sigma_scale
    if n_fine > 0:
        sigma[n_coarse:] *= config.multires_fine_sigma_scale

    return KernelInitialization(centers=kernel_init.centers, sigma=sigma)
