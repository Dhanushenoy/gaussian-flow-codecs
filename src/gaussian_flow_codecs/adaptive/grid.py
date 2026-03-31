import math

import torch

from .base import KernelInitialization


def _choose_grid_shape(n_gaussians: int) -> tuple[int, int, int]:
    """
    Choose a near-cubic structured grid whose point count is at least the
    requested kernel budget.

    The previous implementation rounded the cube root and therefore silently
    changed the budget (for example, requesting 2048 kernels yielded 2197). The
    search below preserves the requested budget exactly after trimming while
    keeping the grid as isotropic as practical.
    """

    if n_gaussians <= 0:
        raise ValueError("n_gaussians must be positive.")

    n_side = math.ceil(n_gaussians ** (1.0 / 3.0))
    best_shape = None
    best_score = None

    for nx in range(max(1, n_side - 2), n_side + 3):
        for ny in range(max(1, n_side - 2), n_side + 3):
            nz = math.ceil(n_gaussians / float(nx * ny))
            total = nx * ny * nz
            if total < n_gaussians:
                continue

            excess = total - n_gaussians
            anisotropy = max(nx, ny, nz) - min(nx, ny, nz)
            score = (excess, anisotropy, total)
            if best_score is None or score < best_score:
                best_score = score
                best_shape = (nx, ny, nz)

    if best_shape is None:
        raise RuntimeError("Failed to build a valid kernel grid shape.")
    return best_shape


def select_spread_subset(tensor: torch.Tensor, n_keep: int) -> torch.Tensor:
    """
    Keep `n_keep` entries from the first dimension while preserving broad
    coverage over the original ordering.

    This is used by adaptive strategies when splitting a uniform kernel layout
    into global and local subsets. Taking the first `n_keep` points directly can
    bias the retained coverage to one corner because meshgrid flattening is
    ordered lexicographically.
    """

    total = tensor.shape[0]
    if n_keep < 0 or n_keep > total:
        raise ValueError("n_keep must satisfy 0 <= n_keep <= tensor.shape[0].")
    if n_keep == 0:
        return tensor[:0]
    if n_keep == total:
        return tensor

    keep = torch.linspace(0, total - 1, n_keep, device=tensor.device)
    keep = torch.round(keep).to(torch.long)
    keep = torch.unique_consecutive(keep)

    # Very rare edge case from rounding collisions: pad with the missing tail.
    if keep.numel() < n_keep:
        mask = torch.ones(total, dtype=torch.bool, device=tensor.device)
        mask[keep] = False
        remaining = torch.nonzero(mask, as_tuple=False).squeeze(1)
        needed = n_keep - keep.numel()
        keep = torch.cat([keep, remaining[:needed]], dim=0)
        keep, _ = torch.sort(keep)

    return tensor[keep]


def build_uniform_kernel_layout(n_gaussians: int, device: str | torch.device) -> KernelInitialization:
    """
    Place kernels on a regular grid covering the full normalized domain.

    This is the baseline allocation used directly by the grid strategy and as
    the global subset for adaptive strategies.
    """

    nx, ny, nz = _choose_grid_shape(n_gaussians)
    x = torch.linspace(0.0, 1.0, nx, device=device)
    y = torch.linspace(0.0, 1.0, ny, device=device)
    z = torch.linspace(0.0, 1.0, nz, device=device)
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")
    centers = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], dim=1)

    if centers.shape[0] > n_gaussians:
        # Keep an even spatial spread when the covering grid slightly exceeds
        # the requested budget.
        centers = select_spread_subset(centers, n_gaussians)

    sigma_value = torch.tensor(
        [
            1.0 / max(nx, 1),
            1.0 / max(ny, 1),
            1.0 / max(nz, 1),
        ],
        dtype=torch.float32,
        device=device,
    )
    sigma = sigma_value.unsqueeze(0).repeat(centers.shape[0], 1)
    return KernelInitialization(centers=centers, sigma=sigma)
