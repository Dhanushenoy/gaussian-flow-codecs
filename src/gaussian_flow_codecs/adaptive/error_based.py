import numpy as np
import torch

from gaussian_flow_codecs.metrics import l2_error
from gaussian_flow_codecs.models import GaussianField

from .base import KernelInitialization, PlacementConfig
from .grid import build_uniform_kernel_layout, select_spread_subset


def _sample_error_weighted_centers(
    snapshot,
    error_magnitude: np.ndarray,
    n_samples: int,
    adaptive_power: float,
) -> torch.Tensor:
    """
    Sample centers using a reconstruction-error probability map.

    The additive floor preserves global support even when the coarse probe is
    already accurate in large parts of the domain.
    """

    weights = np.power(error_magnitude + 1e-6, adaptive_power)
    probabilities = weights.reshape(-1)
    probabilities /= probabilities.sum()

    replace = n_samples > probabilities.size
    indices = np.random.choice(probabilities.size, size=n_samples, replace=replace, p=probabilities)
    coords = snapshot.coords.reshape(-1, 3)[indices]
    return torch.tensor(coords, dtype=torch.float32)


def _run_coarse_error_probe(
    snapshot,
    uniform: KernelInitialization,
    device: str | torch.device,
    config: PlacementConfig,
) -> np.ndarray:
    """
    Fit amplitudes briefly on the uniform layout to reveal high-error regions.

    This coarse probe is intentionally cheap: it trains only amplitudes and uses
    the same Gaussian blending rule as the main run.
    """

    coords_cpu = torch.tensor(snapshot.coords.reshape(-1, 3), dtype=torch.float32)
    velocity_cpu = torch.tensor(snapshot.velocity.reshape(-1, 3), dtype=torch.float32)
    n_points = coords_cpu.shape[0]
    n_kernels = uniform.centers.shape[0]

    idx = torch.randint(0, n_points, (n_kernels,))
    amp = velocity_cpu[idx].clone().to(device)
    amp.requires_grad_(True)

    field = GaussianField(
        uniform.centers,
        uniform.sigma,
        amp,
        blend_mode=config.blend_mode,
    )
    optimizer = torch.optim.Adam([amp], lr=config.error_warmup_lr)

    for _ in range(config.error_warmup_steps):
        idx = torch.randint(0, n_points, (config.error_warmup_batch_size,))
        x_batch = coords_cpu[idx].to(device, non_blocking=True)
        u_batch = velocity_cpu[idx].to(device, non_blocking=True)

        loss = l2_error(field.evaluate(x_batch), u_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    error_chunks = []
    with torch.no_grad():
        for start in range(0, n_points, config.error_eval_chunk_size):
            end = min(start + config.error_eval_chunk_size, n_points)
            coords_chunk = coords_cpu[start:end].to(device, non_blocking=True)
            u_true_chunk = velocity_cpu[start:end].to(device, non_blocking=True)
            u_pred_chunk = field.evaluate(coords_chunk)
            error_chunk = torch.linalg.norm(u_pred_chunk - u_true_chunk, dim=1)
            error_chunks.append(error_chunk.cpu())

    return torch.cat(error_chunks, dim=0).numpy().reshape(snapshot.dimensions)


def build_error_based_initialization(
    snapshot,
    n_gaussians: int,
    device: str | torch.device,
    config: PlacementConfig,
) -> KernelInitialization:
    """
    Build a mixed global/local initialization using a short coarse-fit error map.

    The strategy first probes the field with a cheap uniform model, then
    allocates the local kernel budget to regions with large reconstruction
    error. This keeps the kernel count fixed while making the allocation more
    data-aware than pure vorticity sampling.
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

    error_magnitude = _run_coarse_error_probe(snapshot, uniform, device, config)
    local_centers = _sample_error_weighted_centers(
        snapshot=snapshot,
        error_magnitude=error_magnitude,
        n_samples=n_local,
        adaptive_power=config.adaptive_power,
    ).to(device)
    local_sigma = select_spread_subset(uniform.sigma, n_local) * config.local_sigma_scale

    centers = torch.cat([global_centers, local_centers], dim=0)
    sigma = torch.cat([global_sigma, local_sigma], dim=0)
    return KernelInitialization(centers=centers, sigma=sigma)
