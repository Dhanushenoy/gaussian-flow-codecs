"""Single-snapshot Gaussian fitting workflow.

This module is the main public training entry point for the first release of
gaussian-flow-codecs. It fits a compact Gaussian field to one structured VTK
velocity snapshot and writes both diagnostics and reconstruction artifacts.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from gaussian_flow_codecs.adaptive import PlacementConfig, build_kernel_initialization
from gaussian_flow_codecs.metrics import (
    compute_enstrophy_np,
    compute_enstrophy_torch,
    compute_relative_error,
    compute_vorticity_np,
    compute_vorticity_torch,
    l2_error,
    normalized_mse,
)
from gaussian_flow_codecs.models import GaussianField
from gaussian_flow_codecs.vtk_io import (
    read_and_normalize_vtk,
    write_comparison_vtk,
    write_fast_vector_bundle,
    write_gaussian_kernels,
    write_normalized_vtk,
)
from gaussian_flow_codecs.visualization import (
    plot_distribution_comparison,
    plot_field_slices,
    plot_loss_curve,
    write_metrics_summary,
)


VTK_FILE = Path("example.vtk")
RESULTS_ROOT = Path("results")
N_GAUSSIANS = 512
BATCH_SIZE = 100000
STEPS = 2000
LR = 1e-2
EVAL_CHUNK_SIZE = 50000
PHYSICS_LOSS_MODE = "vorticity"
PHYSICS_LOSS_WEIGHT = 1e-4
PHYSICS_CUBE_SIZE = 24
PHYSICS_START_FRACTION = 0.9
TRAIN_MU = True
MU_LR_SCALE = 0.02
MU_REG_WEIGHT = 1e-6
TRAIN_SIGMA = True
SIGMA_LR_SCALE = 0.1
MIN_SIGMA = 1e-3
MAX_SIGMA = 0.5
SIGMA_REG_WEIGHT = 1e-6
SAVE_VTK_OUTPUTS = True
SAVE_FAST_OUTPUTS = True
BLEND_MODE = "normalized"
COVARIANCE_MODE = "diagonal"
INIT_MODE = "adaptive"
ADAPTIVE_FRACTION = 0.5
ADAPTIVE_POWER = 1.0
GLOBAL_SIGMA_SCALE = 1.0
LOCAL_SIGMA_SCALE = 0.5
ERROR_WARMUP_STEPS = 100
ERROR_WARMUP_BATCH_SIZE = 10000
ERROR_EVAL_CHUNK_SIZE = 10000
ERROR_WARMUP_LR = 1e-2
MULTIRES_FINE_FRACTION = 0.5
MULTIRES_COARSE_SIGMA_SCALE = 1.5
MULTIRES_FINE_SIGMA_SCALE = 0.5
USE_MULTIRES_OVERLAY = False
KERNEL_TYPE = "gaussian"
BETA_SHAPE = 6.0
BETA_SHAPE_LR_SCALE = 0.05
TRAIN_BETA_SHAPE = False


def diagonal_sigma_to_raw_covariance(sigma, min_sigma):
    raw = torch.zeros((sigma.shape[0], 6), dtype=sigma.dtype, device=sigma.device)
    sigma_shifted = torch.clamp(sigma - min_sigma, min=1e-6)
    raw[:, 0] = inverse_softplus(sigma_shifted[:, 0])
    raw[:, 2] = inverse_softplus(sigma_shifted[:, 1])
    raw[:, 5] = inverse_softplus(sigma_shifted[:, 2])
    return raw


def sample_cube_start(dimensions, cube_size):
    starts = []
    for dim in dimensions:
        max_start = dim - cube_size
        starts.append(torch.randint(0, max_start + 1, (1,)).item())
    return starts


def sample_structured_cube(coords_grid, velocity_grid, cube_size):
    start_i, start_j, start_k = sample_cube_start(coords_grid.shape[:3], cube_size)
    cube_slice = (
        slice(start_i, start_i + cube_size),
        slice(start_j, start_j + cube_size),
        slice(start_k, start_k + cube_size),
    )
    return coords_grid[cube_slice], velocity_grid[cube_slice]


def compute_physics_loss(field, coords_cube, velocity_cube, spacing, physics_loss_mode):
    pred_cube = field.evaluate(coords_cube.reshape(-1, 3)).reshape_as(velocity_cube)
    true_vorticity = compute_vorticity_torch(velocity_cube, spacing)
    pred_vorticity = compute_vorticity_torch(pred_cube, spacing)

    if physics_loss_mode == "vorticity":
        return normalized_mse(pred_vorticity, true_vorticity), pred_cube

    if physics_loss_mode == "enstrophy":
        true_enstrophy = compute_enstrophy_torch(true_vorticity)
        pred_enstrophy = compute_enstrophy_torch(pred_vorticity)
        loss = ((pred_enstrophy - true_enstrophy) ** 2) / (true_enstrophy**2 + 1e-12)
        return loss, pred_cube

    raise ValueError(f"Unsupported physics loss mode: {physics_loss_mode}")


def inverse_softplus(x):
    return torch.log(torch.expm1(x))


def inverse_beta_softplus(beta_shape, min_beta_shape):
    shifted = torch.clamp(beta_shape - min_beta_shape, min=1e-6)
    return inverse_softplus(shifted)


def inverse_sigmoid(x):
    eps = 1e-6
    x = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def main(
    vtk_file=VTK_FILE,
    results_root=RESULTS_ROOT,
    n_gaussians=N_GAUSSIANS,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    lr=LR,
    eval_chunk_size=EVAL_CHUNK_SIZE,
    physics_loss_mode=PHYSICS_LOSS_MODE,
    physics_loss_weight=PHYSICS_LOSS_WEIGHT,
    physics_cube_size=PHYSICS_CUBE_SIZE,
    physics_start_fraction=PHYSICS_START_FRACTION,
    train_mu=TRAIN_MU,
    mu_lr_scale=MU_LR_SCALE,
    mu_reg_weight=MU_REG_WEIGHT,
    train_sigma=TRAIN_SIGMA,
    sigma_lr_scale=SIGMA_LR_SCALE,
    min_sigma=MIN_SIGMA,
    max_sigma=MAX_SIGMA,
    sigma_reg_weight=SIGMA_REG_WEIGHT,
    save_vtk_outputs=SAVE_VTK_OUTPUTS,
    save_fast_outputs=SAVE_FAST_OUTPUTS,
    blend_mode=BLEND_MODE,
    covariance_mode=COVARIANCE_MODE,
    init_mode=INIT_MODE,
    adaptive_fraction=ADAPTIVE_FRACTION,
    adaptive_power=ADAPTIVE_POWER,
    global_sigma_scale=GLOBAL_SIGMA_SCALE,
    local_sigma_scale=LOCAL_SIGMA_SCALE,
    error_warmup_steps=ERROR_WARMUP_STEPS,
    error_warmup_batch_size=ERROR_WARMUP_BATCH_SIZE,
    error_eval_chunk_size=ERROR_EVAL_CHUNK_SIZE,
    error_warmup_lr=ERROR_WARMUP_LR,
    multires_fine_fraction=MULTIRES_FINE_FRACTION,
    multires_coarse_sigma_scale=MULTIRES_COARSE_SIGMA_SCALE,
    multires_fine_sigma_scale=MULTIRES_FINE_SIGMA_SCALE,
    use_multires_overlay=USE_MULTIRES_OVERLAY,
    kernel_type=KERNEL_TYPE,
    beta_shape=BETA_SHAPE,
    beta_shape_lr_scale=BETA_SHAPE_LR_SCALE,
    train_beta_shape=TRAIN_BETA_SHAPE,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vtk_file = Path(vtk_file)
    results_root = Path(results_root)
    if physics_cube_size < 3:
        raise ValueError("physics_cube_size must be at least 3 for finite differences.")
    if not 0.0 <= physics_start_fraction <= 1.0:
        raise ValueError("physics_start_fraction must be in [0, 1].")
    if not 0.0 <= adaptive_fraction <= 1.0:
        raise ValueError("adaptive_fraction must be in [0, 1].")
    if covariance_mode not in {"diagonal", "full"}:
        raise ValueError("covariance_mode must be 'diagonal' or 'full'.")
    case_name = vtk_file.stem
    results_dir = results_root / case_name
    normalized_vtk_file = results_dir / f"{case_name}_normalized.vtk"
    comparison_vtk_file = results_dir / f"{case_name}_comparison.vtk"
    kernels_vtk_file = results_dir / f"{case_name}_gaussian_kernels.vtp"
    fast_bundle_dir = results_dir / "fast_arrays"
    metrics_file = results_dir / "diagnostics.json"
    loss_plot_file = results_dir / "loss_curve.png"
    velocity_slice_plot = results_dir / "velocity_magnitude_slices.png"
    vorticity_slice_plot = results_dir / "vorticity_magnitude_slices.png"
    velocity_hist_plot = results_dir / "velocity_magnitude_histogram.png"
    vorticity_hist_plot = results_dir / "vorticity_magnitude_histogram.png"

    # Keep all artifacts for one input snapshot grouped in a single case folder.
    results_dir.mkdir(parents=True, exist_ok=True)

    snapshot = read_and_normalize_vtk(vtk_file)
    if save_vtk_outputs:
        write_normalized_vtk(snapshot, normalized_vtk_file)

    coords_np = snapshot.coords
    u_np = snapshot.velocity
    spacing = tuple(1.0 / (dim - 1) for dim in snapshot.dimensions)

    coords_cpu = torch.tensor(coords_np.reshape(-1, 3), dtype=torch.float32)
    u_true_cpu = torch.tensor(u_np.reshape(-1, 3), dtype=torch.float32)
    coords_grid_cpu = coords_cpu.view(*snapshot.dimensions, 3)
    u_true_grid_cpu = u_true_cpu.view(*snapshot.dimensions, 3)

    n_points = coords_cpu.shape[0]
    loss_history = []
    data_loss_history = []
    physics_loss_history = []

    # Kernel placement is configurable so the same training routine supports
    # the baseline grid initialization as well as the adaptive variants.
    placement_config = PlacementConfig(
        init_mode=init_mode,
        adaptive_fraction=adaptive_fraction,
        adaptive_power=adaptive_power,
        global_sigma_scale=global_sigma_scale,
        local_sigma_scale=local_sigma_scale,
        blend_mode=blend_mode,
        error_warmup_steps=error_warmup_steps,
        error_warmup_batch_size=error_warmup_batch_size,
        error_eval_chunk_size=error_eval_chunk_size,
        error_warmup_lr=error_warmup_lr,
        multires_fine_fraction=multires_fine_fraction,
        multires_coarse_sigma_scale=multires_coarse_sigma_scale,
        multires_fine_sigma_scale=multires_fine_sigma_scale,
        use_multires_overlay=use_multires_overlay,
    )
    kernel_init = build_kernel_initialization(
        snapshot=snapshot,
        n_gaussians=n_gaussians,
        device=device,
        config=placement_config,
    )
    mu, sigma = kernel_init.centers, kernel_init.sigma
    initial_mu = mu.detach().clone()
    raw_mu = None

    if train_mu:
        raw_mu = inverse_sigmoid(initial_mu).detach().clone()
        raw_mu.requires_grad_(True)

    n_gaussians = mu.shape[0]
    initial_sigma = sigma.detach().clone()
    raw_sigma = None
    raw_covariance = None
    raw_beta_shape = None

    if train_sigma and covariance_mode == "diagonal":
        init_sigma_shifted = torch.clamp(initial_sigma - min_sigma, min=1e-6)
        raw_sigma = inverse_softplus(init_sigma_shifted).detach().clone()
        raw_sigma.requires_grad_(True)
    elif train_sigma and covariance_mode == "full":
        raw_covariance = diagonal_sigma_to_raw_covariance(initial_sigma, min_sigma).detach().clone()
        raw_covariance.requires_grad_(True)

    if kernel_type == "beta" and train_beta_shape:
        beta_init = torch.full_like(initial_sigma, beta_shape)
        raw_beta_shape = inverse_beta_softplus(beta_init, 1.01).detach().clone()
        raw_beta_shape.requires_grad_(True)

    idx = torch.randint(0, n_points, (n_gaussians,))
    amp = u_true_cpu[idx].clone().to(device)
    amp.requires_grad_(True)

    field = GaussianField(
        mu,
        sigma,
        amp,
        raw_mu=raw_mu,
        raw_sigma=raw_sigma,
        raw_covariance=raw_covariance,
        raw_beta_shape=raw_beta_shape,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        blend_mode=blend_mode,
        covariance_mode=covariance_mode,
        kernel_type=kernel_type,
        beta_shape=beta_shape,
    )
    optim_params = [{"params": [amp], "lr": lr}]
    if train_mu:
        optim_params.append({"params": [raw_mu], "lr": lr * mu_lr_scale})
    if train_sigma and covariance_mode == "diagonal":
        optim_params.append({"params": [raw_sigma], "lr": lr * sigma_lr_scale})
    if train_sigma and covariance_mode == "full":
        optim_params.append({"params": [raw_covariance], "lr": lr * sigma_lr_scale})
    if raw_beta_shape is not None:
        optim_params.append({"params": [raw_beta_shape], "lr": lr * beta_shape_lr_scale})
    optimizer = torch.optim.Adam(optim_params)
    physics_start_step = int(steps * physics_start_fraction)
    best_amp = amp.detach().clone()
    best_mu = field.mu.detach().clone()
    best_shape_state = field.get_shape_state()
    best_data_loss = float("inf")
    best_step = -1

    for step in range(steps):
        idx = torch.randint(0, n_points, (batch_size,))
        x_batch = coords_cpu[idx].to(device, non_blocking=True)
        u_batch = u_true_cpu[idx].to(device, non_blocking=True)

        u_pred = field.evaluate(x_batch)
        data_loss = l2_error(u_pred, u_batch)
        physics_loss = torch.tensor(0.0, device=device)
        current_physics_weight = 0.0

        if physics_loss_mode != "none" and step >= physics_start_step:
            ramp_denominator = max(steps - physics_start_step, 1)
            current_physics_weight = physics_loss_weight * (
                (step - physics_start_step + 1) / ramp_denominator
            )
            coords_cube, velocity_cube = sample_structured_cube(
                coords_grid_cpu,
                u_true_grid_cpu,
                physics_cube_size,
            )
            coords_cube = coords_cube.to(device, non_blocking=True)
            velocity_cube = velocity_cube.to(device, non_blocking=True)
            physics_loss, _ = compute_physics_loss(
                field,
                coords_cube,
                velocity_cube,
                spacing,
                physics_loss_mode,
            )

        sigma_reg_loss = torch.tensor(0.0, device=device)
        if train_sigma and sigma_reg_weight > 0.0:
            sigma_reg_loss = normalized_mse(field.sigma, initial_sigma)
        mu_reg_loss = torch.tensor(0.0, device=device)
        if train_mu and mu_reg_weight > 0.0:
            mu_reg_loss = normalized_mse(field.mu, initial_mu)

        loss = (
            data_loss
            + current_physics_weight * physics_loss
            + sigma_reg_weight * sigma_reg_loss
            + mu_reg_weight * mu_reg_loss
        )
        loss_history.append(loss.item())
        data_loss_history.append(data_loss.item())
        physics_loss_history.append(physics_loss.item())

        if data_loss.item() < best_data_loss:
            best_data_loss = data_loss.item()
            best_amp = amp.detach().clone()
            best_mu = field.mu.detach().clone()
            best_shape_state = field.get_shape_state()
            best_step = step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"Step {step}, Loss {loss.item()}, "
                f"Data {data_loss.item()}, Physics {physics_loss.item()}, "
                f"PhysicsWeight {current_physics_weight}, "
                f"MuReg {mu_reg_loss.item()}, SigmaReg {sigma_reg_loss.item()}"
            )

    with torch.no_grad():
        amp.copy_(best_amp)
        if train_mu:
            raw_mu.copy_(inverse_sigmoid(best_mu))
        if train_sigma:
            field.load_shape_state_(best_shape_state)

    with torch.no_grad():
        loss_sum = 0.0
        u_pred_full = torch.empty((n_points, 3), dtype=torch.float32)

        for start in range(0, n_points, eval_chunk_size):
            end = min(start + eval_chunk_size, n_points)
            coords_chunk = coords_cpu[start:end].to(device, non_blocking=True)
            u_true_chunk = u_true_cpu[start:end].to(device, non_blocking=True)
            u_pred_chunk = field.evaluate(coords_chunk)
            chunk_loss = l2_error(u_pred_chunk, u_true_chunk)
            loss_sum += chunk_loss.item() * (end - start)
            u_pred_full[start:end] = u_pred_chunk.detach().cpu()

        final_loss = loss_sum / n_points

    u_pred_np = u_pred_full.numpy().reshape(snapshot.dimensions + (3,))
    velocity_mag_true = np.linalg.norm(snapshot.velocity, axis=-1)
    velocity_mag_pred = np.linalg.norm(u_pred_np, axis=-1)
    vorticity_true = compute_vorticity_np(snapshot.velocity, spacing)
    vorticity_pred = compute_vorticity_np(u_pred_np, spacing)
    vorticity_mag_true = np.linalg.norm(vorticity_true, axis=-1)
    vorticity_mag_pred = np.linalg.norm(vorticity_pred, axis=-1)
    enstrophy_true = compute_enstrophy_np(vorticity_true)
    enstrophy_pred = compute_enstrophy_np(vorticity_pred)
    relative_enstrophy_error = compute_relative_error(enstrophy_pred, enstrophy_true)

    if save_vtk_outputs:
        write_comparison_vtk(
            snapshot=snapshot,
            predicted_velocity=u_pred_np,
            output_path=comparison_vtk_file,
            extra_point_data={
                "vorticity_true": vorticity_true,
                "vorticity_pred": vorticity_pred,
                "vorticity_error": vorticity_pred - vorticity_true,
                "vorticity_mag_true": vorticity_mag_true,
                "vorticity_mag_pred": vorticity_mag_pred,
                "vorticity_mag_error": vorticity_mag_pred - vorticity_mag_true,
            },
        )

        write_gaussian_kernels(
            centers=field.mu.detach().cpu().numpy(),
            sigma=field.sigma.detach().cpu().numpy(),
            amplitude=amp.detach().cpu().numpy(),
            output_path=kernels_vtk_file,
            covariance=field.covariance_matrix.detach().cpu().numpy(),
            beta_shape=field.beta_shape.detach().cpu().numpy() if field.beta_shape is not None else None,
        )

    if save_fast_outputs:
        write_fast_vector_bundle(
            snapshot=snapshot,
            predicted_velocity=u_pred_np,
            output_dir=fast_bundle_dir,
            extra_arrays={
                "vorticity_true": vorticity_true,
                "vorticity_pred": vorticity_pred,
                "vorticity_error": vorticity_pred - vorticity_true,
                "vorticity_mag_true": vorticity_mag_true,
                "vorticity_mag_pred": vorticity_mag_pred,
                "vorticity_mag_error": vorticity_mag_pred - vorticity_mag_true,
            },
        )

    original_value_count = snapshot.velocity.size
    kernel_value_count = mu.numel() + field.shape_parameter_count + amp.numel()
    compression_ratio = original_value_count / kernel_value_count
    diagnostics = {
        "case_name": case_name,
        "final_l2": final_loss,
        "enstrophy_true": float(enstrophy_true),
        "enstrophy_pred": float(enstrophy_pred),
        "relative_enstrophy_error": float(relative_enstrophy_error),
        "original_values": int(original_value_count),
        "kernel_values": int(kernel_value_count),
        "compression_ratio": float(compression_ratio),
        "physics_loss_mode": physics_loss_mode,
        "physics_loss_weight": float(physics_loss_weight),
        "physics_cube_size": int(physics_cube_size),
        "physics_start_fraction": float(physics_start_fraction),
        "physics_start_step": int(physics_start_step),
        "train_mu": bool(train_mu),
        "mu_lr_scale": float(mu_lr_scale),
        "mu_reg_weight": float(mu_reg_weight),
        "mu_mean": float(field.mu.mean().item()),
        "mu_min": float(field.mu.min().item()),
        "mu_max": float(field.mu.max().item()),
        "train_sigma": bool(train_sigma),
        "sigma_lr_scale": float(sigma_lr_scale),
        "min_sigma": float(min_sigma),
        "max_sigma": float(max_sigma),
        "sigma_reg_weight": float(sigma_reg_weight),
        "sigma_mean": float(field.sigma.mean().item()),
        "sigma_min": float(field.sigma.min().item()),
        "sigma_max": float(field.sigma.max().item()),
        "final_train_loss": float(loss_history[-1]),
        "final_train_data_loss": float(data_loss_history[-1]),
        "final_train_physics_loss": float(physics_loss_history[-1]),
        "best_data_loss": float(best_data_loss),
        "best_data_loss_step": int(best_step),
        "save_vtk_outputs": bool(save_vtk_outputs),
        "save_fast_outputs": bool(save_fast_outputs),
        "blend_mode": blend_mode,
        "covariance_mode": covariance_mode,
        "init_mode": init_mode,
        "adaptive_fraction": float(adaptive_fraction),
        "adaptive_power": float(adaptive_power),
        "global_sigma_scale": float(global_sigma_scale),
        "local_sigma_scale": float(local_sigma_scale),
        "error_warmup_steps": int(error_warmup_steps),
        "error_warmup_batch_size": int(error_warmup_batch_size),
        "error_eval_chunk_size": int(error_eval_chunk_size),
        "error_warmup_lr": float(error_warmup_lr),
        "multires_fine_fraction": float(multires_fine_fraction),
        "multires_coarse_sigma_scale": float(multires_coarse_sigma_scale),
        "multires_fine_sigma_scale": float(multires_fine_sigma_scale),
        "use_multires_overlay": bool(use_multires_overlay),
        "kernel_type": kernel_type,
        "beta_shape": float(beta_shape),
        "beta_shape_mean": float(field.beta_shape.mean().item()) if field.beta_shape is not None else float(beta_shape),
        "beta_shape_min": float(field.beta_shape.min().item()) if field.beta_shape is not None else float(beta_shape),
        "beta_shape_max": float(field.beta_shape.max().item()) if field.beta_shape is not None else float(beta_shape),
        "beta_shape_lr_scale": float(beta_shape_lr_scale),
        "train_beta_shape": bool(train_beta_shape),
    }

    write_metrics_summary(diagnostics, metrics_file)
    plot_loss_curve(
        loss_history,
        loss_plot_file,
        data_loss_history=data_loss_history,
        physics_loss_history=physics_loss_history,
    )
    plot_field_slices(
        velocity_mag_true,
        velocity_mag_pred,
        velocity_slice_plot,
        title=f"{case_name}: velocity magnitude",
    )
    plot_field_slices(
        vorticity_mag_true,
        vorticity_mag_pred,
        vorticity_slice_plot,
        title=f"{case_name}: vorticity magnitude",
    )
    plot_distribution_comparison(
        velocity_mag_true,
        velocity_mag_pred,
        velocity_hist_plot,
        title=f"{case_name}: velocity magnitude distribution",
        x_label="|u|",
    )
    plot_distribution_comparison(
        vorticity_mag_true,
        vorticity_mag_pred,
        vorticity_hist_plot,
        title=f"{case_name}: vorticity magnitude distribution",
        x_label="|omega|",
    )

    print("Final L2:", final_loss)
    print("True enstrophy:", enstrophy_true)
    print("Pred enstrophy:", enstrophy_pred)
    print("Relative enstrophy error:", relative_enstrophy_error)
    if save_vtk_outputs:
        print("Saved normalized VTK:", normalized_vtk_file)
        print("Saved comparison VTK:", comparison_vtk_file)
        print("Saved kernels VTK:", kernels_vtk_file)
    if save_fast_outputs:
        print("Saved fast array bundle:", fast_bundle_dir)
    print("Saved diagnostics:", metrics_file)
    print("Saved loss plot:", loss_plot_file)
    print("Saved velocity slices:", velocity_slice_plot)
    print("Saved vorticity slices:", vorticity_slice_plot)
    print("Saved velocity histogram:", velocity_hist_plot)
    print("Saved vorticity histogram:", vorticity_hist_plot)
    print("Original values:", original_value_count)
    print("Kernel values:", kernel_value_count)
    print("Compression ratio:", compression_ratio)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtk-file", type=Path, default=VTK_FILE)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--n-gaussians", type=int, default=N_GAUSSIANS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--eval-chunk-size", type=int, default=EVAL_CHUNK_SIZE)
    parser.add_argument(
        "--physics-loss-mode",
        choices=["none", "vorticity", "enstrophy"],
        default=PHYSICS_LOSS_MODE,
    )
    parser.add_argument("--physics-loss-weight", type=float, default=PHYSICS_LOSS_WEIGHT)
    parser.add_argument("--physics-cube-size", type=int, default=PHYSICS_CUBE_SIZE)
    parser.add_argument("--physics-start-fraction", type=float, default=PHYSICS_START_FRACTION)
    parser.add_argument("--train-mu", action=argparse.BooleanOptionalAction, default=TRAIN_MU)
    parser.add_argument("--mu-lr-scale", type=float, default=MU_LR_SCALE)
    parser.add_argument("--mu-reg-weight", type=float, default=MU_REG_WEIGHT)
    parser.add_argument("--train-sigma", action=argparse.BooleanOptionalAction, default=TRAIN_SIGMA)
    parser.add_argument("--sigma-lr-scale", type=float, default=SIGMA_LR_SCALE)
    parser.add_argument("--min-sigma", type=float, default=MIN_SIGMA)
    parser.add_argument("--max-sigma", type=float, default=MAX_SIGMA)
    parser.add_argument("--sigma-reg-weight", type=float, default=SIGMA_REG_WEIGHT)
    parser.add_argument("--save-vtk-outputs", action=argparse.BooleanOptionalAction, default=SAVE_VTK_OUTPUTS)
    parser.add_argument("--save-fast-outputs", action=argparse.BooleanOptionalAction, default=SAVE_FAST_OUTPUTS)
    parser.add_argument("--blend-mode", choices=["raw", "normalized"], default=BLEND_MODE)
    parser.add_argument("--covariance-mode", choices=["diagonal", "full"], default=COVARIANCE_MODE)
    parser.add_argument("--kernel-type", choices=["gaussian", "beta"], default=KERNEL_TYPE)
    parser.add_argument("--beta-shape", type=float, default=BETA_SHAPE)
    parser.add_argument("--beta-shape-lr-scale", type=float, default=BETA_SHAPE_LR_SCALE)
    parser.add_argument("--train-beta-shape", action=argparse.BooleanOptionalAction, default=TRAIN_BETA_SHAPE)
    parser.add_argument(
        "--init-mode",
        choices=["grid", "adaptive", "error_based", "multires"],
        default=INIT_MODE,
    )
    parser.add_argument("--adaptive-fraction", type=float, default=ADAPTIVE_FRACTION)
    parser.add_argument("--adaptive-power", type=float, default=ADAPTIVE_POWER)
    parser.add_argument("--global-sigma-scale", type=float, default=GLOBAL_SIGMA_SCALE)
    parser.add_argument("--local-sigma-scale", type=float, default=LOCAL_SIGMA_SCALE)
    parser.add_argument("--error-warmup-steps", type=int, default=ERROR_WARMUP_STEPS)
    parser.add_argument("--error-warmup-batch-size", type=int, default=ERROR_WARMUP_BATCH_SIZE)
    parser.add_argument("--error-eval-chunk-size", type=int, default=ERROR_EVAL_CHUNK_SIZE)
    parser.add_argument("--error-warmup-lr", type=float, default=ERROR_WARMUP_LR)
    parser.add_argument("--multires-fine-fraction", type=float, default=MULTIRES_FINE_FRACTION)
    parser.add_argument(
        "--multires-coarse-sigma-scale",
        type=float,
        default=MULTIRES_COARSE_SIGMA_SCALE,
    )
    parser.add_argument(
        "--multires-fine-sigma-scale",
        type=float,
        default=MULTIRES_FINE_SIGMA_SCALE,
    )
    parser.add_argument(
        "--use-multires-overlay",
        action=argparse.BooleanOptionalAction,
        default=USE_MULTIRES_OVERLAY,
    )
    args = parser.parse_args()
    main(
        vtk_file=args.vtk_file,
        results_root=args.results_root,
        n_gaussians=args.n_gaussians,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        eval_chunk_size=args.eval_chunk_size,
        physics_loss_mode=args.physics_loss_mode,
        physics_loss_weight=args.physics_loss_weight,
        physics_cube_size=args.physics_cube_size,
        physics_start_fraction=args.physics_start_fraction,
        train_mu=args.train_mu,
        mu_lr_scale=args.mu_lr_scale,
        mu_reg_weight=args.mu_reg_weight,
        train_sigma=args.train_sigma,
        sigma_lr_scale=args.sigma_lr_scale,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        sigma_reg_weight=args.sigma_reg_weight,
        save_vtk_outputs=args.save_vtk_outputs,
        save_fast_outputs=args.save_fast_outputs,
        blend_mode=args.blend_mode,
        covariance_mode=args.covariance_mode,
        kernel_type=args.kernel_type,
        beta_shape=args.beta_shape,
        beta_shape_lr_scale=args.beta_shape_lr_scale,
        train_beta_shape=args.train_beta_shape,
        init_mode=args.init_mode,
        adaptive_fraction=args.adaptive_fraction,
        adaptive_power=args.adaptive_power,
        global_sigma_scale=args.global_sigma_scale,
        local_sigma_scale=args.local_sigma_scale,
        error_warmup_steps=args.error_warmup_steps,
        error_warmup_batch_size=args.error_warmup_batch_size,
        error_eval_chunk_size=args.error_eval_chunk_size,
        error_warmup_lr=args.error_warmup_lr,
        multires_fine_fraction=args.multires_fine_fraction,
        multires_coarse_sigma_scale=args.multires_coarse_sigma_scale,
        multires_fine_sigma_scale=args.multires_fine_sigma_scale,
        use_multires_overlay=args.use_multires_overlay,
    )


if __name__ == "__main__":
    cli()
