from __future__ import annotations

from pathlib import Path

from gaussian_flow_codecs.training.fit_snapshot import main as fit_snapshot_main


DEFAULT_TRAINING_OPTIONS = {
    "batch_size": 100000,
    "steps": 2000,
    "lr": 1e-2,
    "eval_chunk_size": 50000,
    "physics_loss_mode": "none",
    "physics_loss_weight": 1e-4,
    "physics_cube_size": 24,
    "physics_start_fraction": 0.9,
    "train_mu": True,
    "mu_lr_scale": 0.02,
    "mu_reg_weight": 1e-6,
    "train_sigma": True,
    "sigma_lr_scale": 0.1,
    "min_sigma": 1e-3,
    "max_sigma": 0.5,
    "sigma_reg_weight": 1e-6,
    "adaptive_fraction": 0.5,
    "adaptive_power": 1.0,
    "global_sigma_scale": 1.0,
    "local_sigma_scale": 0.5,
    "error_warmup_steps": 100,
    "error_warmup_batch_size": 10000,
    "error_eval_chunk_size": 10000,
    "error_warmup_lr": 1e-2,
    "multires_fine_fraction": 0.5,
    "multires_coarse_sigma_scale": 1.5,
    "multires_fine_sigma_scale": 0.5,
    "beta_shape": 3.0,
    "beta_shape_lr_scale": 0.02,
    "train_beta_shape": False,
}

PRESET_OVERRIDES = {
    "fast": {
        "steps": 800,
        "batch_size": 50000,
        "eval_chunk_size": 50000,
    },
    "default": {},
    "hq": {
        "steps": 4000,
        "batch_size": 100000,
        "eval_chunk_size": 50000,
    },
}


def build_shared_kwargs(args) -> dict:
    preset_name = getattr(args, "preset", "default")
    if preset_name not in PRESET_OVERRIDES:
        raise ValueError(f"Unknown preset: {preset_name}")

    training_options = dict(DEFAULT_TRAINING_OPTIONS)
    training_options.update(PRESET_OVERRIDES[preset_name])

    return {
        "vtk_file": Path(args.vtk_file),
        "results_root": Path(args.results_root),
        "n_gaussians": args.n_gaussians,
        "batch_size": training_options["batch_size"],
        "steps": training_options["steps"],
        "lr": training_options["lr"],
        "eval_chunk_size": training_options["eval_chunk_size"],
        "physics_loss_mode": training_options["physics_loss_mode"],
        "physics_loss_weight": training_options["physics_loss_weight"],
        "physics_cube_size": training_options["physics_cube_size"],
        "physics_start_fraction": training_options["physics_start_fraction"],
        "train_mu": training_options["train_mu"],
        "mu_lr_scale": training_options["mu_lr_scale"],
        "mu_reg_weight": training_options["mu_reg_weight"],
        "train_sigma": training_options["train_sigma"],
        "sigma_lr_scale": training_options["sigma_lr_scale"],
        "min_sigma": training_options["min_sigma"],
        "max_sigma": training_options["max_sigma"],
        "sigma_reg_weight": training_options["sigma_reg_weight"],
        "save_vtk_outputs": args.save_vtk_outputs,
        "save_fast_outputs": args.save_fast_outputs,
    }


def run_with_defaults(args, **method_kwargs) -> None:
    kwargs = build_shared_kwargs(args)
    kwargs.update(method_kwargs)
    fit_snapshot_main(**kwargs)
