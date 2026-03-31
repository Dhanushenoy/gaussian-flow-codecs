from __future__ import annotations

from .common import DEFAULT_TRAINING_OPTIONS, run_with_defaults


def run(args) -> None:
    # Adaptive: redistribute part of the kernel budget toward difficult regions.
    run_with_defaults(
        args,
        blend_mode="normalized",
        covariance_mode="diagonal",
        init_mode="error_based",
        kernel_type="gaussian",
        adaptive_fraction=DEFAULT_TRAINING_OPTIONS["adaptive_fraction"],
        adaptive_power=DEFAULT_TRAINING_OPTIONS["adaptive_power"],
        global_sigma_scale=DEFAULT_TRAINING_OPTIONS["global_sigma_scale"],
        local_sigma_scale=DEFAULT_TRAINING_OPTIONS["local_sigma_scale"],
        error_warmup_steps=DEFAULT_TRAINING_OPTIONS["error_warmup_steps"],
        error_warmup_batch_size=DEFAULT_TRAINING_OPTIONS["error_warmup_batch_size"],
        error_eval_chunk_size=DEFAULT_TRAINING_OPTIONS["error_eval_chunk_size"],
        error_warmup_lr=DEFAULT_TRAINING_OPTIONS["error_warmup_lr"],
        multires_fine_fraction=DEFAULT_TRAINING_OPTIONS["multires_fine_fraction"],
        multires_coarse_sigma_scale=DEFAULT_TRAINING_OPTIONS["multires_coarse_sigma_scale"],
        multires_fine_sigma_scale=DEFAULT_TRAINING_OPTIONS["multires_fine_sigma_scale"],
        use_multires_overlay=False,
        beta_shape=DEFAULT_TRAINING_OPTIONS["beta_shape"],
        beta_shape_lr_scale=DEFAULT_TRAINING_OPTIONS["beta_shape_lr_scale"],
        train_beta_shape=False,
    )
