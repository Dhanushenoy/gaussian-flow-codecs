import json

import matplotlib.pyplot as plt
import numpy as np


def _extract_center_slices(field):
    nx, ny, nz = field.shape
    return {
        "xy": field[:, :, nz // 2],
        "xz": field[:, ny // 2, :],
        "yz": field[nx // 2, :, :],
    }


def plot_field_slices(true_field, pred_field, output_path, title):
    error_field = np.abs(pred_field - true_field)

    true_slices = _extract_center_slices(true_field)
    pred_slices = _extract_center_slices(pred_field)
    error_slices = _extract_center_slices(error_field)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    planes = ["xy", "xz", "yz"]
    rows = [
        ("True", true_slices),
        ("Pred", pred_slices),
        ("Abs Error", error_slices),
    ]

    shared_min = min(np.min(true_field), np.min(pred_field))
    shared_max = max(np.max(true_field), np.max(pred_field))
    error_max = np.max(error_field)

    for row_idx, (row_name, row_slices) in enumerate(rows):
        for col_idx, plane in enumerate(planes):
            ax = axes[row_idx, col_idx]
            image = row_slices[plane]
            if row_name == "Abs Error":
                im = ax.imshow(image.T, origin="lower", cmap="magma", vmin=0.0, vmax=error_max)
            else:
                im = ax.imshow(
                    image.T,
                    origin="lower",
                    cmap="viridis",
                    vmin=shared_min,
                    vmax=shared_max,
                )
            ax.set_title(f"{row_name} {plane.upper()}")
            ax.set_xlabel("i")
            ax.set_ylabel("j")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_scalar_field_comparison(true_field, pred_field, output_path, title):
    """
    Plot a scalar structured field as True / Pred / Abs Error.

    For effectively 2D data (nz == 1), the XY plane is shown. For 3D scalar
    fields, the center XY slice is shown for consistency with the CFD plots.
    """

    if true_field.ndim != 3 or pred_field.ndim != 3:
        raise ValueError("Expected scalar fields with shape (nx, ny, nz).")

    true_slice = true_field[:, :, true_field.shape[2] // 2]
    pred_slice = pred_field[:, :, pred_field.shape[2] // 2]
    error_slice = np.abs(pred_slice - true_slice)

    shared_min = min(np.min(true_slice), np.min(pred_slice))
    shared_max = max(np.max(true_slice), np.max(pred_slice))
    error_max = np.max(error_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        ("True", true_slice, "viridis", shared_min, shared_max),
        ("Pred", pred_slice, "viridis", shared_min, shared_max),
        ("Abs Error", error_slice, "magma", 0.0, error_max),
    ]

    for ax, (name, image, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_distribution_comparison(true_values, pred_values, output_path, title, x_label):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    ax.hist(true_values.ravel(), bins=120, alpha=0.6, density=True, label="true")
    ax.hist(pred_values.ravel(), bins=120, alpha=0.6, density=True, label="pred")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("density")
    ax.legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(loss_history, output_path, data_loss_history=None, physics_loss_history=None):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(loss_history, linewidth=1.5, label="total")
    if data_loss_history is not None:
        ax.plot(data_loss_history, linewidth=1.2, label="data")
    if physics_loss_history is not None:
        ax.plot(physics_loss_history, linewidth=1.2, label="physics")
    ax.set_title("Training Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    if data_loss_history is not None or physics_loss_history is not None:
        ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_metrics_summary(metrics, output_path):
    with open(output_path, "w", encoding="ascii") as f:
        json.dump(metrics, f, indent=2)
