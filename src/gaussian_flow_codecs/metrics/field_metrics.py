import torch
import numpy as np


def l2_error(u_pred, u_true):
    return torch.mean((u_pred - u_true) ** 2)


def divergence(u, coords, spacing):
    """
    simple finite difference divergence (approx)
    u: (Nx,Ny,Nz,3)
    """
    dx, dy, dz = spacing

    dudx = (u[2:, 1:-1, 1:-1, 0] - u[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
    dvdy = (u[1:-1, 2:, 1:-1, 1] - u[1:-1, :-2, 1:-1, 1]) / (2 * dy)
    dwdz = (u[1:-1, 1:-1, 2:, 2] - u[1:-1, 1:-1, :-2, 2]) / (2 * dz)

    div = dudx + dvdy + dwdz
    return torch.mean(div**2)


def compute_vorticity_np(velocity, spacing):
    dx, dy, dz = spacing

    du_dx = np.gradient(velocity[..., 0], dx, axis=0)
    du_dy = np.gradient(velocity[..., 0], dy, axis=1)
    du_dz = np.gradient(velocity[..., 0], dz, axis=2)

    dv_dx = np.gradient(velocity[..., 1], dx, axis=0)
    dv_dy = np.gradient(velocity[..., 1], dy, axis=1)
    dv_dz = np.gradient(velocity[..., 1], dz, axis=2)

    dw_dx = np.gradient(velocity[..., 2], dx, axis=0)
    dw_dy = np.gradient(velocity[..., 2], dy, axis=1)
    dw_dz = np.gradient(velocity[..., 2], dz, axis=2)

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy

    return np.stack([omega_x, omega_y, omega_z], axis=-1)


def compute_vorticity_torch(velocity, spacing):
    dx, dy, dz = spacing

    du_dy = (velocity[1:-1, 2:, 1:-1, 0] - velocity[1:-1, :-2, 1:-1, 0]) / (2 * dy)
    du_dz = (velocity[1:-1, 1:-1, 2:, 0] - velocity[1:-1, 1:-1, :-2, 0]) / (2 * dz)

    dv_dx = (velocity[2:, 1:-1, 1:-1, 1] - velocity[:-2, 1:-1, 1:-1, 1]) / (2 * dx)
    dv_dz = (velocity[1:-1, 1:-1, 2:, 1] - velocity[1:-1, 1:-1, :-2, 1]) / (2 * dz)

    dw_dx = (velocity[2:, 1:-1, 1:-1, 2] - velocity[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
    dw_dy = (velocity[1:-1, 2:, 1:-1, 2] - velocity[1:-1, :-2, 1:-1, 2]) / (2 * dy)

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy

    return torch.stack([omega_x, omega_y, omega_z], dim=-1)


def compute_enstrophy_np(vorticity):
    return 0.5 * np.mean(np.sum(vorticity**2, axis=-1))


def compute_enstrophy_torch(vorticity):
    return 0.5 * torch.mean(torch.sum(vorticity**2, dim=-1))


def normalized_mse(pred, true, eps=1e-12):
    numerator = torch.mean((pred - true) ** 2)
    denominator = torch.mean(true**2) + eps
    return numerator / denominator


def compute_relative_error(pred_value, true_value, eps=1e-12):
    return abs(pred_value - true_value) / max(abs(true_value), eps)


def compute_scalar_gradient_torch(field, spacing):
    """
    Finite-difference gradient of a scalar structured field.

    Parameters
    ----------
    field:
        Scalar field with shape (nx, ny, nz).
    spacing:
        Tuple of grid spacings.

    Returns
    -------
    torch.Tensor
        Gradient components on the valid interior. For effectively 2D data
        (nz == 1), the output shape is (nx-2, ny-2, 2). For 3D data the output
        shape is (nx-2, ny-2, nz-2, 3).
    """

    dx, dy, dz = spacing

    if field.ndim != 3:
        raise ValueError("Expected scalar field with shape (nx, ny, nz).")

    if field.shape[2] == 1:
        plane = field[:, :, 0]
        dphi_dx = (plane[2:, 1:-1] - plane[:-2, 1:-1]) / (2 * dx)
        dphi_dy = (plane[1:-1, 2:] - plane[1:-1, :-2]) / (2 * dy)
        return torch.stack([dphi_dx, dphi_dy], dim=-1)

    dphi_dx = (field[2:, 1:-1, 1:-1] - field[:-2, 1:-1, 1:-1]) / (2 * dx)
    dphi_dy = (field[1:-1, 2:, 1:-1] - field[1:-1, :-2, 1:-1]) / (2 * dy)
    dphi_dz = (field[1:-1, 1:-1, 2:] - field[1:-1, 1:-1, :-2]) / (2 * dz)
    return torch.stack([dphi_dx, dphi_dy, dphi_dz], dim=-1)
