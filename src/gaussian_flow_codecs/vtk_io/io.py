import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass
class VTKSnapshot:
    file_path: str
    field_name: str
    dimensions: tuple[int, int, int]
    mesh: pv.StructuredGrid
    coords: np.ndarray
    velocity: np.ndarray
    points_min: np.ndarray
    points_max: np.ndarray


@dataclass
class ScalarVTKSnapshot:
    file_path: str
    field_name: str
    dimensions: tuple[int, int, int]
    mesh: pv.StructuredGrid
    coords: np.ndarray
    field: np.ndarray
    points_min: np.ndarray
    points_max: np.ndarray


def _detect_vector_field_name(mesh: pv.DataSet, field_name: str | None) -> str:
    if field_name is not None:
        if field_name not in mesh.point_data:
            split_components = _detect_split_vector_components(mesh, field_name)
            if split_components is not None:
                return field_name
            available = ", ".join(mesh.point_data.keys())
            raise KeyError(
                f"Velocity field not found. Requested field={field_name!r}. "
                f"Available point_data arrays: [{available}]"
            )
        return field_name

    preferred_names = ("velocity", "u", "U", "vel")
    for name in preferred_names:
        if name in mesh.point_data:
            return name

    split_candidates = ("velocity", "u", "vel")
    for prefix in split_candidates:
        if _detect_split_vector_components(mesh, prefix) is not None:
            return prefix

    for name, array in mesh.point_data.items():
        if getattr(array, "ndim", 0) == 2 and array.shape[1] == 3:
            return name

    available = ", ".join(mesh.point_data.keys())
    raise KeyError(
        "No vector field with shape (N_points, 3) was found. "
        f"Available point_data arrays: [{available}]"
    )


def _detect_split_vector_components(mesh: pv.DataSet, prefix: str) -> tuple[str, str, str] | None:
    candidates = [
        (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"),
        (f"{prefix}x", f"{prefix}y", f"{prefix}z"),
        (f"{prefix}_1", f"{prefix}_2", f"{prefix}_3"),
    ]
    for names in candidates:
        if all(name in mesh.point_data for name in names):
            return names
    return None


def _load_vector_field(mesh: pv.DataSet, field_name: str, dimensions: tuple[int, int, int]) -> np.ndarray:
    if field_name in mesh.point_data:
        return _reshape_vector_field(mesh.point_data[field_name], dimensions)

    split_components = _detect_split_vector_components(mesh, field_name)
    if split_components is None:
        available = ", ".join(mesh.point_data.keys())
        raise KeyError(
            f"Velocity field {field_name!r} was not found as a vector array or split components. "
            f"Available point_data arrays: [{available}]"
        )

    stacked = np.stack([mesh.point_data[name] for name in split_components], axis=1)
    return _reshape_vector_field(stacked, dimensions)


def _detect_scalar_field_name(mesh: pv.DataSet, field_name: str | None) -> str:
    if field_name is not None:
        if field_name not in mesh.point_data:
            available = ", ".join(mesh.point_data.keys())
            raise KeyError(
                f"Scalar field not found. Requested field={field_name!r}. "
                f"Available point_data arrays: [{available}]"
            )
        array = mesh.point_data[field_name]
        if getattr(array, "ndim", 0) != 1:
            raise ValueError(
                f"Requested field={field_name!r} has shape {getattr(array, 'shape', None)}; "
                "expected a scalar field with shape (N_points,)."
            )
        return field_name

    preferred_names = ("velocity_mag", "speed", "pressure", "density")
    for name in preferred_names:
        if name in mesh.point_data and getattr(mesh.point_data[name], "ndim", 0) == 1:
            return name

    for name, array in mesh.point_data.items():
        if getattr(array, "ndim", 0) == 1:
            return name

    available = ", ".join(mesh.point_data.keys())
    raise KeyError(
        "No scalar point-data field with shape (N_points,) was found. "
        f"Available point_data arrays: [{available}]"
    )


def _reshape_vector_field(array: np.ndarray, dimensions: tuple[int, int, int]) -> np.ndarray:
    if getattr(array, "ndim", 0) != 2 or array.shape[1] != 3:
        raise ValueError(
            f"Field has shape {getattr(array, 'shape', None)}; "
            "expected a vector field with shape (N_points, 3)."
        )

    nx, ny, nz = dimensions
    return np.asarray(array).reshape(nx, ny, nz, 3, order="F")


def _reshape_scalar_field(array: np.ndarray, dimensions: tuple[int, int, int]) -> np.ndarray:
    if getattr(array, "ndim", 0) != 1:
        raise ValueError(
            f"Field has shape {getattr(array, 'shape', None)}; "
            "expected a scalar field with shape (N_points,)."
        )

    nx, ny, nz = dimensions
    return np.asarray(array).reshape(nx, ny, nz, order="F")


def _reshape_points(points: np.ndarray, dimensions: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dimensions
    return np.asarray(points, dtype=np.float32).reshape(nx, ny, nz, 3, order="F")


def _flatten_vector_field(field: np.ndarray) -> np.ndarray:
    return np.asarray(field).reshape(-1, 3, order="F")


def _flatten_scalar_field(field: np.ndarray) -> np.ndarray:
    return np.asarray(field).reshape(-1, order="F")


def _flatten_points(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 3, order="F")


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    extent = points_max - points_min
    extent[extent == 0.0] = 1.0
    normalized = (points - points_min) / extent
    return normalized, points_min, points_max


def _make_normalized_mesh(snapshot: VTKSnapshot) -> pv.StructuredGrid:
    normalized_mesh = snapshot.mesh.copy(deep=True)
    normalized_mesh.points = _flatten_points(snapshot.coords)
    return normalized_mesh


def read_and_normalize_vtk(file_path: str, field_name: str | None = None) -> VTKSnapshot:
    mesh = pv.read(file_path)

    if not hasattr(mesh, "dimensions"):
        raise TypeError(
            f"Expected a structured grid-like dataset, got {type(mesh).__name__}."
        )

    dimensions = tuple(int(v) for v in mesh.dimensions)
    vector_field_name = _detect_vector_field_name(mesh, field_name)
    velocity = _load_vector_field(mesh, vector_field_name, dimensions)

    normalized_points, points_min, points_max = _normalize_points(mesh.points)
    coords = _reshape_points(normalized_points, dimensions)

    return VTKSnapshot(
        file_path=file_path,
        field_name=vector_field_name,
        dimensions=dimensions,
        mesh=mesh,
        coords=coords,
        velocity=velocity,
        points_min=points_min,
        points_max=points_max,
    )


def read_and_normalize_scalar_vtk(file_path: str, field_name: str | None = None) -> ScalarVTKSnapshot:
    mesh = pv.read(file_path)

    if not hasattr(mesh, "dimensions"):
        raise TypeError(
            f"Expected a structured grid-like dataset, got {type(mesh).__name__}."
        )

    dimensions = tuple(int(v) for v in mesh.dimensions)
    scalar_field_name = _detect_scalar_field_name(mesh, field_name)
    field = _reshape_scalar_field(mesh.point_data[scalar_field_name], dimensions)

    normalized_points, points_min, points_max = _normalize_points(mesh.points)
    coords = _reshape_points(normalized_points, dimensions)

    return ScalarVTKSnapshot(
        file_path=file_path,
        field_name=scalar_field_name,
        dimensions=dimensions,
        mesh=mesh,
        coords=coords,
        field=field,
        points_min=points_min,
        points_max=points_max,
    )


def write_normalized_vtk(
    snapshot: VTKSnapshot,
    output_path: str,
    normalized_field_name: str = "velocity",
) -> None:
    mesh = _make_normalized_mesh(snapshot)
    flat_velocity = _flatten_vector_field(snapshot.velocity)
    mesh.point_data[normalized_field_name] = flat_velocity
    mesh.point_data["speed"] = np.linalg.norm(flat_velocity, axis=1)
    mesh.save(output_path)


def write_normalized_scalar_vtk(
    snapshot: ScalarVTKSnapshot,
    output_path: str,
    normalized_field_name: str = "field",
) -> None:
    mesh = snapshot.mesh.copy(deep=True)
    mesh.points = _flatten_points(snapshot.coords)
    mesh.point_data[normalized_field_name] = _flatten_scalar_field(snapshot.field)
    mesh.save(output_path)


def write_comparison_vtk(
    snapshot: VTKSnapshot,
    predicted_velocity: np.ndarray,
    output_path: str,
    true_field_name: str = "velocity_true",
    pred_field_name: str = "velocity_pred",
    extra_point_data: dict[str, np.ndarray] | None = None,
) -> None:
    predicted_velocity = np.asarray(predicted_velocity)
    if predicted_velocity.shape == snapshot.dimensions + (3,):
        predicted = predicted_velocity
    else:
        predicted = _reshape_vector_field(predicted_velocity.reshape(-1, 3), snapshot.dimensions)
    truth = snapshot.velocity
    error = predicted - truth

    mesh = _make_normalized_mesh(snapshot)
    flat_truth = _flatten_vector_field(truth)
    flat_pred = _flatten_vector_field(predicted)
    flat_error = _flatten_vector_field(error)
    mesh.point_data[true_field_name] = flat_truth
    mesh.point_data[pred_field_name] = flat_pred
    mesh.point_data["velocity_error"] = flat_error
    mesh.point_data["speed_true"] = np.linalg.norm(flat_truth, axis=1)
    mesh.point_data["speed_pred"] = np.linalg.norm(flat_pred, axis=1)
    mesh.point_data["speed_error"] = mesh.point_data["speed_pred"] - mesh.point_data["speed_true"]

    if extra_point_data is not None:
        for name, array in extra_point_data.items():
            array = np.asarray(array)
            if array.ndim == 4 and array.shape[-1] == 3:
                mesh.point_data[name] = _flatten_vector_field(array)
            elif array.ndim == 3:
                mesh.point_data[name] = _flatten_scalar_field(array)
            else:
                mesh.point_data[name] = array.reshape(-1, *array.shape[3:], order="F")

    mesh.save(output_path)


def write_scalar_comparison_vtk(
    snapshot: ScalarVTKSnapshot,
    predicted_field: np.ndarray,
    output_path: str,
    true_field_name: str = "field_true",
    pred_field_name: str = "field_pred",
    use_normalized_points: bool = False,
) -> None:
    predicted_field = np.asarray(predicted_field)
    if predicted_field.shape == snapshot.dimensions:
        predicted = predicted_field
    else:
        predicted = _reshape_scalar_field(predicted_field.reshape(-1), snapshot.dimensions)
    truth = snapshot.field
    error = predicted - truth

    mesh = snapshot.mesh.copy(deep=True)
    if use_normalized_points:
        mesh.points = _flatten_points(snapshot.coords)
    mesh.point_data[true_field_name] = _flatten_scalar_field(truth)
    mesh.point_data[pred_field_name] = _flatten_scalar_field(predicted)
    mesh.point_data["field_error"] = _flatten_scalar_field(error)
    mesh.save(output_path)


def write_gaussian_kernels(
    centers: np.ndarray,
    sigma: np.ndarray,
    amplitude: np.ndarray,
    output_path: str,
    covariance: np.ndarray | None = None,
    beta_shape: np.ndarray | None = None,
) -> None:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(-1, 3)
    amplitude = np.asarray(amplitude, dtype=np.float32).reshape(-1, 3)

    if not (len(centers) == len(sigma) == len(amplitude)):
        raise ValueError("centers, sigma, and amplitude must have the same length.")

    kernels = pv.PolyData(centers)
    kernels.point_data["sigma"] = sigma
    kernels.point_data["sigma_mean"] = sigma.mean(axis=1)
    kernels.point_data["sigma_volume"] = sigma.prod(axis=1)
    kernels.point_data["amplitude"] = amplitude
    kernels.point_data["amplitude_norm"] = np.linalg.norm(amplitude, axis=1)
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=np.float32).reshape(-1, 3, 3)
        if len(covariance) != len(centers):
            raise ValueError("covariance must have the same number of kernels as centers.")
        kernels.point_data["covariance_flat"] = covariance.reshape(-1, 9)
    if beta_shape is not None:
        beta_shape = np.asarray(beta_shape, dtype=np.float32).reshape(-1, 3)
        if len(beta_shape) != len(centers):
            raise ValueError("beta_shape must have the same number of kernels as centers.")
        kernels.point_data["beta_shape"] = beta_shape
        kernels.point_data["beta_shape_mean"] = beta_shape.mean(axis=1)
    kernels.save(output_path)


def write_fast_vector_bundle(
    snapshot: VTKSnapshot,
    predicted_velocity: np.ndarray,
    output_dir: str | Path,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Save a machine-friendly result bundle for fast post-processing.

    Arrays are written as standalone `.npy` files so they can be loaded with
    `numpy.load(..., mmap_mode='r')` for cheap slicing and derivative work.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_velocity = np.asarray(predicted_velocity, dtype=np.float32)
    if predicted_velocity.shape != snapshot.dimensions + (3,):
        predicted_velocity = _reshape_vector_field(predicted_velocity.reshape(-1, 3), snapshot.dimensions)

    velocity_true = np.asarray(snapshot.velocity, dtype=np.float32)
    velocity_error = predicted_velocity - velocity_true

    bundle_arrays = {
        "coords": np.asarray(snapshot.coords, dtype=np.float32),
        "velocity_true": velocity_true,
        "velocity_pred": predicted_velocity,
        "velocity_error": velocity_error,
        "speed_true": np.linalg.norm(velocity_true, axis=-1).astype(np.float32),
        "speed_pred": np.linalg.norm(predicted_velocity, axis=-1).astype(np.float32),
        "speed_error": np.linalg.norm(predicted_velocity, axis=-1).astype(np.float32)
        - np.linalg.norm(velocity_true, axis=-1).astype(np.float32),
    }

    if extra_arrays is not None:
        for name, array in extra_arrays.items():
            bundle_arrays[name] = np.asarray(array, dtype=np.float32)

    for name, array in bundle_arrays.items():
        np.save(output_dir / f"{name}.npy", array)

    metadata = {
        "file_path": str(snapshot.file_path),
        "field_name": snapshot.field_name,
        "dimensions": list(snapshot.dimensions),
        "points_min": snapshot.points_min.tolist(),
        "points_max": snapshot.points_max.tolist(),
        "array_files": {name: f"{name}.npy" for name in bundle_arrays},
    }
    with open(output_dir / "metadata.json", "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)
