"""Validation and preprocessing helpers for ExpoNet estimators."""

from __future__ import annotations

import re
from numbers import Real

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy import sparse


def resolve_device(device: object) -> torch.device:
    """Resolve an estimator device without silently falling back from CUDA."""
    if not isinstance(device, str):
        raise ValueError("device must be 'cpu', 'auto', 'cuda', or 'cuda:N'")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("device must be 'cpu', 'auto', 'cuda', or 'cuda:N'")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is False")
    resolved = torch.device(device)
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {resolved.index} was requested, but only "
            f"{torch.cuda.device_count()} device(s) are available"
        )
    return resolved


def as_float32_features(
    x: ArrayLike, *, allow_empty: bool = False, name: str = "X"
) -> NDArray[np.float32]:
    """Validate a dense, finite rank-two numeric feature array."""
    if sparse.issparse(x):
        raise ValueError(f"{name} must be a dense numeric array, not sparse")
    if isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be an array-like object, not a torch.Tensor")
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank two, got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError(
            f"{name} has 0 feature(s) (shape={array.shape}) while a minimum of 1 "
            "is required."
        )
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one sample")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"Complex data not supported for {name}")
    if array.dtype == object and all(
        isinstance(value, (Real, np.number))
        and not isinstance(value, (bool, np.bool_))
        and not isinstance(value, complex)
        for value in array.ravel()
    ):
        pass
    elif array.dtype == object:
        raise TypeError("argument must be a string or number")
    elif not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain real numeric values")
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return np.ascontiguousarray(converted)


def as_regression_targets(
    y: ArrayLike, *, n_samples: int, name: str = "y"
) -> tuple[NDArray[np.float32], bool]:
    """Validate regression targets and retain whether callers supplied rank one."""
    if sparse.issparse(y):
        raise ValueError(f"{name} must be a dense numeric array, not sparse")
    if isinstance(y, torch.Tensor):
        raise TypeError(f"{name} must be an array-like object, not a torch.Tensor")
    array = np.asarray(y)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be rank one or two, got shape {array.shape}")
    if array.shape[0] != n_samples:
        raise ValueError(
            f"{name} has {array.shape[0]} samples, but X has {n_samples} samples"
        )
    if array.ndim == 2 and array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one output")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"Complex data not supported for {name}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain real numeric values")
    rank_one = array.ndim == 1
    converted = np.asarray(
        array.reshape(-1, 1) if rank_one else array, dtype=np.float32
    )
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return np.ascontiguousarray(converted), rank_one


def as_classification_labels(
    y: ArrayLike, *, n_samples: int, name: str = "y"
) -> NDArray[np.generic]:
    """Validate one-dimensional homogeneous integer or string class labels."""
    if sparse.issparse(y):
        raise ValueError(f"{name} must be a dense one-dimensional label array")
    if isinstance(y, torch.Tensor):
        raise TypeError(f"{name} must be an array-like object, not a torch.Tensor")
    array = np.asarray(y)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label array")
    if len(array) != n_samples:
        raise ValueError(
            f"{name} has {len(array)} samples, but X has {n_samples} samples"
        )
    if np.issubdtype(array.dtype, np.integer) and not np.issubdtype(
        array.dtype, np.bool_
    ):
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.str_):
        if np.any(array == ""):
            raise ValueError(f"{name} must not contain empty string labels")
        return np.ascontiguousarray(array)
    if array.dtype != object:
        raise ValueError(f"{name} must contain only integer or string labels")
    values = array.tolist()
    if not values or any(value is None for value in values):
        raise ValueError(f"{name} must contain only integer or string labels")
    if all(isinstance(value, (str, np.str_)) and value != "" for value in values):
        return np.ascontiguousarray(np.asarray(values, dtype=str))
    if all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in values
    ):
        return np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    raise ValueError(
        f"{name} must contain one consistent type of integer or string labels"
    )


def encode_labels(
    labels: NDArray[np.generic], classes: NDArray[np.generic], *, name: str
) -> NDArray[np.int64]:
    """Encode labels in a fitted class order and reject unseen values."""
    lookup = {value: index for index, value in enumerate(classes.tolist())}
    try:
        encoded = np.asarray(
            [lookup[value] for value in labels.tolist()], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            f"{name} contains a label absent from the training classes"
        ) from error
    return np.ascontiguousarray(encoded)


def require_finite(array: NDArray[np.float32], *, name: str) -> NDArray[np.float32]:
    """Check transformations did not create an invalid float32 value."""
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name} became nonfinite after preprocessing")
    return np.ascontiguousarray(converted)
