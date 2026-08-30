"""Restricted, versioned inference snapshots for ExpoNet estimators."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

FORMAT_VERSION = 1


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy state tensors to CPU without changing the live model's device."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _scaler_payload(scaler: StandardScaler | None) -> dict[str, object] | None:
    if scaler is None:
        return None
    return {
        "mean": torch.as_tensor(scaler.mean_, dtype=torch.float64).clone(),
        "scale": torch.as_tensor(scaler.scale_, dtype=torch.float64).clone(),
        "var": torch.as_tensor(scaler.var_, dtype=torch.float64).clone(),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).item()),
    }


def _primitive_constructor(params: dict[str, object]) -> dict[str, object]:
    result = dict(params)
    result["hidden_dims"] = list(result["hidden_dims"])
    return result


def save_snapshot(estimator: Any, *, kind: str, path: str | Path) -> None:
    """Atomically write a restricted inference snapshot for a fitted estimator."""
    destination = Path(path)
    if not destination.parent.exists():
        raise ValueError(f"snapshot directory does not exist: {destination.parent}")
    if not destination.parent.is_dir():
        raise ValueError(f"snapshot parent is not a directory: {destination.parent}")
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "estimator_kind": kind,
        "constructor": _primitive_constructor(estimator.get_params(deep=False)),
        "model_state": _cpu_state_dict(estimator.model_),
        "fitted": {
            "n_features_in": int(estimator.n_features_in_),
            "n_iter": int(estimator.n_iter_),
            "best_epoch": estimator.best_epoch_,
            "history": estimator.history_,
            "feature_scaler": _scaler_payload(estimator.feature_scaler_),
        },
        "producer": {
            "exponet": "0.1.0",
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    fitted = payload["fitted"]
    assert isinstance(fitted, dict)
    if kind == "regressor":
        fitted.update(
            {
                "n_outputs": int(estimator.n_outputs_),
                "target_was_1d": bool(estimator._target_was_1d),
                "target_scaler": _scaler_payload(estimator.target_scaler_),
            }
        )
    elif kind == "classifier":
        fitted.update(
            {
                "n_classes": int(estimator.n_classes_),
                "classes": estimator.classes_.tolist(),
            }
        )
    else:
        raise ValueError(f"unsupported estimator kind: {kind!r}")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_snapshot(path: str | Path, *, expected_kind: str) -> dict[str, object]:
    """Read and validate only the outer restricted snapshot envelope."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("invalid or unreadable ExpoNet snapshot") from error
    if not isinstance(payload, dict):
        raise ValueError("snapshot must contain a dictionary payload")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported ExpoNet snapshot format version")
    if payload.get("estimator_kind") != expected_kind:
        raise ValueError(
            f"snapshot is for {payload.get('estimator_kind')!r}, not {expected_kind!r}"
        )
    if not isinstance(payload.get("constructor"), dict):
        raise ValueError("snapshot constructor metadata is invalid")
    if not isinstance(payload.get("fitted"), dict):
        raise ValueError("snapshot fitted metadata is invalid")
    if not isinstance(payload.get("model_state"), dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload["model_state"].items()
    ):
        raise ValueError("snapshot model state is invalid")
    return payload


def restore_scaler(
    payload: object, *, expected_features: int, name: str
) -> StandardScaler | None:
    """Rebuild a fitted StandardScaler from finite tensor-only metadata."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot {name} metadata is invalid")
    try:
        count = payload["n_features_in"]
        seen = payload["n_samples_seen"]
        mean = payload["mean"]
        scale = payload["scale"]
        var = payload["var"]
    except KeyError as error:
        raise ValueError(f"snapshot {name} metadata is incomplete") from error
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count != expected_features
        or not isinstance(seen, int)
        or isinstance(seen, bool)
        or seen <= 0
        or not all(isinstance(value, torch.Tensor) for value in (mean, scale, var))
    ):
        raise ValueError(f"snapshot {name} metadata is invalid")
    arrays = [
        value.detach().cpu().numpy().astype(np.float64, copy=True)
        for value in (mean, scale, var)
    ]
    if any(array.shape != (expected_features,) for array in arrays) or not all(
        np.isfinite(array).all() for array in arrays
    ):
        raise ValueError(f"snapshot {name} arrays are invalid")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] < 0):
        raise ValueError(f"snapshot {name} arrays are invalid")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = arrays
    scaler.n_features_in_ = expected_features
    scaler.n_samples_seen_ = seen
    return scaler
