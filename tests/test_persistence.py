"""Tests for restricted, versioned ExpoNet inference snapshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from exponet import ExpoClassifier, ExpoRegressor


def regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a compact regression fixture with two numeric features."""
    features = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(32, 2)
    targets = (1.25 * features[:, 0] - 0.5 * features[:, 1] + 0.3).astype(np.float32)
    return features, targets


def classifier_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a compact string-label three-class fixture."""
    features = np.array(
        [
            [-2.0, -2.0],
            [-1.8, -2.1],
            [-2.2, -1.8],
            [2.0, -1.0],
            [1.8, -1.1],
            [2.1, -0.8],
            [0.0, 2.0],
            [0.2, 1.8],
            [-0.2, 2.1],
        ],
        dtype=np.float32,
    )
    labels = np.repeat(np.array(["alpha", "beta", "gamma"]), 3)
    return features, labels


def small_regressor(**overrides: object) -> ExpoRegressor:
    options: dict[str, object] = {
        "hidden_dims": (4,),
        "epochs": 18,
        "batch_size": 7,
        "lr": 0.03,
        "random_state": 5,
    }
    options.update(overrides)
    return ExpoRegressor(**options)


def small_classifier(**overrides: object) -> ExpoClassifier:
    options: dict[str, object] = {
        "hidden_dims": (5,),
        "normalization": "none",
        "epochs": 35,
        "batch_size": 4,
        "lr": 0.03,
        "random_state": 6,
    }
    options.update(overrides)
    return ExpoClassifier(**options)


@pytest.mark.parametrize("normalization", ["none", "layer"])
@pytest.mark.parametrize("rank_one", [True, False])
def test_regression_snapshot_preserves_scalers_rank_and_normalization(
    tmp_path: Path, normalization: str, rank_one: bool
) -> None:
    features, targets = regression_data()
    y = targets if rank_one else np.column_stack((targets, 2.0 * targets - 1.0))
    source = small_regressor(normalization=normalization, target_standardize=True).fit(
        features, y
    )
    path = tmp_path / "regression.pt"
    source.save(path)
    restored = ExpoRegressor.load(path)
    np.testing.assert_allclose(
        restored.predict(features), source.predict(features), rtol=1e-5, atol=1e-6
    )
    assert restored.predict(features[:2]).shape == y[:2].shape
    assert restored.target_scaler_ is not None
    assert restored.feature_scaler_ is not None
    assert restored.normalization == normalization
    for actual, expected in zip(
        restored.get_blend_weights(), source.get_blend_weights(), strict=True
    ):
        np.testing.assert_allclose(actual, expected)


def test_snapshot_preserves_disabled_scalers_and_loaded_fit_starts_fresh(
    tmp_path: Path,
) -> None:
    features, targets = regression_data()
    source = small_regressor(
        normalization="none", standardize=False, target_standardize=False, epochs=4
    ).fit(features, targets)
    path = tmp_path / "unscaled.pt"
    source.save(path)
    restored = ExpoRegressor.load(path)
    assert restored.feature_scaler_ is None
    assert restored.target_scaler_ is None
    loaded_model = restored.model_
    restored.fit(features, -targets)
    assert restored.model_ is not loaded_model


def test_fixed_blend_coefficients_survive_snapshot(tmp_path: Path) -> None:
    features, targets = regression_data()
    source = small_regressor(
        normalization="none",
        trainable_blend=False,
        blend_init=1.0,
        epochs=5,
    ).fit(features, targets)
    path = tmp_path / "fixed-blend.pt"
    source.save(path)
    np.testing.assert_allclose(ExpoRegressor.load(path).get_blend_weights()[0], 1.0)


def test_classifier_string_labels_and_probability_order_survive_snapshot(
    tmp_path: Path,
) -> None:
    features, labels = classifier_data()
    source = small_classifier().fit(features, labels)
    path = tmp_path / "classifier.pt"
    source.save(path)
    restored = ExpoClassifier.load(path)
    np.testing.assert_array_equal(restored.classes_, source.classes_)
    np.testing.assert_allclose(
        restored.predict_proba(features), source.predict_proba(features)
    )
    np.testing.assert_array_equal(restored.predict(features), source.predict(features))


def test_incompatible_kind_unknown_version_damaged_and_wrong_state_fail(
    tmp_path: Path,
) -> None:
    features, targets = regression_data()
    source = small_regressor(normalization="none").fit(features, targets)
    valid_path = tmp_path / "valid.pt"
    source.save(valid_path)
    with pytest.raises(ValueError, match="not 'classifier'"):
        ExpoClassifier.load(valid_path)

    unknown_version = tmp_path / "unknown-version.pt"
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    payload["format_version"] = 999
    torch.save(payload, unknown_version)
    with pytest.raises(ValueError, match="format version"):
        ExpoRegressor.load(unknown_version)

    wrong_state = tmp_path / "wrong-state.pt"
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    payload["model_state"]["output_layer.weight"] = torch.zeros((9, 9))
    torch.save(payload, wrong_state)
    with pytest.raises(ValueError, match="model state"):
        ExpoRegressor.load(wrong_state)

    damaged = tmp_path / "damaged.pt"
    damaged.write_bytes(b"not a torch snapshot")
    with pytest.raises(ValueError, match="invalid or unreadable"):
        ExpoRegressor.load(damaged)


def test_failed_atomic_write_preserves_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, targets = regression_data()
    source = small_regressor(normalization="none", epochs=3).fit(features, targets)
    destination = tmp_path / "preserve.pt"
    destination.write_bytes(b"existing snapshot")

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("exponet._persistence.torch.save", fail_save)
    with pytest.raises(OSError, match="simulated write failure"):
        source.save(destination)
    assert destination.read_bytes() == b"existing snapshot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_to_cpu_snapshot_is_portable_without_moving_live_model(
    tmp_path: Path,
) -> None:
    features, targets = regression_data()
    source = small_regressor(device="cuda", normalization="none").fit(features, targets)
    path = tmp_path / "cuda-source.pt"
    source.save(path)
    assert all(
        parameter.device.type == "cuda" for parameter in source.model_.parameters()
    )
    restored = ExpoRegressor.load(path, device="cpu")
    assert restored.device_ == "cpu"
    assert all(
        parameter.device.type == "cpu" for parameter in restored.model_.parameters()
    )
    np.testing.assert_allclose(
        restored.predict(features), source.predict(features), rtol=1e-5, atol=1e-6
    )
