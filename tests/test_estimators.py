"""Tests for the P3 scikit-learn regression workflow."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exponet import ExpoRegressor
from exponet._training import make_optimizer
from exponet.nn import ExpoMLP


def regression_data(
    n_samples: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a small deterministic regression problem with a linear target."""
    x = np.linspace(-1.0, 1.0, n_samples * 2, dtype=np.float32).reshape(n_samples, 2)
    y = (1.5 * x[:, 0] - 0.75 * x[:, 1] + 0.2).astype(np.float32)
    return x, y


def small_regressor(**overrides: object) -> ExpoRegressor:
    """Build a fast, deterministic estimator suitable for unit tests."""
    options: dict[str, object] = {
        "hidden_dims": (4,),
        "normalization": "none",
        "epochs": 40,
        "batch_size": 7,
        "lr": 0.03,
        "random_state": 7,
    }
    options.update(overrides)
    return ExpoRegressor(**options)


class TestExpoRegressorDataAndLifecycle:
    """Exercise data validation, output rank, and sklearn lifecycle semantics."""

    @pytest.mark.parametrize(
        "invalid",
        [
            np.ones((2, 2, 1)),
            np.empty((0, 2)),
            np.ones((2, 0)),
            np.array([[1.0, np.nan]]),
            np.array([[1.0 + 2.0j]]),
            np.array([["not-numeric"]]),
            sparse.csr_matrix(np.eye(2)),
            torch.ones(2, 2),
        ],
    )
    def test_invalid_features_fail_before_model_creation(self, invalid: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            small_regressor().fit(invalid, np.ones(2, dtype=np.float32))

    def test_invalid_targets_and_mismatched_sample_counts_fail(self) -> None:
        x, y = regression_data()
        with pytest.raises(ValueError, match="samples"):
            small_regressor().fit(x, y[:-1])
        with pytest.raises(ValueError, match="finite"):
            small_regressor().fit(x, np.full(len(x), np.inf))
        with pytest.raises(ValueError, match="rank one or two"):
            small_regressor().fit(x, np.ones((len(x), 1, 1)))

    def test_predict_before_fit_and_parameter_change_require_refit(self) -> None:
        x, y = regression_data()
        estimator = small_regressor()
        with pytest.raises(NotFittedError):
            estimator.predict(x[:1])
        estimator.fit(x, y)
        estimator.set_params(lr=0.02)
        with pytest.raises(NotFittedError):
            estimator.predict(x[:1])

    def test_fit_predict_score_and_rank_preservation(self) -> None:
        x, y = regression_data()
        estimator = small_regressor().fit(x, y)
        prediction = estimator.predict(x)
        assert prediction.shape == y.shape
        assert estimator.score(x, y) > 0.98
        assert estimator.n_features_in_ == 2
        assert estimator.n_outputs_ == 1
        assert estimator.n_iter_ == 40
        assert estimator.best_epoch_ is None
        assert len(estimator.history_) == estimator.n_iter_
        assert estimator.history_[0]["val_loss"] is None

    def test_multioutput_and_target_scaling_preserve_rank(self) -> None:
        x, y = regression_data()
        targets = np.column_stack((y, 2.0 * y - 1.0)).astype(np.float32)
        estimator = small_regressor(target_standardize=True).fit(x, targets)
        prediction = estimator.predict(x[:3])
        assert prediction.shape == (3, 2)
        assert estimator.target_scaler_ is not None
        assert estimator.score(x, targets) > 0.98

    def test_scalers_only_fit_training_rows_not_explicit_validation(self) -> None:
        x = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
        y = x[:, 0].copy()
        x_val = np.array([[100.0]], dtype=np.float32)
        y_val = np.array([100.0], dtype=np.float32)
        estimator = small_regressor(epochs=2, early_stopping=True).fit(
            x, y, validation_data=(x_val, y_val)
        )
        assert estimator.feature_scaler_ is not None
        np.testing.assert_allclose(estimator.feature_scaler_.mean_, [3.0])
        assert estimator.history_[0]["val_loss"] is not None

    def test_fixed_blends_do_not_change_and_inspection_returns_copies(self) -> None:
        x, y = regression_data()
        estimator = small_regressor(trainable_blend=False, blend_init=0.5).fit(x, y)
        weights = estimator.get_blend_weights()
        assert len(weights) == 1
        np.testing.assert_allclose(weights[0], 0.5)
        weights[0][:] = 0.0
        np.testing.assert_allclose(estimator.get_blend_weights()[0], 0.5)

    def test_failed_refit_invalidates_old_model(self) -> None:
        x, y = regression_data()
        estimator = small_regressor().fit(x, y)
        with pytest.raises(ValueError):
            estimator.fit(np.array([[np.nan]], dtype=np.float32), np.array([1.0]))
        with pytest.raises(NotFittedError):
            estimator.predict(x[:1])

    def test_repeated_fit_replaces_history_and_model_state(self) -> None:
        x, y = regression_data()
        estimator = small_regressor(epochs=5).fit(x, y)
        first_model = estimator.model_
        estimator.fit(x, -y)
        assert estimator.model_ is not first_model
        assert estimator.n_iter_ == 5
        assert len(estimator.history_) == 5

    def test_clone_pipeline_and_grid_search_smoke(self) -> None:
        x, y = regression_data(30)
        estimator = small_regressor(epochs=12)
        cloned = clone(estimator)
        assert cloned.get_params() == estimator.get_params()
        pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        pipeline.fit(x, y)
        assert pipeline.predict(x[:2]).shape == (2,)
        search = GridSearchCV(
            small_regressor(epochs=10), {"lr": [0.02, 0.03]}, cv=2
        ).fit(x, y)
        assert isinstance(search.best_estimator_, ExpoRegressor)


class TestExpoRegressorTrainingAndDevices:
    """Verify trainer behavior, CUDA execution, and CPU/GPU inference parity."""

    def test_early_stopping_tracks_and_restores_a_best_epoch(self) -> None:
        x, y = regression_data()
        estimator = small_regressor(
            epochs=20,
            early_stopping=True,
            patience=2,
            min_delta=10.0,
        ).fit(x, y, validation_data=(x[:8], -y[:8]))
        assert estimator.best_epoch_ == 1
        assert estimator.n_iter_ == 3
        assert estimator.history_[0]["val_loss"] is not None

    def test_optimizer_groups_are_complete_and_decay_only_linear_weights(self) -> None:
        model = ExpoMLP(2, 1, hidden_dims=(3,), normalization="none")
        optimizer = make_optimizer(model, lr=0.01, blend_lr=0.02, weight_decay=0.3)
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        assert {id(parameter) for parameter in grouped} == {
            id(parameter) for parameter in model.parameters()
        }
        assert len(grouped) == len({id(parameter) for parameter in grouped})
        theta = model.hidden_blocks[0].activation.theta
        assert theta is not None
        theta_group = next(
            group
            for group in optimizer.param_groups
            if any(parameter is theta for parameter in group["params"])
        )
        assert theta_group["lr"] == 0.02
        assert theta_group["weight_decay"] == 0.0

    def test_explicit_cuda_never_falls_back(self) -> None:
        x, y = regression_data(8)
        with pytest.raises(ValueError, match=r"CUDA (was requested|device 99)"):
            small_regressor(device="cuda:99", epochs=1).fit(x, y)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_regression_and_cpu_gpu_state_parity(self) -> None:
        x, y = regression_data()
        estimator = small_regressor(device="cuda", epochs=25).fit(x, y)
        prediction = estimator.predict(x[:4])
        assert estimator.device_ == "cuda"
        assert np.isfinite(prediction).all()
        learned = estimator.get_blend_weights()[0]
        assert np.isfinite(learned).all()
        assert not np.allclose(learned, 0.5)
        cpu_model = estimator.model_.to("cpu").eval()
        with torch.inference_mode():
            cpu_output = cpu_model(torch.as_tensor(x[:4]))
            gpu_output = cpu_model.to("cuda")(torch.as_tensor(x[:4], device="cuda"))
        torch.testing.assert_close(cpu_output, gpu_output.cpu(), rtol=1e-5, atol=1e-6)
