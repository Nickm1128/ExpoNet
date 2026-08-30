"""Tests for the P4 scikit-learn classification workflow."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from exponet import ExpoClassifier


def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a small, clearly separated three-class numeric dataset."""
    generator = np.random.default_rng(12)
    centers = np.array([[-2.0, -2.0], [2.0, -1.0], [0.0, 2.0]], dtype=np.float32)
    features = np.concatenate(
        [
            center + 0.2 * generator.standard_normal((15, 2)).astype(np.float32)
            for center in centers
        ]
    )
    labels = np.repeat(np.array(["alpha", "beta", "gamma"]), 15)
    return features, labels


def small_classifier(**overrides: object) -> ExpoClassifier:
    """Build a fast deterministic classifier for tests."""
    options: dict[str, object] = {
        "hidden_dims": (6,),
        "normalization": "none",
        "epochs": 50,
        "batch_size": 8,
        "lr": 0.03,
        "random_state": 4,
    }
    options.update(overrides)
    return ExpoClassifier(**options)


class TestExpoClassifierLabelsAndPredictions:
    """Verify label contracts, K-logit inference, and fitted lifecycle."""

    def test_string_multiclass_probabilities_and_accuracy(self) -> None:
        x, y = classification_data()
        estimator = small_classifier().fit(x, y)
        probabilities = estimator.predict_proba(x)
        prediction = estimator.predict(x)
        np.testing.assert_array_equal(estimator.classes_, ["alpha", "beta", "gamma"])
        assert probabilities.shape == (len(x), 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert prediction.shape == y.shape
        assert estimator.score(x, y) > 0.98

    def test_integer_binary_labels_still_use_two_probability_columns(self) -> None:
        x, _ = classification_data()
        y = np.where(x[:, 0] > 0, 10, -3).astype(np.int64)
        estimator = small_classifier(epochs=35).fit(x, y)
        probabilities = estimator.predict_proba(x[:3])
        np.testing.assert_array_equal(estimator.classes_, [-3, 10])
        assert probabilities.shape == (3, 2)
        assert set(estimator.predict(x[:3])).issubset({-3, 10})

    @pytest.mark.parametrize(
        "invalid",
        [
            np.array([["alpha"], ["beta"]]),
            np.array([1.0, 2.0]),
            np.array([True, False]),
            np.array(["alpha", 1], dtype=object),
            np.array(["alpha", None], dtype=object),
            np.array(["only", "only"]),
        ],
    )
    def test_invalid_or_single_class_labels_fail_clearly(
        self, invalid: np.ndarray
    ) -> None:
        x = np.ones((2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            small_classifier(epochs=1).fit(x, invalid)

    def test_unseen_validation_or_score_label_fails(self) -> None:
        x, y = classification_data()
        with pytest.raises(ValueError, match="absent"):
            small_classifier(epochs=2).fit(
                x, y, validation_data=(x[:3], np.array(["unknown"] * 3))
            )
        estimator = small_classifier().fit(x, y)
        with pytest.raises(ValueError, match="absent"):
            estimator.score(x[:2], np.array(["unknown", "alpha"]))

    def test_not_fitted_parameter_change_and_repeated_fit_reset_state(self) -> None:
        x, y = classification_data()
        estimator = small_classifier(epochs=4)
        with pytest.raises(NotFittedError):
            estimator.predict(x[:1])
        estimator.fit(x, y)
        first_model = estimator.model_
        estimator.fit(x, np.where(y == "alpha", "left", "right"))
        assert estimator.model_ is not first_model
        np.testing.assert_array_equal(estimator.classes_, ["left", "right"])
        estimator.set_params(lr=0.02)
        with pytest.raises(NotFittedError):
            estimator.predict(x[:1])

    def test_fixed_blend_and_copied_weight_inspection(self) -> None:
        x, y = classification_data()
        estimator = small_classifier(
            trainable_blend=False, blend_init=0.0, epochs=10
        ).fit(x, y)
        weights = estimator.get_blend_weights()
        np.testing.assert_allclose(weights[0], 0.0)
        weights[0][:] = 1.0
        np.testing.assert_allclose(estimator.get_blend_weights()[0], 0.0)


class TestExpoClassifierIntegrationAndDevices:
    """Verify stratification, sklearn integration, and CUDA execution."""

    def test_stratified_holdout_failure_is_actionable(self) -> None:
        x = np.arange(6, dtype=np.float32).reshape(3, 2)
        y = np.array(["a", "a", "b"])
        with pytest.raises(ValueError, match="stratified validation split failed"):
            small_classifier(epochs=2, early_stopping=True).fit(x, y)

    def test_clone_pipeline_and_cross_validation_smoke(self) -> None:
        x, y = classification_data()
        estimator = small_classifier(epochs=15)
        assert clone(estimator).get_params() == estimator.get_params()
        pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        pipeline.fit(x, y)
        assert pipeline.predict(x[:2]).shape == (2,)
        scores = cross_val_score(small_classifier(epochs=15), x, y, cv=3)
        assert scores.shape == (3,)
        assert np.all(scores > 0.8)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_classification_fit_predict_and_blend_update(self) -> None:
        x, y = classification_data()
        estimator = small_classifier(device="cuda", epochs=30).fit(x, y)
        probabilities = estimator.predict_proba(x[:5])
        assert estimator.device_ == "cuda"
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert not np.allclose(estimator.get_blend_weights()[0], 0.5)
