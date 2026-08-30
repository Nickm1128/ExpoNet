"""Scikit-learn compatible estimators built from ExpoNet modules."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from exponet._persistence import load_snapshot, restore_scaler, save_snapshot
from exponet._training import train_classification, train_regression
from exponet._validation import (
    as_classification_labels,
    as_float32_features,
    as_regression_targets,
    encode_labels,
    require_finite,
    resolve_device,
)
from exponet.nn import ExpoMLP


@contextmanager
def _torch_seed(seed: int | None) -> Iterator[None]:
    """Seed model initialization without leaving process-global RNG changed."""
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class ExpoRegressor(RegressorMixin, BaseEstimator):
    """Dense numeric regression with a trainable ReLU/squared-ReLU blend."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        blend_mode: str = "per_neuron",
        blend_init: float = 0.5,
        trainable_blend: bool = True,
        normalization: str = "layer",
        standardize: bool = True,
        target_standardize: bool = False,
        lr: float = 1e-3,
        blend_lr: float | None = None,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        patience: int = 15,
        min_delta: float = 0.0,
        max_grad_norm: float | None = None,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.blend_mode = blend_mode
        self.blend_init = blend_init
        self.trainable_blend = trainable_blend
        self.normalization = normalization
        self.standardize = standardize
        self.target_standardize = target_standardize
        self.lr = lr
        self.blend_lr = blend_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _clear_fitted_state(self) -> None:
        for name in (
            "model_",
            "n_features_in_",
            "n_outputs_",
            "history_",
            "n_iter_",
            "best_epoch_",
            "device_",
            "feature_scaler_",
            "target_scaler_",
            "_target_was_1d",
        ):
            self.__dict__.pop(name, None)

    def set_params(self, **params: object) -> "ExpoRegressor":
        result = super().set_params(**params)
        if params:
            self._clear_fitted_state()
        return result

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _finite(name: str, value: object, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result) or (positive and result <= 0):
            qualifier = "positive finite" if positive else "finite"
            raise ValueError(f"{name} must be a {qualifier} number")
        return result

    def _validate_configuration(self) -> None:
        self._positive_int("batch_size", self.batch_size)
        self._positive_int("epochs", self.epochs)
        self._positive_int("patience", self.patience)
        self._finite("lr", self.lr, positive=True)
        if self.blend_lr is not None:
            self._finite("blend_lr", self.blend_lr, positive=True)
        if self._finite("weight_decay", self.weight_decay) < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self._finite("min_delta", self.min_delta) < 0:
            raise ValueError("min_delta must be nonnegative")
        if self.max_grad_norm is not None:
            self._finite("max_grad_norm", self.max_grad_norm, positive=True)
        if not isinstance(self.standardize, bool) or not isinstance(
            self.target_standardize, bool
        ):
            raise ValueError("standardize and target_standardize must be booleans")
        if not isinstance(self.shuffle, bool) or not isinstance(
            self.early_stopping, bool
        ):
            raise ValueError("shuffle and early_stopping must be booleans")
        if not isinstance(self.verbose, int) or self.verbose not in (0, 1):
            raise ValueError("verbose must be 0 or 1")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
        ):
            raise ValueError("random_state must be an integer or None")

    def _split_training_data(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.float32],
        validation_data: tuple[ArrayLike, ArrayLike] | None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32] | None,
        NDArray[np.float32] | None,
    ]:
        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (X_val, y_val)")
            x_val = as_float32_features(validation_data[0], name="X_val")
            y_val, _ = as_regression_targets(
                validation_data[1], n_samples=len(x_val), name="y_val"
            )
            if x_val.shape[1] != x.shape[1] or y_val.shape[1] != y.shape[1]:
                raise ValueError(
                    "validation_data feature and output shapes must match training"
                )
            return x, y, x_val, y_val
        if not self.early_stopping:
            return x, y, None, None
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be in (0, 1) with early_stopping"
            )
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            shuffle=True,
        )
        return x_train, y_train, x_val, y_val

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> "ExpoRegressor":
        self._clear_fitted_state()
        self._validate_configuration()
        device = resolve_device(self.device)
        x = as_float32_features(X)
        targets, target_was_1d = as_regression_targets(y, n_samples=len(x))
        x_train, y_train, x_val, y_val = self._split_training_data(
            x, targets, validation_data
        )

        feature_scaler: StandardScaler | None = None
        if self.standardize:
            feature_scaler = StandardScaler().fit(x_train)
            x_train = require_finite(feature_scaler.transform(x_train), name="X_train")
            if x_val is not None:
                x_val = require_finite(feature_scaler.transform(x_val), name="X_val")

        target_scaler: StandardScaler | None = None
        if self.target_standardize:
            target_scaler = StandardScaler().fit(y_train)
            y_train = require_finite(target_scaler.transform(y_train), name="y_train")
            if y_val is not None:
                y_val = require_finite(target_scaler.transform(y_val), name="y_val")

        with _torch_seed(self.random_state):
            model = ExpoMLP(
                x.shape[1],
                targets.shape[1],
                hidden_dims=self.hidden_dims,
                blend_mode=self.blend_mode,
                blend_init=self.blend_init,
                trainable_blend=self.trainable_blend,
                normalization=self.normalization,
            ).to(device=device, dtype=torch.float32)
        result = train_regression(
            model,
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            lr=float(self.lr),
            blend_lr=None if self.blend_lr is None else float(self.blend_lr),
            weight_decay=float(self.weight_decay),
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            shuffle=self.shuffle,
            early_stopping=self.early_stopping,
            patience=int(self.patience),
            min_delta=float(self.min_delta),
            max_grad_norm=None
            if self.max_grad_norm is None
            else float(self.max_grad_norm),
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_ = model
        self.n_features_in_ = x.shape[1]
        self.n_outputs_ = targets.shape[1]
        self.history_ = result.history
        self.n_iter_ = result.n_iter
        self.best_epoch_ = result.best_epoch
        self.device_ = str(device)
        self.feature_scaler_ = feature_scaler
        self.target_scaler_ = target_scaler
        self._target_was_1d = target_was_1d
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("ExpoRegressor is not fitted")

    def predict(self, X: ArrayLike) -> NDArray[np.float32]:
        self._require_fitted()
        x = as_float32_features(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but ExpoRegressor is expecting "
                f"{self.n_features_in_} features as input."
            )
        if self.feature_scaler_ is not None:
            x = require_finite(self.feature_scaler_.transform(x), name="X")
        chunks: list[NDArray[np.float32]] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(x), int(self.batch_size)):
                batch = torch.as_tensor(
                    x[start : start + int(self.batch_size)], device=self.device_
                )
                output = self.model_(batch)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("nonfinite model output during prediction")
                chunks.append(output.detach().cpu().numpy())
        prediction = np.concatenate(chunks, axis=0)
        if self.target_scaler_ is not None:
            prediction = require_finite(
                self.target_scaler_.inverse_transform(prediction), name="prediction"
            )
        return prediction[:, 0] if self._target_was_1d else prediction

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        self._require_fitted()
        targets, rank_one = as_regression_targets(
            y, n_samples=len(as_float32_features(X))
        )
        actual = targets[:, 0] if rank_one else targets
        return float(r2_score(actual, self.predict(X)))

    def get_blend_weights(self) -> list[NDArray[np.float32]]:
        """Return detached copies of one effective blend vector per hidden layer."""
        self._require_fitted()
        return [
            block.activation.blend_weight.detach().cpu().numpy().copy()
            for block in self.model_.hidden_blocks
        ]

    def save(self, path: str) -> None:
        """Write a versioned inference snapshot without moving the live model."""
        self._require_fitted()
        save_snapshot(self, kind="regressor", path=path)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "ExpoRegressor":
        """Load a restricted regression snapshot onto the requested device."""
        payload = load_snapshot(path, expected_kind="regressor")
        estimator, fitted, resolved = _restore_estimator_shell(cls, payload, device)
        try:
            n_outputs = fitted["n_outputs"]
            target_was_1d = fitted["target_was_1d"]
            target_scaler = restore_scaler(
                fitted["target_scaler"],
                expected_features=n_outputs,
                name="target_scaler",
            )
        except KeyError as error:
            raise ValueError("snapshot regression metadata is incomplete") from error
        if (
            not isinstance(n_outputs, int)
            or isinstance(n_outputs, bool)
            or n_outputs <= 0
            or not isinstance(target_was_1d, bool)
            or bool(estimator.target_standardize) != (target_scaler is not None)
        ):
            raise ValueError("snapshot regression metadata is invalid")
        model = _restore_model(
            payload,
            estimator,
            in_features=estimator.n_features_in_,
            out_features=n_outputs,
            device=resolved,
        )
        estimator.model_ = model
        estimator.n_outputs_ = n_outputs
        estimator.target_scaler_ = target_scaler
        estimator._target_was_1d = target_was_1d
        return estimator


class ExpoClassifier(ClassifierMixin, BaseEstimator):
    """Dense integer or string-label classification with K-logit training."""

    def __init__(
        self,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        blend_mode: str = "per_neuron",
        blend_init: float = 0.5,
        trainable_blend: bool = True,
        normalization: str = "layer",
        standardize: bool = True,
        lr: float = 1e-3,
        blend_lr: float | None = None,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        epochs: int = 100,
        shuffle: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        patience: int = 15,
        min_delta: float = 0.0,
        max_grad_norm: float | None = None,
        device: str = "cpu",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.blend_mode = blend_mode
        self.blend_init = blend_init
        self.trainable_blend = trainable_blend
        self.normalization = normalization
        self.standardize = standardize
        self.lr = lr
        self.blend_lr = blend_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _clear_fitted_state(self) -> None:
        for name in (
            "model_",
            "n_features_in_",
            "n_classes_",
            "classes_",
            "history_",
            "n_iter_",
            "best_epoch_",
            "device_",
            "feature_scaler_",
        ):
            self.__dict__.pop(name, None)

    def set_params(self, **params: object) -> "ExpoClassifier":
        result = super().set_params(**params)
        if params:
            self._clear_fitted_state()
        return result

    def _validate_configuration(self) -> None:
        ExpoRegressor._positive_int("batch_size", self.batch_size)
        ExpoRegressor._positive_int("epochs", self.epochs)
        ExpoRegressor._positive_int("patience", self.patience)
        ExpoRegressor._finite("lr", self.lr, positive=True)
        if self.blend_lr is not None:
            ExpoRegressor._finite("blend_lr", self.blend_lr, positive=True)
        if ExpoRegressor._finite("weight_decay", self.weight_decay) < 0:
            raise ValueError("weight_decay must be nonnegative")
        if ExpoRegressor._finite("min_delta", self.min_delta) < 0:
            raise ValueError("min_delta must be nonnegative")
        if self.max_grad_norm is not None:
            ExpoRegressor._finite("max_grad_norm", self.max_grad_norm, positive=True)
        if not isinstance(self.standardize, bool):
            raise ValueError("standardize must be a boolean")
        if not isinstance(self.shuffle, bool) or not isinstance(
            self.early_stopping, bool
        ):
            raise ValueError("shuffle and early_stopping must be booleans")
        if not isinstance(self.verbose, int) or self.verbose not in (0, 1):
            raise ValueError("verbose must be 0 or 1")
        if self.random_state is not None and (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
        ):
            raise ValueError("random_state must be an integer or None")

    def _split_training_data(
        self,
        x: NDArray[np.float32],
        encoded: NDArray[np.int64],
        classes: NDArray[np.generic],
        validation_data: tuple[ArrayLike, ArrayLike] | None,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.float32] | None,
        NDArray[np.int64] | None,
    ]:
        if validation_data is not None:
            if not isinstance(validation_data, tuple) or len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (X_val, y_val)")
            x_val = as_float32_features(validation_data[0], name="X_val")
            labels_val = as_classification_labels(
                validation_data[1], n_samples=len(x_val), name="y_val"
            )
            if x_val.shape[1] != x.shape[1]:
                raise ValueError("validation_data feature count must match training")
            return x, encoded, x_val, encode_labels(labels_val, classes, name="y_val")
        if not self.early_stopping:
            return x, encoded, None, None
        if not 0 < self.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be in (0, 1) with early_stopping"
            )
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=True,
                stratify=encoded,
            )
        except ValueError as error:
            raise ValueError(f"stratified validation split failed: {error}") from error
        return x_train, y_train, x_val, y_val

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        validation_data: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> "ExpoClassifier":
        self._clear_fitted_state()
        self._validate_configuration()
        device = resolve_device(self.device)
        x = as_float32_features(X)
        labels = as_classification_labels(y, n_samples=len(x))
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("y must contain at least two training classes")
        encoded = encode_labels(labels, classes, name="y")
        x_train, y_train, x_val, y_val = self._split_training_data(
            x, encoded, classes, validation_data
        )

        feature_scaler: StandardScaler | None = None
        if self.standardize:
            feature_scaler = StandardScaler().fit(x_train)
            x_train = require_finite(feature_scaler.transform(x_train), name="X_train")
            if x_val is not None:
                x_val = require_finite(feature_scaler.transform(x_val), name="X_val")

        with _torch_seed(self.random_state):
            model = ExpoMLP(
                x.shape[1],
                len(classes),
                hidden_dims=self.hidden_dims,
                blend_mode=self.blend_mode,
                blend_init=self.blend_init,
                trainable_blend=self.trainable_blend,
                normalization=self.normalization,
            ).to(device=device, dtype=torch.float32)
        result = train_classification(
            model,
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            device=device,
            lr=float(self.lr),
            blend_lr=None if self.blend_lr is None else float(self.blend_lr),
            weight_decay=float(self.weight_decay),
            batch_size=int(self.batch_size),
            epochs=int(self.epochs),
            shuffle=self.shuffle,
            early_stopping=self.early_stopping,
            patience=int(self.patience),
            min_delta=float(self.min_delta),
            max_grad_norm=None
            if self.max_grad_norm is None
            else float(self.max_grad_norm),
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.model_ = model
        self.n_features_in_ = x.shape[1]
        self.n_classes_ = len(classes)
        self.classes_ = classes.copy()
        self.history_ = result.history
        self.n_iter_ = result.n_iter
        self.best_epoch_ = result.best_epoch
        self.device_ = str(device)
        self.feature_scaler_ = feature_scaler
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise NotFittedError("ExpoClassifier is not fitted")

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        self._require_fitted()
        x = as_float32_features(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {x.shape[1]} features, but ExpoClassifier is expecting "
                f"{self.n_features_in_} features as input."
            )
        if self.feature_scaler_ is not None:
            x = require_finite(self.feature_scaler_.transform(x), name="X")
        chunks: list[NDArray[np.float32]] = []
        self.model_.eval()
        with torch.inference_mode():
            for start in range(0, len(x), int(self.batch_size)):
                batch = torch.as_tensor(
                    x[start : start + int(self.batch_size)], device=self.device_
                )
                logits = self.model_(batch)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite model output during prediction")
                chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict(self, X: ArrayLike) -> NDArray[np.generic]:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        self._require_fitted()
        x = as_float32_features(X)
        labels = as_classification_labels(y, n_samples=len(x))
        encode_labels(labels, self.classes_, name="y")
        return float(accuracy_score(labels, self.predict(x)))

    def get_blend_weights(self) -> list[NDArray[np.float32]]:
        """Return detached copies of one effective blend vector per hidden layer."""
        self._require_fitted()
        return [
            block.activation.blend_weight.detach().cpu().numpy().copy()
            for block in self.model_.hidden_blocks
        ]

    def save(self, path: str) -> None:
        """Write a versioned inference snapshot without moving the live model."""
        self._require_fitted()
        save_snapshot(self, kind="classifier", path=path)

    @classmethod
    def load(cls, path: str, *, device: str = "cpu") -> "ExpoClassifier":
        """Load a restricted classification snapshot onto the requested device."""
        payload = load_snapshot(path, expected_kind="classifier")
        estimator, fitted, resolved = _restore_estimator_shell(cls, payload, device)
        try:
            n_classes = fitted["n_classes"]
            raw_classes = fitted["classes"]
        except KeyError as error:
            raise ValueError("snapshot classifier metadata is incomplete") from error
        if (
            not isinstance(n_classes, int)
            or isinstance(n_classes, bool)
            or n_classes < 2
            or not isinstance(raw_classes, list)
            or len(raw_classes) != n_classes
        ):
            raise ValueError("snapshot classifier metadata is invalid")
        classes = as_classification_labels(
            np.asarray(raw_classes), n_samples=n_classes, name="snapshot classes"
        )
        if len(np.unique(classes)) != n_classes or not np.array_equal(
            classes, np.unique(classes)
        ):
            raise ValueError("snapshot classes must be distinct and sorted")
        model = _restore_model(
            payload,
            estimator,
            in_features=estimator.n_features_in_,
            out_features=n_classes,
            device=resolved,
        )
        estimator.model_ = model
        estimator.n_classes_ = n_classes
        estimator.classes_ = classes.copy()
        return estimator


def _restore_estimator_shell(
    estimator_class: type[ExpoRegressor] | type[ExpoClassifier],
    payload: dict[str, object],
    device: str,
) -> tuple[ExpoRegressor | ExpoClassifier, dict[str, object], torch.device]:
    """Validate common metadata and rebuild an unfitted estimator shell."""
    constructor = payload["constructor"]
    fitted = payload["fitted"]
    assert isinstance(constructor, dict)
    assert isinstance(fitted, dict)
    config = dict(constructor)
    hidden_dims = config.get("hidden_dims")
    if not isinstance(hidden_dims, list):
        raise ValueError("snapshot hidden_dims metadata is invalid")
    config["hidden_dims"] = tuple(hidden_dims)
    config["device"] = device
    try:
        estimator = estimator_class(**config)
        estimator._validate_configuration()
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot constructor metadata is invalid") from error
    resolved = resolve_device(device)
    try:
        n_features = fitted["n_features_in"]
        n_iter = fitted["n_iter"]
        best_epoch = fitted["best_epoch"]
        history = fitted["history"]
        feature_scaler = restore_scaler(
            fitted["feature_scaler"],
            expected_features=n_features,
            name="feature_scaler",
        )
    except KeyError as error:
        raise ValueError("snapshot fitted metadata is incomplete") from error
    if (
        not isinstance(n_features, int)
        or isinstance(n_features, bool)
        or n_features <= 0
        or not isinstance(n_iter, int)
        or isinstance(n_iter, bool)
        or n_iter <= 0
        or (
            best_epoch is not None
            and (not isinstance(best_epoch, int) or best_epoch <= 0)
        )
        or not isinstance(history, list)
        or bool(estimator.standardize) != (feature_scaler is not None)
    ):
        raise ValueError("snapshot fitted metadata is invalid")
    estimator.n_features_in_ = n_features
    estimator.n_iter_ = n_iter
    estimator.best_epoch_ = best_epoch
    estimator.history_ = history
    estimator.device_ = str(resolved)
    estimator.feature_scaler_ = feature_scaler
    return estimator, fitted, resolved


def _restore_model(
    payload: dict[str, object],
    estimator: ExpoRegressor | ExpoClassifier,
    *,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> ExpoMLP:
    """Construct the only supported model type and load its state strictly."""
    model = ExpoMLP(
        in_features,
        out_features,
        hidden_dims=estimator.hidden_dims,
        blend_mode=estimator.blend_mode,
        blend_init=estimator.blend_init,
        trainable_blend=estimator.trainable_blend,
        normalization=estimator.normalization,
    ).to(device=device, dtype=torch.float32)
    state = payload["model_state"]
    assert isinstance(state, dict)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "snapshot model state is incompatible with its metadata"
        ) from error
    return model
