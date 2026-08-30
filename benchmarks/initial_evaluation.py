"""Run ExpoNet's paired initial activation evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import sklearn
import torch
from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from exponet import ExpoClassifier, ExpoRegressor
from exponet._training import train_classification, train_regression

SEEDS = (11, 23, 37, 53, 71)
VARIANTS = (
    "native_relu",
    "native_squared_relu",
    "fixed_expo_0",
    "fixed_expo_05",
    "fixed_expo_1",
    "learned_per_layer",
    "learned_per_neuron",
)
NORMALIZATIONS = ("none", "layer")
HIDDEN_DIMS = (16,)
LEARNING_RATE = 0.01
BATCH_SIZE = 128
EPOCHS = 35


class SquaredReLU(nn.Module):
    """Native squared-ReLU reference used only by the evaluation runner."""

    def forward(self, x: Tensor) -> Tensor:
        u = torch.relu(x)
        return u * u


class NativeMLP(nn.Module):
    """Matched dense reference that differs only in its native activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        normalization: str,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unsupported normalization: {normalization}")
        norm: nn.Module = nn.Identity()
        if normalization == "layer":
            norm = nn.LayerNorm(HIDDEN_DIMS[0], eps=1e-5, elementwise_affine=True)
        self.hidden = nn.Sequential(
            nn.Linear(in_features, HIDDEN_DIMS[0]), norm, activation
        )
        self.output = nn.Linear(HIDDEN_DIMS[0], out_features)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.output(self.hidden(x))


@dataclass(frozen=True)
class Dataset:
    """One numeric workload and its reproducible provenance details."""

    name: str
    task: str
    features: np.ndarray
    targets: np.ndarray
    source: str
    license_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    try:
        args.seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    except ValueError as error:
        parser.error(f"--seeds must be comma-separated integers: {error}")
    if len(args.seeds) < 1:
        parser.error("--seeds must contain at least one seed")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is False")
    return args


def datasets() -> tuple[Dataset, ...]:
    """Load synthetic controls and bundled real numeric datasets without I/O."""
    synthetic_x, synthetic_y = make_regression(
        n_samples=240,
        n_features=8,
        n_informative=6,
        noise=10.0,
        random_state=20260830,
    )
    synthetic_class_x, synthetic_class_y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=20260830,
    )
    diabetes = load_diabetes()
    iris = load_iris()
    return (
        Dataset(
            "synthetic_regression",
            "regression",
            synthetic_x.astype(np.float32),
            synthetic_y.astype(np.float32),
            "sklearn.datasets.make_regression(seed=20260830)",
            "Generated locally by scikit-learn; no external dataset license applies.",
        ),
        Dataset(
            "diabetes",
            "regression",
            diabetes.data.astype(np.float32),
            diabetes.target.astype(np.float32),
            "scikit-learn bundled Diabetes dataset (Efron et al., 2004)",
            "Bundled by scikit-learn (BSD-3-Clause); its description does not "
            "specify a separate source-data license.",
        ),
        Dataset(
            "synthetic_multiclass",
            "classification",
            synthetic_class_x.astype(np.float32),
            synthetic_class_y.astype(np.int64),
            "sklearn.datasets.make_classification(seed=20260830)",
            "Generated locally by scikit-learn; no external dataset license applies.",
        ),
        Dataset(
            "iris",
            "classification",
            iris.data.astype(np.float32),
            iris.target.astype(np.int64),
            "scikit-learn bundled Iris dataset (Fisher, 1936)",
            "Bundled by scikit-learn (BSD-3-Clause); its description does not "
            "specify a separate source-data license.",
        ),
    )


def digest(dataset: Dataset) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(dataset.features).tobytes())
    hasher.update(np.ascontiguousarray(dataset.targets).tobytes())
    return hasher.hexdigest()


def variant_config(variant: str) -> dict[str, object]:
    """Return the estimator settings for one accepted blend control."""
    settings: dict[str, object] = {}
    if variant == "fixed_expo_0":
        settings.update(trainable_blend=False, blend_init=0.0)
    elif variant == "fixed_expo_05":
        settings.update(trainable_blend=False, blend_init=0.5)
    elif variant == "fixed_expo_1":
        settings.update(trainable_blend=False, blend_init=1.0)
    elif variant == "learned_per_layer":
        settings.update(trainable_blend=True, blend_mode="per_layer", blend_init=0.5)
    elif variant == "learned_per_neuron":
        settings.update(trainable_blend=True, blend_mode="per_neuron", blend_init=0.5)
    elif variant not in {"native_relu", "native_squared_relu"}:
        raise ValueError(f"unknown variant: {variant}")
    return settings


def split_dataset(
    dataset: Dataset, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create paired 64/16/20 train/validation/test partitions."""
    stratify = dataset.targets if dataset.task == "classification" else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        dataset.features,
        dataset.targets,
        test_size=0.20,
        random_state=seed,
        stratify=stratify,
    )
    stratify_train = y_train_val if dataset.task == "classification" else None
    train_val_split = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.20,
        random_state=seed + 10_000,
        stratify=stratify_train,
    )
    return (*train_val_split, x_test, y_test)


def _native_model(
    variant: str, in_features: int, out_features: int, normalization: str, seed: int
) -> NativeMLP:
    torch_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        activation: nn.Module = nn.ReLU() if variant == "native_relu" else SquaredReLU()
        return NativeMLP(
            in_features,
            out_features,
            normalization=normalization,
            activation=activation,
        )
    finally:
        torch.random.set_rng_state(torch_state)


def _native_predict(
    model: nn.Module, features: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return model(torch.as_tensor(features, device=device)).detach().cpu().numpy()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_native(
    dataset: Dataset,
    variant: str,
    normalization: str,
    seed: int,
    splits: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    device: torch.device,
    epochs: int,
) -> dict[str, object]:
    """Train a matched native activation reference on prepared paired splits."""
    x_train, x_val, y_train, y_val, x_test, y_test = splits
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    target_scaler: StandardScaler | None = None
    if dataset.task == "regression":
        target_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_model = target_scaler.transform(y_train.reshape(-1, 1)).astype(
            np.float32
        )
        y_val_model = target_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
        model = _native_model(variant, x_train.shape[1], 1, normalization, seed).to(
            device
        )
        started = time.perf_counter()
        result = train_regression(
            model,
            x_train,
            y_train_model,
            x_val=x_val,
            y_val=y_val_model,
            device=device,
            lr=LEARNING_RATE,
            blend_lr=None,
            weight_decay=0.0,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            shuffle=True,
            early_stopping=False,
            patience=15,
            min_delta=0.0,
            max_grad_norm=None,
            random_state=seed,
            verbose=0,
        )
        elapsed = time.perf_counter() - started
        prediction = target_scaler.inverse_transform(
            _native_predict(model, x_test, device)
        )[:, 0]
        metrics = {
            "rmse": float(np.sqrt(np.mean((prediction - y_test) ** 2))),
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
    else:
        classes = np.unique(y_train)
        encoded_train = np.searchsorted(classes, y_train).astype(np.int64)
        encoded_val = np.searchsorted(classes, y_val).astype(np.int64)
        model = _native_model(
            variant, x_train.shape[1], len(classes), normalization, seed
        ).to(device)
        started = time.perf_counter()
        result = train_classification(
            model,
            x_train,
            encoded_train,
            x_val=x_val,
            y_val=encoded_val,
            device=device,
            lr=LEARNING_RATE,
            blend_lr=None,
            weight_decay=0.0,
            batch_size=BATCH_SIZE,
            epochs=epochs,
            shuffle=True,
            early_stopping=False,
            patience=15,
            min_delta=0.0,
            max_grad_norm=None,
            random_state=seed,
            verbose=0,
        )
        elapsed = time.perf_counter() - started
        probabilities = torch.softmax(
            torch.as_tensor(_native_predict(model, x_test, device)), dim=1
        ).numpy()
        metrics = {
            "accuracy": float(
                accuracy_score(y_test, classes[probabilities.argmax(axis=1)])
            ),
            "log_loss": float(log_loss(y_test, probabilities, labels=classes)),
        }
    return {
        "elapsed_seconds": elapsed,
        "parameter_count": _parameter_count(model),
        "n_iter": result.n_iter,
        "metrics": metrics,
        "blend_weights": [],
    }


def run_exponet(
    dataset: Dataset,
    variant: str,
    normalization: str,
    seed: int,
    splits: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    device: str,
    epochs: int,
) -> dict[str, object]:
    """Train one ExpoNet activation variant on the paired raw-data split."""
    x_train, x_val, y_train, y_val, x_test, y_test = splits
    config = {
        "hidden_dims": HIDDEN_DIMS,
        "normalization": normalization,
        "standardize": True,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": epochs,
        "early_stopping": False,
        "device": device,
        "random_state": seed,
        **variant_config(variant),
    }
    started = time.perf_counter()
    if dataset.task == "regression":
        estimator = ExpoRegressor(target_standardize=True, **config).fit(
            x_train, y_train, validation_data=(x_val, y_val)
        )
        prediction = estimator.predict(x_test)
        metrics = {
            "rmse": float(np.sqrt(np.mean((prediction - y_test) ** 2))),
            "mae": float(mean_absolute_error(y_test, prediction)),
            "r2": float(r2_score(y_test, prediction)),
        }
    else:
        estimator = ExpoClassifier(**config).fit(
            x_train, y_train, validation_data=(x_val, y_val)
        )
        probabilities = estimator.predict_proba(x_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, estimator.predict(x_test))),
            "log_loss": float(
                log_loss(y_test, probabilities, labels=estimator.classes_)
            ),
        }
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "parameter_count": _parameter_count(estimator.model_),
        "n_iter": estimator.n_iter_,
        "metrics": metrics,
        "blend_weights": [
            weights.tolist() for weights in estimator.get_blend_weights()
        ],
    }


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize completed paired runs while retaining raw records separately."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for record in records:
        if record["status"] == "completed":
            key = (record["dataset"], record["normalization"], record["variant"])
            grouped.setdefault(key, []).append(record)
    summaries: list[dict[str, object]] = []
    for key, runs in sorted(grouped.items()):
        metric_names = runs[0]["metrics"].keys()
        metrics = {
            name: {
                "mean": mean(run["metrics"][name] for run in runs),
                "std": stdev(run["metrics"][name] for run in runs)
                if len(runs) > 1
                else 0.0,
            }
            for name in metric_names
        }
        summaries.append(
            {
                "dataset": key[0],
                "normalization": key[1],
                "variant": key[2],
                "completed_runs": len(runs),
                "elapsed_seconds": {
                    "mean": mean(run["elapsed_seconds"] for run in runs),
                    "std": stdev(run["elapsed_seconds"] for run in runs)
                    if len(runs) > 1
                    else 0.0,
                },
                "parameter_count": runs[0]["parameter_count"],
                "metrics": metrics,
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    records: list[dict[str, object]] = []
    dataset_metadata = []
    for dataset in datasets():
        dataset_metadata.append(
            {
                "name": dataset.name,
                "task": dataset.task,
                "samples": len(dataset.features),
                "features": dataset.features.shape[1],
                "source": dataset.source,
                "license_note": dataset.license_note,
                "sha256": digest(dataset),
            }
        )
        for seed in args.seeds:
            splits = split_dataset(dataset, seed)
            split_sizes = {
                "train": len(splits[0]),
                "validation": len(splits[2]),
                "test": len(splits[4]),
            }
            for normalization in NORMALIZATIONS:
                for variant in VARIANTS:
                    record: dict[str, object] = {
                        "dataset": dataset.name,
                        "task": dataset.task,
                        "seed": seed,
                        "normalization": normalization,
                        "variant": variant,
                        "split_sizes": split_sizes,
                    }
                    try:
                        result = (
                            run_native(
                                dataset,
                                variant,
                                normalization,
                                seed,
                                splits,
                                device,
                                args.epochs,
                            )
                            if variant.startswith("native_")
                            else run_exponet(
                                dataset,
                                variant,
                                normalization,
                                seed,
                                splits,
                                args.device,
                                args.epochs,
                            )
                        )
                        record.update(status="completed", **result)
                    except Exception as error:
                        record.update(
                            status="failed", error=f"{type(error).__name__}: {error}"
                        )
                    records.append(record)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "protocol": {
            "seeds": list(args.seeds),
            "variants": VARIANTS,
            "normalizations": NORMALIZATIONS,
            "hidden_dims": HIDDEN_DIMS,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "splits": "train/validation/test = 64/16/20 by paired seed",
            "feature_standardization": "fit on train partition only",
            "target_standardization": "regression only, fit on train partition only",
        },
        "datasets": dataset_metadata,
        "runs": records,
        "summary": aggregate(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    completed = sum(record["status"] == "completed" for record in records)
    print(f"wrote={args.output.resolve()}")
    print(f"completed={completed} attempted={len(records)}")


if __name__ == "__main__":
    main()
