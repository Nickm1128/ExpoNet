"""Shared supervised training utilities for ExpoNet estimators."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from exponet.activations import ExpoActivation


@dataclass(frozen=True)
class TrainingResult:
    """Outputs retained by an estimator after a completed training run."""

    history: list[dict[str, object]]
    n_iter: int
    best_epoch: int | None


def make_optimizer(
    model: nn.Module, *, lr: float, blend_lr: float | None, weight_decay: float
) -> torch.optim.AdamW:
    """Create non-overlapping AdamW groups with decay only on linear weights."""
    linear_weights = {
        id(module.weight) for module in model.modules() if isinstance(module, nn.Linear)
    }
    blend_parameters = {
        id(module.theta)
        for module in model.modules()
        if isinstance(module, ExpoActivation) and module.theta is not None
    }
    backbone: list[nn.Parameter] = []
    blend: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in linear_weights:
            backbone.append(parameter)
        elif id(parameter) in blend_parameters:
            blend.append(parameter)
        else:
            no_decay.append(parameter)
    grouped_ids = [
        id(parameter) for group in (backbone, blend, no_decay) for parameter in group
    ]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {
        id(parameter) for parameter in model.parameters()
    }:
        raise RuntimeError(
            "optimizer parameter grouping must include every parameter once"
        )
    groups: list[dict[str, object]] = []
    if backbone:
        groups.append({"params": backbone, "lr": lr, "weight_decay": weight_decay})
    if blend:
        groups.append(
            {
                "params": blend,
                "lr": lr if blend_lr is None else blend_lr,
                "weight_decay": 0.0,
            }
        )
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def _assert_finite(tensor: Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"nonfinite {name} encountered during training")


def _blend_summary(model: nn.Module) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    for module in model.modules():
        if isinstance(module, ExpoActivation):
            values = module.blend_weight.detach()
            summaries.append(
                {
                    "min": float(values.min().item()),
                    "mean": float(values.mean().item()),
                    "max": float(values.max().item()),
                }
            )
    return summaries


def _loss_over_dataset(
    model: nn.Module,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    *,
    batch_size: int,
    device: torch.device,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> float:
    model.eval()
    weighted_loss = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            features = torch.as_tensor(x[start:stop], device=device)
            targets = torch.as_tensor(y[start:stop], device=device)
            output = model(features)
            _assert_finite(output, name="validation output")
            loss = loss_fn(output, targets)
            _assert_finite(loss, name="validation loss")
            weighted_loss += float(loss.item()) * (stop - start)
    return weighted_loss / len(x)


def train_supervised(
    model: nn.Module,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.generic],
    *,
    x_val: NDArray[np.float32] | None,
    y_val: NDArray[np.generic] | None,
    device: torch.device,
    lr: float,
    blend_lr: float | None,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    shuffle: bool,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    max_grad_norm: float | None,
    random_state: int | None,
    verbose: int,
    loss_fn: nn.Module,
    target_dtype: torch.dtype,
) -> TrainingResult:
    """Train from CPU arrays using bounded batches and a caller-provided loss."""
    optimizer = make_optimizer(
        model, lr=lr, blend_lr=blend_lr, weight_decay=weight_decay
    )
    rng = np.random.default_rng(random_state)
    history: list[dict[str, object]] = []
    best_loss = math.inf
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    stalled_epochs = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = rng.permutation(len(x_train)) if shuffle else np.arange(len(x_train))
        model.train()
        weighted_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = torch.as_tensor(x_train[indices], device=device)
            targets = torch.as_tensor(
                y_train[indices], device=device, dtype=target_dtype
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            _assert_finite(output, name="model output")
            loss = loss_fn(output, targets)
            _assert_finite(loss, name="loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    _assert_finite(parameter.grad, name="gradient")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            for parameter in model.parameters():
                _assert_finite(parameter, name="model parameter")
            weighted_loss += float(loss.item()) * len(indices)

        train_loss = weighted_loss / len(x_train)
        val_loss = (
            None
            if x_val is None or y_val is None
            else _loss_over_dataset(
                model,
                x_val,
                y_val,
                batch_size=batch_size,
                device=device,
                loss_fn=loss_fn,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_seconds": time.perf_counter() - started,
                "blend_weights": _blend_summary(model),
            }
        )
        if verbose:
            suffix = "" if val_loss is None else f" val_loss={val_loss:.6g}"
            print(f"epoch={epoch} train_loss={train_loss:.6g}{suffix}")

        if early_stopping:
            if val_loss is None:
                raise RuntimeError("early stopping requires validation data")
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stalled_epochs = 0
            else:
                stalled_epochs += 1
                if stalled_epochs >= patience:
                    break

    if early_stopping:
        if best_state is None or best_epoch is None:
            raise RuntimeError(
                "early stopping did not observe a finite validation result"
            )
        model.load_state_dict(best_state)
    return TrainingResult(history=history, n_iter=len(history), best_epoch=best_epoch)


def train_regression(
    model: nn.Module,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    **kwargs: object,
) -> TrainingResult:
    """Train a regression model through the shared supervised loop."""
    return train_supervised(
        model,
        x_train,
        y_train,
        loss_fn=nn.MSELoss(),
        target_dtype=torch.float32,
        **kwargs,
    )


def train_classification(
    model: nn.Module,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    **kwargs: object,
) -> TrainingResult:
    """Train a K-logit classifier through the shared supervised loop."""
    return train_supervised(
        model,
        x_train,
        y_train,
        loss_fn=nn.CrossEntropyLoss(),
        target_dtype=torch.long,
        **kwargs,
    )
