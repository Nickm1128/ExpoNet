"""ExpoActivation: Learnable blend of ReLU and squared ReLU."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ExpoActivation(nn.Module):
    """Learnable blend of ReLU and squared ReLU.

    Computes:
        u = relu(x)
        f(x; a) = (1 - a) * u + a * u * u

    The blend coefficient a is learned via backpropagation, constrained to [0, 1]
    using a = sigmoid(theta) for an unconstrained parameter theta.

    Args:
        num_features: Number of features. Required for per-neuron mode. Ignored
            for per-layer mode.
        blend_mode: "per_neuron" (default) or "per_layer".
        blend_init: Initial blend coefficient. Must be in (0, 1) for trainable
            mode; [0, 1] for fixed mode.
        trainable: If True, blend coefficient is learned; if False, fixed.
        feature_dim: Axis containing features. Default -1 (last axis).
    """

    def __init__(
        self,
        num_features: int | None = None,
        *,
        blend_mode: str = "per_neuron",
        blend_init: float = 0.5,
        trainable: bool = True,
        feature_dim: int = -1,
    ) -> None:
        super().__init__()
        self.blend_mode = blend_mode
        self.blend_init = blend_init
        self.trainable = trainable
        self.feature_dim = feature_dim

        if blend_mode not in ("per_neuron", "per_layer"):
            raise ValueError(
                f"blend_mode must be 'per_neuron' or 'per_layer', got {blend_mode!r}"
            )

        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise ValueError(
                "feature_dim must be an integer axis, "
                f"got {feature_dim!r} ({type(feature_dim).__name__})"
            )

        if blend_mode == "per_neuron":
            if num_features is None:
                raise ValueError("num_features is required for per_neuron mode")
            if isinstance(num_features, bool) or not isinstance(num_features, int):
                raise ValueError(
                    "num_features must be a positive integer for per_neuron mode, "
                    f"got {num_features!r} ({type(num_features).__name__})"
                )
            if num_features <= 0:
                raise ValueError(
                    "num_features must be a positive integer for per_neuron mode, "
                    f"got {num_features}"
                )
            self.num_features = num_features
        else:
            if num_features is not None:
                raise ValueError("num_features must be None for per_layer mode")
            self.num_features = None

        if trainable:
            if not (0 < blend_init < 1):
                raise ValueError(
                    f"blend_init must be in (0, 1) for trainable mode, got {blend_init}"
                )
            theta_init = math.log(blend_init / (1 - blend_init))
            if blend_mode == "per_neuron":
                theta = torch.full((num_features,), theta_init)
                self.theta = nn.Parameter(theta)
            else:
                theta = torch.tensor([theta_init])
                self.theta = nn.Parameter(theta)
        else:
            if not (0 <= blend_init <= 1):
                raise ValueError(
                    f"blend_init must be in [0, 1] for fixed mode, got {blend_init}"
                )
            if blend_mode == "per_neuron":
                a_init = torch.full((num_features,), blend_init)
            else:
                a_init = torch.tensor([blend_init])
            self.register_buffer("a_fixed", a_init)
            self.theta = None

    @property
    def blend_weight(self) -> Tensor:
        """Return effective blend coefficient a in [0, 1]."""
        if self.trainable:
            return torch.sigmoid(self.theta)
        return self.a_fixed

    def forward(self, x: Tensor) -> Tensor:
        """Apply the blend activation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with same shape and dtype as input.
        """
        if x.ndim == 0:
            raise ValueError(f"Scalar inputs are not supported, got {x.shape}")

        u = torch.relu(x)

        if self.blend_mode == "per_neuron":
            feature_dim = self.feature_dim
            if feature_dim < 0:
                feature_dim = x.ndim + feature_dim
            if feature_dim < 0 or feature_dim >= x.ndim:
                raise ValueError(
                    f"feature_dim={self.feature_dim} out of range for {x.ndim}-D input"
                )
            if x.shape[feature_dim] != self.num_features:
                raise ValueError(
                    f"Input dimension {feature_dim} has size {x.shape[feature_dim]}, "
                    f"expected {self.num_features}"
                )
            a = self.blend_weight.view(self._expand_shape(x.ndim))
        else:
            if x.ndim == 1:
                a = self.blend_weight.view(-1)
            else:
                a = self.blend_weight.view(1, -1, *(1,) * (x.ndim - 2))

        return u * ((1.0 - a) + a * u)

    def _expand_shape(self, ndim: int) -> tuple[int, ...]:
        """Expand shape for broadcasting along feature_dim."""
        if self.feature_dim < 0:
            feature_dim = ndim + self.feature_dim
        else:
            feature_dim = self.feature_dim

        if self.num_features is None:
            raise RuntimeError("num_features must be set for per_neuron mode")

        shape = [1] * ndim
        shape[feature_dim] = self.num_features
        return tuple(shape)

    def extra_repr(self) -> str:
        """Set extra representation string."""
        parts = [
            f"blend_mode={self.blend_mode!r}",
            f"trainable={self.trainable}",
        ]
        if self.num_features is not None:
            parts.append(f"num_features={self.num_features}")
        parts.append(f"blend_init={self.blend_init}")
        parts.append(f"feature_dim={self.feature_dim}")
        return ", ".join(parts)
