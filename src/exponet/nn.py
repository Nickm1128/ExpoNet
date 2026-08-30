"""Dense PyTorch modules composed from ExpoActivation blocks."""

from __future__ import annotations

from collections import OrderedDict

from torch import Tensor, nn

from exponet.activations import ExpoActivation


class ExpoMLP(nn.Module):
    """A dense MLP with trainable ReLU/squared-ReLU blend activations.

    Each hidden block is composed in this exact order:
    ``Linear -> LayerNorm or Identity -> ExpoActivation``. The output layer is
    linear only, so the module supports signed regression outputs and logits.

    Args:
        in_features: Required number of input features.
        out_features: Required number of output features.
        hidden_dims: Nonempty tuple of positive hidden-layer widths.
        blend_mode: Coefficient-sharing mode passed to each ExpoActivation.
        blend_init: Initial effective blend coefficient passed to each
            ExpoActivation.
        trainable_blend: Whether each activation's blend coefficient is learned.
        normalization: ``"layer"`` for LayerNorm before each activation or
            ``"none"`` to omit normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_dims: tuple[int, ...] = (64, 64),
        blend_mode: str = "per_neuron",
        blend_init: float = 0.5,
        trainable_blend: bool = True,
        normalization: str = "layer",
    ) -> None:
        super().__init__()
        self.in_features = self._validate_positive_integer("in_features", in_features)
        self.out_features = self._validate_positive_integer(
            "out_features", out_features
        )
        self.hidden_dims = self._validate_hidden_dims(hidden_dims)

        if normalization not in ("layer", "none"):
            raise ValueError(
                f"normalization must be 'layer' or 'none', got {normalization!r}"
            )
        if normalization == "layer" and any(width == 1 for width in self.hidden_dims):
            raise ValueError(
                "hidden_dims cannot contain width 1 when normalization='layer'"
            )

        self.blend_mode = blend_mode
        self.blend_init = blend_init
        self.trainable_blend = trainable_blend
        self.normalization = normalization

        blocks: list[nn.Sequential] = []
        previous_width = self.in_features
        for width in self.hidden_dims:
            norm: nn.Module
            if normalization == "layer":
                norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=True)
            else:
                norm = nn.Identity()
            block = nn.Sequential(
                OrderedDict(
                    [
                        ("linear", nn.Linear(previous_width, width)),
                        ("normalization", norm),
                        (
                            "activation",
                            ExpoActivation(
                                width if blend_mode == "per_neuron" else None,
                                blend_mode=blend_mode,
                                blend_init=blend_init,
                                trainable=trainable_blend,
                            ),
                        ),
                    ]
                )
            )
            blocks.append(block)
            previous_width = width

        self.hidden_blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(previous_width, self.out_features)
        self._reset_linear_parameters()

    @staticmethod
    def _validate_positive_integer(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r} "
                f"({type(value).__name__})"
            )
        return value

    @classmethod
    def _validate_hidden_dims(cls, hidden_dims: object) -> tuple[int, ...]:
        if not isinstance(hidden_dims, tuple):
            raise ValueError(
                "hidden_dims must be a nonempty tuple of positive integers, "
                f"got {hidden_dims!r} ({type(hidden_dims).__name__})"
            )
        if not hidden_dims:
            raise ValueError(
                "hidden_dims must be a nonempty tuple of positive integers"
            )
        for index, width in enumerate(hidden_dims):
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError(
                    "hidden_dims must contain only positive integers; "
                    f"hidden_dims[{index}] is {width!r} "
                    f"({type(width).__name__})"
                )
        return hidden_dims

    def _reset_linear_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the hidden blocks and final linear readout to a rank-two tensor."""
        if not isinstance(x, Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.ndim != 2:
            raise ValueError(
                "ExpoMLP expects a rank-two input shaped (batch, in_features), "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_features:
            raise ValueError(
                "ExpoMLP input feature dimension mismatch: "
                f"got {x.shape[1]}, expected {self.in_features}"
            )

        for block in self.hidden_blocks:
            x = block(x)
        return self.output_layer(x)
