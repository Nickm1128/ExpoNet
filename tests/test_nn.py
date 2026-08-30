"""Tests for the direct-PyTorch ExpoMLP module."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from exponet import ExpoActivation, ExpoMLP


class TestExpoMLPComposition:
    """Verify the declared dense architecture and configuration behavior."""

    def test_exact_hidden_block_order_and_module_counts(self) -> None:
        """Hidden blocks contain Linear, LayerNorm, then separate activations."""
        model = ExpoMLP(3, 2, hidden_dims=(4, 5), normalization="layer")

        assert len(model.hidden_blocks) == 2
        for block in model.hidden_blocks:
            assert list(block._modules) == ["linear", "normalization", "activation"]
            assert isinstance(block.linear, nn.Linear)
            assert isinstance(block.normalization, nn.LayerNorm)
            assert isinstance(block.activation, ExpoActivation)
        assert len(list(model.modules())) == 11
        assert (
            len([module for module in model.modules() if isinstance(module, nn.Linear)])
            == 3
        )
        assert (
            len(
                [
                    module
                    for module in model.modules()
                    if isinstance(module, nn.LayerNorm)
                ]
            )
            == 2
        )
        assert (
            len(
                [
                    module
                    for module in model.modules()
                    if isinstance(module, ExpoActivation)
                ]
            )
            == 2
        )
        assert isinstance(model.output_layer, nn.Linear)

    @pytest.mark.parametrize(
        ("hidden_dims", "batch_size", "out_features"),
        [((4,), 1, 2), ((4, 5, 6), 7, 3)],
    )
    def test_single_and_multiple_hidden_layer_output_shapes(
        self, hidden_dims: tuple[int, ...], batch_size: int, out_features: int
    ) -> None:
        """The model preserves batch size and produces the configured output width."""
        model = ExpoMLP(3, out_features, hidden_dims=hidden_dims)
        output = model(torch.randn(batch_size, 3))
        assert output.shape == (batch_size, out_features)

    @pytest.mark.parametrize("name", ["in_features", "out_features"])
    @pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "2", None])
    def test_feature_counts_require_positive_non_boolean_integers(
        self, name: str, value: object
    ) -> None:
        """Input and output widths reject invalid integer-like values."""
        kwargs: dict[str, object] = {"in_features": 3, "out_features": 2}
        kwargs[name] = value
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            ExpoMLP(**kwargs)

    @pytest.mark.parametrize(
        "hidden_dims",
        [(), [], (True,), (False,), (0,), (-1,), (2.5,), ("2",), (2, 0)],
    )
    def test_hidden_dims_requires_nonempty_tuple_of_positive_integer_widths(
        self, hidden_dims: object
    ) -> None:
        """Hidden widths reject malformed tuple and element values with context."""
        with pytest.raises(ValueError, match="hidden_dims"):
            ExpoMLP(3, 2, hidden_dims=hidden_dims)

    @pytest.mark.parametrize("normalization", [None, True, "batch", "identity"])
    def test_normalization_accepts_only_layer_or_none(
        self, normalization: object
    ) -> None:
        """Only the documented normalization modes are accepted."""
        with pytest.raises(ValueError, match="normalization must be 'layer' or 'none'"):
            ExpoMLP(3, 2, normalization=normalization)

    def test_normalization_off_uses_identity_and_allows_width_one(self) -> None:
        """The no-normalization path has Identity modules and supports width one."""
        model = ExpoMLP(2, 1, hidden_dims=(1, 3), normalization="none")
        assert all(
            isinstance(block.normalization, nn.Identity)
            for block in model.hidden_blocks
        )
        assert model(torch.randn(4, 2)).shape == (4, 1)

    def test_layer_normalization_rejects_width_one(self) -> None:
        """LayerNorm would degenerate at width one and is deliberately rejected."""
        with pytest.raises(ValueError, match="width 1"):
            ExpoMLP(2, 1, hidden_dims=(1,), normalization="layer")

    def test_each_hidden_layer_has_independent_activation_parameters(self) -> None:
        """Hidden activations never accidentally share raw blend parameters."""
        model = ExpoMLP(3, 2, hidden_dims=(4, 4), blend_mode="per_neuron")
        first = model.hidden_blocks[0].activation
        second = model.hidden_blocks[1].activation
        assert first is not second
        assert first.theta is not second.theta
        assert first.theta.data_ptr() != second.theta.data_ptr()
        with torch.no_grad():
            first.theta.add_(1.0)
        assert not torch.equal(first.theta, second.theta)

    @pytest.mark.parametrize(
        ("blend_mode", "expected_shapes"),
        [("per_neuron", [(4,), (5,)]), ("per_layer", [(1,), (1,)])],
    )
    def test_per_neuron_and_per_layer_coefficient_shapes(
        self, blend_mode: str, expected_shapes: list[tuple[int, ...]]
    ) -> None:
        """Activation sharing mode is preserved for every hidden width."""
        model = ExpoMLP(3, 2, hidden_dims=(4, 5), blend_mode=blend_mode)
        actual_shapes = [
            tuple(block.activation.blend_weight.shape) for block in model.hidden_blocks
        ]
        assert actual_shapes == expected_shapes

    def test_linear_layers_use_xavier_uniform_weights_and_zero_biases(self) -> None:
        """Every linear layer observes the documented initialization rule."""
        torch.manual_seed(17)
        model = ExpoMLP(3, 2, hidden_dims=(4, 5), normalization="none")
        linear_layers = [
            module for module in model.modules() if isinstance(module, nn.Linear)
        ]
        assert len(linear_layers) == 3
        for layer in linear_layers:
            bound = math.sqrt(6.0 / (layer.in_features + layer.out_features))
            assert torch.all(layer.weight <= bound)
            assert torch.all(layer.weight >= -bound)
            assert torch.count_nonzero(layer.weight) > 0
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    @pytest.mark.parametrize("shape", [(3,), (2, 3, 1), ()])
    def test_rank_two_input_is_required(self, shape: tuple[int, ...]) -> None:
        """The direct module rejects ranks that could hide a caller mistake."""
        model = ExpoMLP(3, 2, hidden_dims=(4,))
        with pytest.raises(ValueError, match="rank-two input"):
            model(torch.randn(shape))

    def test_input_feature_mismatch_is_clear(self) -> None:
        """Input feature mismatches fail before the first linear layer."""
        model = ExpoMLP(3, 2, hidden_dims=(4,))
        with pytest.raises(ValueError, match="got 4, expected 3"):
            model(torch.randn(2, 4))

    def test_final_linear_readout_preserves_signed_outputs(self) -> None:
        """A negative readout is retained instead of being clipped."""
        model = ExpoMLP(
            1, 1, hidden_dims=(2,), normalization="none", trainable_blend=False
        )
        with torch.no_grad():
            model.hidden_blocks[0].linear.weight.fill_(1.0)
            model.hidden_blocks[0].linear.bias.zero_()
            model.output_layer.weight.fill_(-2.0)
            model.output_layer.bias.fill_(-0.5)
        output = model(torch.tensor([[1.0], [2.0]]))
        assert torch.all(output < 0)


class TestExpoMLPDirectPyTorch:
    """Verify regular PyTorch optimizer, device, and serialization behavior."""

    def test_caller_owned_optimizer_changes_linear_and_trainable_theta(self) -> None:
        """A standard optimizer updates backbone weights and learned coefficients."""
        torch.manual_seed(3)
        model = ExpoMLP(2, 1, hidden_dims=(3,), normalization="none")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 0.5]])
        targets = torch.tensor([[1.0], [-1.0], [0.5]])
        linear_before = model.hidden_blocks[0].linear.weight.detach().clone()
        theta_before = model.hidden_blocks[0].activation.theta.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

        assert not torch.equal(model.hidden_blocks[0].linear.weight, linear_before)
        assert not torch.equal(model.hidden_blocks[0].activation.theta, theta_before)

    def test_fixed_coefficients_stay_fixed_while_linear_parameters_change(self) -> None:
        """Fixed blends are buffers excluded from caller optimizer updates."""
        torch.manual_seed(4)
        model = ExpoMLP(
            2,
            1,
            hidden_dims=(3,),
            normalization="none",
            trainable_blend=False,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[0.5, 1.0], [1.0, 2.0], [1.5, 0.5]])
        targets = torch.tensor([[1.0], [-1.0], [0.5]])
        blend_before = model.hidden_blocks[0].activation.blend_weight.detach().clone()
        linear_before = model.hidden_blocks[0].linear.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()

        torch.testing.assert_close(
            model.hidden_blocks[0].activation.blend_weight, blend_before
        )
        assert not torch.equal(model.hidden_blocks[0].linear.weight, linear_before)
        assert model.hidden_blocks[0].activation.theta is None

    def test_to_moves_every_parameter_and_buffer_to_requested_dtype(self) -> None:
        """Parameters and fixed blend buffers follow ordinary .to behavior."""
        model = ExpoMLP(
            2,
            1,
            hidden_dims=(3,),
            normalization="layer",
            trainable_blend=False,
        ).to(device="cpu", dtype=torch.float64)
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
        assert all(buffer.device.type == "cpu" for buffer in model.buffers())
        assert all(buffer.dtype == torch.float64 for buffer in model.buffers())
        output = model(torch.randn(2, 2, dtype=torch.float64))
        assert output.dtype == torch.float64

    def test_strict_state_dict_round_trip_preserves_outputs_and_coefficients(
        self,
    ) -> None:
        """Strict loading restores ordinary state and activation values."""
        torch.manual_seed(5)
        source = ExpoMLP(3, 2, hidden_dims=(4, 5), blend_init=0.3)
        inputs = torch.randn(4, 3)
        source.eval()
        expected = source(inputs)
        expected_blends = [
            block.activation.blend_weight.detach().clone()
            for block in source.hidden_blocks
        ]

        torch.manual_seed(6)
        restored = ExpoMLP(3, 2, hidden_dims=(4, 5), blend_init=0.7)
        restored.eval()
        assert not torch.equal(
            restored.hidden_blocks[0].activation.blend_weight,
            source.hidden_blocks[0].activation.blend_weight,
        )
        restored.load_state_dict(source.state_dict(), strict=True)

        torch.testing.assert_close(restored(inputs), expected)
        for actual, expected_blend in zip(restored.hidden_blocks, expected_blends):
            torch.testing.assert_close(actual.activation.blend_weight, expected_blend)

    def test_wrong_shaped_and_incompatible_state_dicts_fail_clearly(self) -> None:
        """Strict state loading rejects wrong tensor shapes and architectures."""
        model = ExpoMLP(3, 2, hidden_dims=(4,))
        wrong_shape = model.state_dict()
        wrong_shape["output_layer.weight"] = torch.zeros(3, 4)
        with pytest.raises(RuntimeError, match="size mismatch for output_layer.weight"):
            model.load_state_dict(wrong_shape, strict=True)

        incompatible = ExpoMLP(3, 2, hidden_dims=(5,))
        with pytest.raises(RuntimeError, match="size mismatch"):
            incompatible.load_state_dict(model.state_dict(), strict=True)

    def test_eval_singleton_inference_matches_corresponding_batched_row(self) -> None:
        """LayerNorm gives the same eval result for one row and its batch row."""
        torch.manual_seed(7)
        model = ExpoMLP(3, 2, hidden_dims=(4, 5), normalization="layer").eval()
        batch = torch.randn(5, 3)
        with torch.inference_mode():
            batched_output = model(batch)
            singleton_output = model(batch[2:3])
        torch.testing.assert_close(singleton_output, batched_output[2:3])

    def test_public_import(self) -> None:
        """ExpoMLP is available from the documented top-level package surface."""
        from exponet import ExpoMLP as PublicExpoMLP

        assert PublicExpoMLP is ExpoMLP
