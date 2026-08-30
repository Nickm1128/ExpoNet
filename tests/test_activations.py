"""Tests for ExpoActivation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from exponet.activations import ExpoActivation


class TestExpoActivationForward:
    """Test forward pass values."""

    def test_per_neuron_exact_values(self) -> None:
        """Test exact endpoint and interior values."""
        x = torch.tensor([[-1.0, 0.0, 0.25, 1.0, 4.0]])
        act = ExpoActivation(num_features=5, blend_mode="per_neuron", blend_init=0.5)

        with torch.no_grad():
            out = act(x)

        u = torch.relu(x)
        a = torch.tensor([[0.5]])
        expected = u * ((1.0 - a) + a * u)

        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-8)

    def test_x4_a05_produces_10(self) -> None:
        """Test x=4, a=0.5 -> output=10 (documented value)."""
        x = torch.tensor([[4.0]])
        act = ExpoActivation(num_features=1, blend_mode="per_neuron", blend_init=0.5)
        with torch.no_grad():
            out = act(x)
        torch.testing.assert_close(out, torch.tensor([[10.0]]), rtol=1e-6, atol=1e-8)

    def test_fixed_a0_reproduces_relu(self) -> None:
        """Test a=0 (fixed) reproduces ReLU."""
        x = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]])
        act = ExpoActivation(
            num_features=5, blend_mode="per_neuron", blend_init=0.0, trainable=False
        )
        with torch.no_grad():
            out = act(x)
        expected = torch.relu(x)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-8)

    def test_fixed_a1_reproduces_squared_relu(self) -> None:
        """Test a=1 (fixed) reproduces squared ReLU."""
        x = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]])
        act = ExpoActivation(
            num_features=5, blend_mode="per_neuron", blend_init=1.0, trainable=False
        )
        with torch.no_grad():
            out = act(x)
        u = torch.relu(x)
        expected = u * u
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-8)

    def test_per_layer_mode(self) -> None:
        """Test per-layer mode broadcasts coefficient."""
        x = torch.randn(3, 4, 5)
        act = ExpoActivation(blend_mode="per_layer", blend_init=0.5)
        out = act(x)
        assert out.shape == x.shape

    def test_per_layer_1d_preserves_shape(self) -> None:
        """Test per-layer mode with 1D input preserves shape."""
        x = torch.randn(3)
        act = ExpoActivation(blend_mode="per_layer", blend_init=0.5)
        out = act(x)
        assert out.shape == x.shape
        assert out.ndim == 1

    def test_negative_inputs_zero_output(self) -> None:
        """Test negative inputs produce zero output."""
        x = torch.tensor([[-10.0, -1.0, -0.1]])
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-6, atol=1e-8)

    def test_zero_input_output_zero(self) -> None:
        """Test zero input produces zero output."""
        x = torch.zeros(2, 3)
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=1e-6, atol=1e-8)

    def test_non_last_feature_axis(self) -> None:
        """Test broadcasting with non-last feature dimension."""
        x = torch.randn(2, 5, 3)
        act = ExpoActivation(
            num_features=5, blend_mode="per_neuron", feature_dim=1, blend_init=0.5
        )
        out = act(x)
        assert out.shape == x.shape

    def test_batch_handling(self) -> None:
        """Test batched inputs preserve shape."""
        x = torch.randn(10, 8, 6)
        act = ExpoActivation(num_features=6, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        assert out.shape == x.shape


class TestExpoActivationGradients:
    """Test gradient computations."""

    def test_input_gradient_analytic(self) -> None:
        """Test input gradient matches analytic: (1-a)+2*a*x for x>0."""
        x = torch.tensor([[0.5, 1.5, 2.5]], requires_grad=True)
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()

        a = torch.sigmoid(act.theta).detach()
        x_val = x.detach()
        for i in range(3):
            if x_val[0, i] > 0:
                expected_grad = (1.0 - a[i]).item() + 2.0 * a[i].item() * x_val[
                    0, i
                ].item()
            else:
                expected_grad = 0.0
            torch.testing.assert_close(
                x.grad[0, i].item(), expected_grad, rtol=1e-5, atol=1e-6
            )

    def test_raw_coefficient_gradient_analytic(self) -> None:
        """Test theta gradient matches analytic: (u*u-u)*a*(1-a)."""
        x = torch.tensor([[0.5, 1.5, 2.5]])
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()

        u = torch.relu(x).detach()
        theta = act.theta
        a = torch.sigmoid(theta).detach()
        for i in range(3):
            expected_grad = (
                (u[0, i].item() ** 2 - u[0, i].item())
                * a[i].item()
                * (1.0 - a[i].item())
            )
            torch.testing.assert_close(
                theta.grad[i].item(), expected_grad, rtol=1e-5, atol=1e-6
            )

    def test_zero_input_gradient_zero(self) -> None:
        """Test gradients are zero at zero input."""
        x = torch.zeros(2, 3, requires_grad=True)
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()
        torch.testing.assert_close(
            x.grad, torch.zeros_like(x.grad), rtol=1e-6, atol=1e-8
        )

    def test_negative_input_gradient_zero(self) -> None:
        """Test gradients are zero for negative inputs."""
        x = torch.tensor([[-10.0, -1.0, -0.1]], requires_grad=True)
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()
        torch.testing.assert_close(
            x.grad, torch.zeros_like(x.grad), rtol=1e-6, atol=1e-8
        )

    def test_coefficient_gradient_analytic_zero(self) -> None:
        """Test coefficient gradients are zero for zero input."""
        x = torch.zeros(2, 3)
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            assert torch.all(act.theta.grad == 0) or act.theta.grad is None

    def test_coefficient_gradient_analytic_negative(self) -> None:
        """Test coefficient gradients are zero for negative inputs."""
        x = torch.tensor([[-10.0, -1.0, -0.1]])
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            assert torch.all(act.theta.grad == 0)

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gradcheck_input_coefficient(self, dtype: torch.dtype) -> None:
        """Verify finite gradients for input and theta using autograd.gradcheck."""
        act = ExpoActivation(
            num_features=3, blend_mode="per_neuron", blend_init=0.5
        ).to(dtype)

        def func(x_input, theta_input):
            return torch.func.functional_call(act, {"theta": theta_input}, (x_input,))

        x_val = torch.tensor([[0.5, 1.5, 2.5]], dtype=dtype, requires_grad=True)
        theta_val = torch.tensor([0.0, 0.5, -0.5], dtype=dtype, requires_grad=True)

        torch.autograd.gradcheck(func, (x_val, theta_val), nondet_tol=1e-5)


class TestExpoActivationConfiguration:
    """Test configuration validation and behavior."""

    def test_invalid_blend_mode_raises(self) -> None:
        """Test invalid blend_mode raises ValueError."""
        with pytest.raises(ValueError, match="blend_mode must be"):
            ExpoActivation(num_features=4, blend_mode="invalid")

    def test_per_neuron_requires_num_features(self) -> None:
        """Test per_neuron requires num_features."""
        with pytest.raises(ValueError, match="num_features is required"):
            ExpoActivation(blend_mode="per_neuron")

    def test_per_neuron_rejects_zero_features(self) -> None:
        """Test per_neuron rejects num_features <= 0."""
        with pytest.raises(ValueError, match="num_features must be a positive integer"):
            ExpoActivation(num_features=0, blend_mode="per_neuron")
        with pytest.raises(ValueError, match="num_features must be a positive integer"):
            ExpoActivation(num_features=-1, blend_mode="per_neuron")

    @pytest.mark.parametrize("num_features", [True, 3.5, "3"])
    def test_per_neuron_rejects_non_integer_features(self, num_features) -> None:
        """Test per_neuron requires an integer feature count, excluding bool."""
        with pytest.raises(ValueError, match="num_features must be a positive integer"):
            ExpoActivation(num_features=num_features, blend_mode="per_neuron")

    def test_per_layer_rejects_num_features(self) -> None:
        """Test per_layer rejects num_features."""
        with pytest.raises(ValueError, match="num_features must be None"):
            ExpoActivation(num_features=4, blend_mode="per_layer")

    @pytest.mark.parametrize("feature_dim", [True, 1.0, "1"])
    @pytest.mark.parametrize("blend_mode", ["per_neuron", "per_layer"])
    def test_rejects_non_integer_feature_dim(
        self, feature_dim, blend_mode: str
    ) -> None:
        """Test feature_dim is validated even when per-layer mode does not use it."""
        kwargs = {"blend_mode": blend_mode, "feature_dim": feature_dim}
        if blend_mode == "per_neuron":
            kwargs["num_features"] = 3
        with pytest.raises(ValueError, match="feature_dim must be an integer axis"):
            ExpoActivation(**kwargs)

    def test_trainable_rejects_endpoint_init(self) -> None:
        """Test trainable mode rejects blend_init at endpoints."""
        with pytest.raises(ValueError, match="blend_init must be in \\(0, 1\\)"):
            ExpoActivation(
                num_features=4, blend_mode="per_neuron", blend_init=0.0, trainable=True
            )
        with pytest.raises(ValueError, match="blend_init must be in \\(0, 1\\)"):
            ExpoActivation(
                num_features=4, blend_mode="per_neuron", blend_init=1.0, trainable=True
            )

    def test_fixed_mode_accepts_endpoint_init(self) -> None:
        """Test fixed mode accepts blend_init in [0, 1]."""
        act0 = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.0, trainable=False
        )
        act1 = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=1.0, trainable=False
        )
        assert torch.all(act0.blend_weight == 0.0)
        assert torch.all(act1.blend_weight == 1.0)

    def test_blend_weight_property(self) -> None:
        """Test blend_weight returns effective coefficient."""
        x = torch.randn(2, 4)
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.25)
        act(x)
        a = act.blend_weight
        assert a.shape == (4,)
        assert torch.all((a >= 0) & (a <= 1))

    def test_scalar_rejection_per_neuron(self) -> None:
        """Test scalar inputs are rejected."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        x = torch.tensor(1.5)
        with pytest.raises(ValueError, match="Scalar inputs are not supported"):
            act(x)

    def test_scalar_rejection_per_layer(self) -> None:
        """Test scalar inputs are rejected in per-layer mode."""
        act = ExpoActivation(blend_mode="per_layer", blend_init=0.5)
        x = torch.tensor(1.5)
        with pytest.raises(ValueError, match="Scalar inputs are not supported"):
            act(x)

    def test_feature_dim_mismatch_raises(self) -> None:
        """Test feature dimension mismatch raises ValueError."""
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        x = torch.randn(2, 1)
        with pytest.raises(ValueError, match="Input dimension.*size 1.*expected 3"):
            act(x)

    def test_negative_feature_dim_out_of_range_raises(self) -> None:
        """Test negative feature_dim out of range raises ValueError."""
        act = ExpoActivation(
            num_features=3, blend_mode="per_neuron", blend_init=0.5, feature_dim=-3
        )
        x = torch.randn(2, 3)
        with pytest.raises(ValueError, match="out of range"):
            act(x)

    def test_positive_feature_dim_valid(self) -> None:
        """Test valid positive feature_dim works."""
        x = torch.randn(2, 5, 3)
        act = ExpoActivation(
            num_features=5, blend_mode="per_neuron", blend_init=0.5, feature_dim=1
        )
        out = act(x)
        assert out.shape == x.shape

    def test_negative_feature_dim_valid(self) -> None:
        """Test valid negative feature_dim works."""
        x = torch.randn(2, 5, 3)
        act = ExpoActivation(
            num_features=5, blend_mode="per_neuron", blend_init=0.5, feature_dim=-2
        )
        out = act(x)
        assert out.shape == x.shape

    def test_noncontiguous_input(self) -> None:
        """Test noncontiguous inputs work correctly."""
        x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).permute(0, 2, 1)
        assert not x.is_contiguous()
        act = ExpoActivation(
            num_features=3, blend_mode="per_neuron", blend_init=0.5, feature_dim=1
        )
        out = act(x)
        assert out.shape == x.shape

    def test_singleton_dimension(self) -> None:
        """Test singleton dimensions are handled correctly."""
        x = torch.randn(2, 1)
        act = ExpoActivation(num_features=1, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        assert out.shape == x.shape

    def test_distinct_per_neuron_coefficients(self) -> None:
        """Test distinct per-neuron coefficients are applied to correct axis."""
        x = torch.tensor([[0.5, 2.0, 3.0], [4.0, 0.25, 1.5]])
        known_coefficients = torch.tensor([0.25, 0.5, 0.75])
        act = ExpoActivation(num_features=3, blend_mode="per_neuron", blend_init=0.5)
        with torch.no_grad():
            act.theta.copy_(torch.logit(known_coefficients))
        out = act(x)
        u = torch.relu(x)
        expected = u * ((1.0 - known_coefficients) + known_coefficients * u)
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-8)

    def test_trainable_coefficients_are_parameters(self) -> None:
        """Test trainable theta is registered as Parameter."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        params = {n for n, _ in act.named_parameters()}
        assert "theta" in params

    def test_fixed_coefficients_are_buffers(self) -> None:
        """Test fixed coefficient is registered as buffer."""
        act = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        buffers = {n for n, _ in act.named_buffers()}
        assert "a_fixed" in buffers

    def test_fixed_coefficient_unchanged_by_optimizer(self) -> None:
        """Test fixed coefficients remain unchanged after optimizer step."""

        class DummyModule(nn.Module):
            def __init__(self, act):
                super().__init__()
                self.act = act
                self.lin = nn.Linear(4, 4)

            def forward(self, x):
                return self.lin(self.act(x))

        x = torch.randn(10, 4)
        act = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        model = DummyModule(act)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        a_before = act.blend_weight.clone()
        linear_weight_before = model.lin.weight.detach().clone()
        optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        torch.testing.assert_close(act.blend_weight, a_before, rtol=1e-6, atol=1e-8)
        assert not torch.equal(model.lin.weight.detach(), linear_weight_before)
        assert len(list(act.parameters())) == 0


class TestExpoActivationDeviceAndDtype:
    """Test .to() behavior and dtype handling."""

    def test_cpu_float32(self) -> None:
        """Test CPU float32 execution."""
        x = torch.randn(2, 4, dtype=torch.float32)
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        out = act(x)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"

    def test_cuda_float32(self) -> None:
        """Test CUDA forward, backward, and theta update when available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this machine")
        x = torch.tensor(
            [[0.25, 0.5, 2.0, 3.0], [0.75, 1.5, 2.5, 4.0]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        act = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5
        ).to("cuda")
        optimizer = torch.optim.SGD([act.theta], lr=0.01)

        optimizer.zero_grad(set_to_none=True)
        out = act(x)
        assert out.dtype == torch.float32
        assert out.device.type == "cuda"
        assert torch.all(torch.isfinite(out))
        out.sum().backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))
        assert act.theta.grad is not None
        assert torch.all(torch.isfinite(act.theta.grad))

        with torch.no_grad():
            theta_before = act.theta.clone()
        optimizer.step()
        assert not torch.allclose(act.theta, theta_before)

    def test_float64_cpu(self) -> None:
        """Test CPU float64 execution for mathematical validation."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        act = act.to(torch.float64)
        x = torch.randn(2, 4, dtype=torch.float64)
        out = act(x)
        assert out.dtype == torch.float64
        assert out.device.type == "cpu"

    def test_device_move(self) -> None:
        """Test .to() moves all parameters and buffers."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        act.to("cpu")
        out = act(torch.randn(2, 4))
        assert out.device.type == "cpu"


class TestPublicImport:
    """Test public import of ExpoActivation."""

    def test_public_import(self) -> None:
        """Test ExpoActivation can be imported from exponet."""
        from exponet import ExpoActivation

        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        x = torch.randn(2, 4)
        out = act(x)
        assert out.shape == (2, 4)

    def test_version_available(self) -> None:
        """Test __version__ is available from exponet."""
        import exponet

        assert hasattr(exponet, "__version__")
        assert exponet.__version__ == "0.1.0"


class TestExpoActivationStateDict:
    """Test state_dict and loading."""

    def test_state_dict_round_trip_trainable(self) -> None:
        """Test state_dict round trip for trainable mode."""
        x = torch.randn(2, 4)
        act1 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        out1 = act1(x)
        state = act1.state_dict()
        act2 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        act2.load_state_dict(state)
        out2 = act2(x)
        torch.testing.assert_close(out2, out1, rtol=1e-6, atol=1e-8)

    def test_state_dict_round_trip_fixed(self) -> None:
        """Test state_dict round trip for fixed mode."""
        x = torch.randn(2, 4)
        act1 = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.75, trainable=False
        )
        out1 = act1(x)
        state = act1.state_dict()
        act2 = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.25, trainable=False
        )
        assert not torch.equal(act2.blend_weight, act1.blend_weight)
        act2.load_state_dict(state)
        out2 = act2(x)
        torch.testing.assert_close(out2, out1, rtol=1e-6, atol=1e-8)

    def test_strict_load_rejects_unexpected_keys(self) -> None:
        """Test strict=True rejects unexpected keys."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        state = act.state_dict()
        state["unexpected_key"] = torch.tensor([0.5])
        act2 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            act2.load_state_dict(state, strict=True)

    def test_strict_load_rejects_missing_keys(self) -> None:
        """Test strict=True rejects missing keys."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        state = act.state_dict()
        del state["theta"]
        act2 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        with pytest.raises(RuntimeError, match="Missing key"):
            act2.load_state_dict(state, strict=True)

    def test_strict_load_rejects_wrong_trainable_shape(self) -> None:
        """Test strict loading rejects a wrong-shaped theta tensor."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        with pytest.raises(RuntimeError, match="size mismatch for theta"):
            act.load_state_dict({"theta": torch.zeros(3)}, strict=True)

    def test_strict_load_rejects_wrong_fixed_shape(self) -> None:
        """Test strict loading rejects a wrong-shaped fixed coefficient tensor."""
        act = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        with pytest.raises(RuntimeError, match="size mismatch for a_fixed"):
            act.load_state_dict({"a_fixed": torch.zeros(3)}, strict=True)

    def test_trainable_to_fixed_rejects(self) -> None:
        """Test loading trainable state into fixed mode raises."""
        act_trainable = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5
        )
        state = act_trainable.state_dict()
        act_fixed = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        with pytest.raises(RuntimeError):
            act_fixed.load_state_dict(state, strict=True)

    def test_fixed_to_trainable_rejects(self) -> None:
        """Test loading fixed state into trainable mode raises."""
        act_fixed = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        state = act_fixed.state_dict()
        act_trainable = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5
        )
        with pytest.raises(RuntimeError):
            act_trainable.load_state_dict(state, strict=True)

    def test_strict_round_trip_different_initial_coefficients(self) -> None:
        """Test round trip with different initial coefficients."""
        x = torch.randn(2, 4)
        act1 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.3)
        with torch.no_grad():
            act1.theta.copy_(torch.tensor([0.2, -0.1, 0.5, 0.3]))
        out1 = act1(x)
        state = act1.state_dict()
        act2 = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.7)
        act2.load_state_dict(state)
        out2 = act2(x)
        torch.testing.assert_close(out2, out1, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(
            act2.blend_weight, act1.blend_weight, rtol=1e-6, atol=1e-8
        )

    def test_state_dict_contains_expected_keys_trainable(self) -> None:
        """Test state_dict keys for trainable mode."""
        act = ExpoActivation(num_features=4, blend_mode="per_neuron", blend_init=0.5)
        state = act.state_dict()
        assert set(state.keys()) == {"theta"}

    def test_state_dict_contains_expected_keys_fixed(self) -> None:
        """Test state_dict keys for fixed mode."""
        act = ExpoActivation(
            num_features=4, blend_mode="per_neuron", blend_init=0.5, trainable=False
        )
        state = act.state_dict()
        assert set(state.keys()) == {"a_fixed"}


class TestIsolatedRecovery:
    """Test isolated coefficient recovery experiment."""

    def test_recovery_simple(self) -> None:
        """Test recovery of known blend with positive inputs."""
        torch.manual_seed(42)
        n = 1000
        x = torch.linspace(0.1, 3.0, n).unsqueeze(-1)
        target_coefficient = torch.tensor([0.7])
        convergence_tolerance = 0.01
        minimum_movement = 0.15
        y = (1.0 - target_coefficient) * x + target_coefficient * x * x

        act = ExpoActivation(num_features=1, blend_mode="per_neuron", blend_init=0.5)
        optimizer = torch.optim.SGD((act.theta,), lr=0.5)
        initial_coefficient = act.blend_weight.detach().clone()
        assert torch.abs(initial_coefficient - target_coefficient).item() > 0.1

        for _ in range(500):
            optimizer.zero_grad(set_to_none=True)
            out = act(x)
            loss = ((out - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_coefficient = act.blend_weight.detach()
        movement = torch.abs(final_coefficient - initial_coefficient).item()
        assert movement >= minimum_movement
        torch.testing.assert_close(
            final_coefficient,
            target_coefficient,
            rtol=0.0,
            atol=convergence_tolerance,
        )
