# Mathematical and architectural design

Status: blend equation and per-neuron coefficients accepted; mapping, defaults, and remaining scope are proposed. Approval is tracked in [DECISIONS.md](DECISIONS.md).

## 1. Objective and boundaries

Test whether learning a per-neuron blend between ReLU and squared ReLU provides useful adaptive curvature at a small parameter and computation cost. The name ExpoNet is retained, but the original continuous-power equation is superseded. Do not implement an exponent mode or present blend weights as exponents.

Keep a reusable PyTorch activation, one dense MLP, one shared supervised training loop, and two estimator wrappers. Exclude sine activations, learned sparse expanders, spectral gates, recurrent state, episodic objectives, attention, architecture registries, and serving infrastructure.

## 2. Activation and gradients

For finite real input `x`:

```text
u = max(x, 0)
f(x; a) = (1 - a) * u + a * u * u
0 <= a <= 1

df/dx = (1 - a) + 2 * a * x    for x > 0
df/da = u * u - u              for all finite x
```

Specify the implemented input derivative as zero on `x <= 0`, using ReLU's origin convention. The coefficient derivative is also zero there. For `a < 1`, the right input derivative at zero is `1-a`, so the activation retains a corner; for `a=1`, it is squared ReLU and the input derivative is continuous at zero.

Consequences derived directly from these equations:

- Fixed `a=0` is ReLU; fixed `a=1` is squared ReLU.
- Increasing `a` suppresses values between 0 and 1 and amplifies values above 1. At each input the blend lies between the two endpoint outputs.
- At `x=1`, the coefficient has no local gradient. Negative or zero inputs provide no coefficient gradient either.
- Negative units retain ReLU's inactive-region limitation; the blend does not fix it.
- For positive inputs the second derivative with respect to input is the bounded constant `2*a`. The origin still requires the convention above; do not advertise global smoothness.
- For every fixed `a>0`, growth is asymptotically quadratic. This is a mixture of a linear and a quadratic term, not a continuously varying growth exponent.
- Bounding `a` does not bound the output. Large positive activations and composition across layers can still overflow.

Example values:

| x | a=0 | a=0.5 | a=1 |
| --- | --- | --- | --- |
| -1 | 0 | 0 | 0 |
| 0.25 | 0.25 | 0.15625 | 0.0625 |
| 1 | 1 | 1 | 1 |
| 4 | 4 | 10 | 16 |

### Constrained learning

Propose an unconstrained PyTorch parameter `theta` for each learned coefficient:

```text
a = sigmoid(theta)
theta_init = log(blend_init / (1 - blend_init))
da/dtheta = a * (1 - a)
df/dtheta = (u * u - u) * a * (1 - a)
```

Under this proposed mapping, trainable initialization must satisfy `0 < blend_init < 1`; reject endpoints instead of silently shifting them. The default proposal is `blend_init=0.5`, giving `theta_init=0`. Finite precision can round extreme sigmoid results to endpoints; promise bounded values in `[0, 1]`, not strict floating-point interior membership. Large raw magnitudes can saturate gradients; record this in diagnostics.

For fixed blends, store `a` as a registered buffer and permit `[0, 1]` inclusively. This makes exact endpoint comparisons possible without infinite raw parameters. Fixed mode has no trainable coefficient parameter.

Confirmed primary behavior is per neuron, with a vector matching the hidden width. The proposed per-layer experimental control uses one scalar per activation module. Create a new activation instance in each hidden layer; reusing one instance would unintentionally tie parameters. Do not use a separate coefficient per example or batch position.

### Computational cost

The proposed sigmoid constraint and the activation operate at different scales. For hidden width `H` and batch size `B`, compute sigmoid on the `H` raw parameters before broadcasting; apply the rectifier and polynomial arithmetic to `B * H` values. For example, `H=256` and `B=128` means 256 sigmoid evaluations and 32,768 polynomial evaluations per layer forward. This is an element-count comparison, not a measured runtime ratio. Small GPU kernel launches and temporary tensors can still matter.

The activation-value path and its input/coefficient derivatives require no general power, logarithm, exponential, or sine. The proposed sigmoid mapping still uses an exponential on the small parameter vector; this is not a promise to remove transcendental operations from the entire model. See the [PyTorch sigmoid kernel](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu). Frozen inference could precompute effective coefficients, but no inference cache is part of the current contract.

Dense matrix multiplication performs different amounts of work and uses different kernels. Neither operation counts alone nor scalar-operation latency determines whole-model speed. Profile activation forward/backward, complete blocks including normalization, and end-to-end training before making speed claims against ReLU or sine.

### Reference evaluation and numerical limits

Compute ReLU once and use the factored polynomial, which avoids forming an unweighted square as an intermediate:

```python
u = torch.relu(x)
output = u * ((1 - a) + a * u)
```

The factored and expanded forms are algebraically equivalent; floating-point order can change rounding. For finite inputs, fixed `a=0` reduces to ReLU without computing `u*u`, and fixed `a=1` reduces to squared ReLU. Use explicit multiplication, not a general `torch.pow` call. No positive-input floor, additive epsilon, clipping, absolute value, leaky negative branch, or custom backward is needed. Test the ReLU origin convention explicitly.

This is a design reference, not tested ExpoNet code. Validate forward values and gradients of input and raw coefficient before adoption. Ordinary floating-point underflow/overflow remains possible, including when a gradient is larger than a finite forward value. The estimator rejects nonfinite inputs and fails on nonfinite training state; the standalone module's contract is for finite inputs and does not hide invalid values through repair.

## 3. Scaling and hidden normalization

Keep these separate:

1. **Feature standardization:** fit per-column mean/scale using only the training partition. Reuse it for validation and prediction. Optional and disabled when the caller already owns preprocessing.
2. **Hidden normalization:** optionally normalize each sample's hidden features after a linear layer and before the blend.
3. **Target standardization:** optional for regression only; inverse-transform predictions. Default off.

Proposed hidden block:

```text
Linear -> LayerNorm or Identity -> ExpoActivation
```

Final layer: `Linear`, without the blend or hidden normalization. Regression outputs remain signed and unbounded; classification outputs are logits.

Use built-in `nn.LayerNorm(width, eps=1e-5, elementwise_affine=True)`. It computes statistics over each sample's hidden dimension in both training and evaluation, so singleton prediction batches are usable. [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html).

LayerNorm is not a hard magnitude bound: its learned affine parameters can grow. It also removes per-sample location and much of the scale before its affine transform. Width one degenerates to an input-independent normalized value; reject hidden widths of one when LayerNorm is selected. Very narrow layers need particular scrutiny. Test both normalized and unnormalized models, including tasks where input magnitude is informative.

Do not place LayerNorm directly on raw features by default. Feature standardization acts across training examples and has different information effects. Do not add post-activation normalization or a learnable activation gain without a new decision.

## 4. Initialization and optimization

- Hidden linear weights: start with Xavier uniform, biases zero. Output weights: Xavier uniform, bias zero. This is a reproducible starting rule, not a variance-preservation theorem for the blend activation.
- LayerNorm starts with scale one and bias zero. Blend coefficients start uniformly at the configured `blend_init`; random linear weights already break neuron symmetry.
- AdamW is the single initial optimizer. Default learning rate `1e-3`, default weight decay zero.
- Use separate parameter groups for linear weight matrices and all other parameters. Apply configured weight decay only to linear weight matrices. Do not decay raw blend parameters, normalization parameters, or biases.
- The blend learning rate defaults to the main learning rate and can be set separately. Decaying `theta` would bias `a` toward 0.5, which is an unintended extra assumption.
- Optional global gradient-norm clipping is applied after backward and finite-gradient checks, before the optimizer step. Clipping cannot repair forward overflow.
- Fail if the output, loss, gradient, or updated parameter becomes nonfinite. Do not silently replace values or skip unsuccessful batches.
- Validate CPU/CUDA float32 before considering reduced precision. No custom autograd function or custom CUDA kernel is needed for the initial version.

## 5. Proposed package boundaries

```text
src/exponet/
  __init__.py       # small public export list
  activations.py    # coefficient mapping, broadcasting, blend module
  nn.py             # dense model composition
  _training.py      # one minibatch loop and early stopping
  _validation.py    # estimator data/config/device validation
  estimators.py     # regressor/classifier and fitted-state ownership
  _persistence.py   # versioned inference snapshot
tests/             # unit and integration checks
examples/          # short regression/classification/direct-Torch usage
benchmarks/        # small controlled experiment runner
docs/              # contracts, roadmap, reviewed summaries
```

The Torch modules do not import estimator or persistence internals. Estimators compose modules and the shared trainer. Use scikit-learn's StandardScaler rather than copying PSANN's streaming scaler framework. Split a module only when responsibilities actually warrant it; do not pre-create a plugin system or a forest of empty files.

No runtime import from PSANN, no installation dependency on PSANN, and no serialized PSANN class aliases. See [PSANN_REUSE.md](PSANN_REUSE.md) for the source adaptation policy.
