# ExpoNet

ExpoNet is a compact PyTorch library for dense numeric models with a trainable
blend of ReLU and squared ReLU:

```text
u = max(x, 0)
f(x; a) = (1 - a) * u + a * u * u
0 <= a <= 1
```

The primary activation uses one learned coefficient per hidden neuron. It uses
multiplication and addition on activation values, not a general power. ExpoNet
also provides scikit-learn-compatible dense regression and classification
estimators with CPU or a single CUDA device.

## Install

ExpoNet requires Python 3.11 or later, PyTorch, NumPy, and scikit-learn.
Install the PyTorch build appropriate for your platform and GPU first, using
the [official PyTorch selector](https://pytorch.org/get-started/locally/), then
install this checkout:

```powershell
python -m pip install .
```

For a checkout, install the project and its development tools with:

```powershell
python -m pip install -e ".[dev]"
```

Package publication is not part of this release-readiness milestone; install a
built wheel locally with `python -m pip install dist/exponet-0.1.0-py3-none-any.whl`.

`device="auto"` uses CUDA when `torch.cuda.is_available()` is true, otherwise
CPU. `device="cuda"` or `device="cuda:N"` requires that device and raises if
it is unavailable; ExpoNet never silently falls back to CPU. A CUDA toolkit
compiler (`nvcc`) is not required for the prebuilt-PyTorch path.

Verified release-candidate coverage is Windows 11 CPU and CUDA on an NVIDIA
GeForce RTX 5060 (PyTorch 2.11.0+cu128, CUDA runtime 12.8, Python 3.12.10,
NumPy 2.5.2, scikit-learn 1.9.0). Linux has not yet been validated, so it is
not claimed as supported coverage.

## Quick start

```python
import numpy as np
from exponet import ExpoRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 2)).astype(np.float32)
y = (1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.2).astype(np.float32)

model = ExpoRegressor(
    hidden_dims=(8,),
    normalization="none",
    trainable_blend=False,
    blend_init=0.0,  # ReLU control
    epochs=80,
    lr=0.03,
    device="auto",
    random_state=0,
).fit(X, y)

predictions = model.predict(X[:3])  # shape: (3,)
print(model.device_, predictions)
```

The same verified examples are available for [direct PyTorch use](examples/direct_torch.py),
[regression](examples/regression.py), and [multiclass classification](examples/classification.py).

## Public API

| Export | Purpose | Output shape |
| --- | --- | --- |
| `ExpoActivation` | Reusable PyTorch activation. | Same as input tensor. |
| `ExpoMLP` | Dense `Linear -> optional LayerNorm -> ExpoActivation` blocks and a linear readout. | `(batch, out_features)` |
| `ExpoRegressor` | Dense numeric regression estimator. | `predict`: `(N,)` for 1-D targets, otherwise `(N, K)` |
| `ExpoClassifier` | Binary/multiclass integer or string-label estimator. | `predict`: `(N,)`; `predict_proba`: `(N, K)` in `classes_` order |

Both estimators accept dense finite numeric features shaped `(N, F)`, train in
float32, provide `fit`, `predict`, `score`, `get_blend_weights`, and restricted
inference `save`/`load` methods. See the [API contract](docs/API_CONTRACT.md)
for constructor parameters, validation, scaling, early stopping, persistence,
and reproducibility semantics.

## Scope and limitations

- This is a dense numeric-data library. Sparse inputs, missing values,
  categorical encoding, image/sequence architectures, sample/class weights,
  streaming, warm starts, AMP, compilation, distributed execution, and custom
  training hooks are outside this release.
- Classification accepts homogeneous integer or string labels only; continuous,
  mixed, multilabel, and multioutput labels are rejected.
- Float32 is the primary runtime dtype. CPU float64 is supported for low-level
  activation mathematics tests. Very large positive activations can overflow,
  particularly toward the squared-ReLU endpoint.
- Saved snapshots restore inference state only, not optimizer or RNG state; do
  not treat a loaded estimator as an exact training-resume checkpoint.
- The initial controlled evaluation did not show a consistent predictive
  advantage over native ReLU, so no superiority or universal-normalization
  claim is made. See the [initial evaluation](docs/INITIAL_EVALUATION.md).

## Development validation

```powershell
python -B -m ruff format --check --no-cache --no-respect-gitignore src tests benchmarks examples
python -B -m ruff check --no-cache --no-respect-gitignore src tests benchmarks examples
python -B -m pytest -q -p no:cacheprovider
```

The [roadmap](docs/ROADMAP.md) records exact validation evidence and release
scope. ExpoNet is available under the [MIT License](LICENSE).

## Design and evidence

| Document | Purpose |
| --- | --- |
| [API contract](docs/API_CONTRACT.md) | Implemented interfaces and exact behavior. |
| [Design](docs/DESIGN.md) | Mathematics, numerical behavior, normalization, and architecture. |
| [Decisions](docs/DECISIONS.md) | Accepted scope and defaults. |
| [Validation](docs/VALIDATION.md) | Correctness checks and experimental criteria. |
| [Initial evaluation](docs/INITIAL_EVALUATION.md) | P6 five-seed activation comparison. |
| [Activation timing](docs/ACTIVATION_BENCHMARK.md) | P1.04 CPU timing evidence and limitations. |
| [PSANN reuse assessment](docs/PSANN_REUSE.md) | Selective adaptation provenance; no PSANN runtime dependency. |
| [Roadmap](docs/ROADMAP.md) | Completed work, release evidence, and deferred scope. |
