# Correctness and experiment plan

Status: planned checks, not executed library validation. The [roadmap](ROADMAP.md) records completion evidence.

## 1. Activation tests

Use a focused `tests/test_activations.py` rather than separate files for every edge case.

- Analytic values at negative inputs, zero, fractions, one, and values above one. Fixed endpoints must reproduce ReLU and squared ReLU forward values and the documented origin derivative.
- Analytical first derivatives: `(1-a)+2*a*x` with respect to positive input and `(u*u-u)*a*(1-a)` with respect to raw `theta` under the proposed sigmoid mapping. Test `0 < x < 1` and `x > 1` separately so coefficient-gradient signs cannot cancel unnoticed.
- Zero/negative inputs produce exact zero outputs and zero input/coefficient gradients. Include all-negative and all-zero batches. For an interior blend the right derivative at zero is nonzero, but the implemented derivative at zero follows ReLU's zero convention.
- CPU float64 `torch.autograd.gradcheck` for input and raw coefficient jointly, with inputs bounded away from zero. Test the origin convention separately; do not finite-difference across the ReLU corner and call a failure a bug.
- Tiny positive and moderately large positive inputs, several interior coefficients, and extreme finite raw parameters. Require finite results where the reference is representable. An intentionally overflowing case tests estimator error behavior, not a false promise of unlimited dynamic range.
- A hand-calculated interior fixture distinguishes the accepted blend from the superseded equation: `x=4, a=0.5` must produce 10. Compare factored and expanded polynomial forms on modest inputs within floating-point tolerances. Fixed `a=0` should retain a finite large positive ReLU output without first forming an overflowing square.
- Per-neuron and per-layer broadcasting, non-last feature axis, invalid feature sizes/axes, noncontiguous inputs, singleton dimensions, shape/dtype preservation, and no input mutation.
- Every learned raw coefficient is registered as a Parameter; every fixed coefficient is a buffer. Fixed coefficients remain unchanged after optimizer steps. No accidental parameter sharing between hidden layers.
- Isolated recovery: generate `y=0.3*x+0.7*x*x` on positive inputs spanning below/above one, start at `a=0.5`, and optimize only `theta`. Predeclare a tolerance and bounded step budget after a calibration run, then lock the deterministic test. This proves coefficient learnability without allowing other weights to compensate.

Starting tolerances for modest-scale fixtures: float64 analytic checks `rtol=1e-6, atol=1e-8`; float32 `rtol=1e-5, atol=1e-6`. Record justified adjustments, especially across devices; do not loosen thresholds simply to hide failures.

## 2. Model and training tests

Extend cohesive `test_nn.py`, `test_training.py`, and `test_estimators.py` modules as behavior grows.

- Exact hidden ordering and a linear output head; negative regression predictions remain possible. LayerNorm-off path and its allowed width-one case are covered.
- Each trainable parameter appears once in optimizer groups, with no blend/bias/normalization decay. Both backbone and raw blend parameters change on a suitable training fixture.
- Shape errors fail before a misleading broadcast loss. Reject nonfinite data before and after float32 conversion and preprocessing.
- A validation-only outlier cannot alter fitted feature or target scaler statistics. Constant feature/target columns remain finite. Prediction never refits a scaler.
- Mini-batch remainder retention, sample-weighted epoch aggregation, validation without gradients, clipping order, and nonfinite failures before unsafe continuation.
- Early stopping first result, ties, `min_delta`, patience boundary, epoch-limit best-state restoration, deep-copy behavior, and complete parameter/buffer restoration. No validation means no early stopping.
- Repeated fit reset, failed-fit invalidation, `set_params` invalidation, before-fit errors, target rank preservation, multioutput scores, class ordering, and invalid label rejection.
- CPU seeded reproducibility within declared tolerances; process-global RNG state is not unexpectedly left changed. Test independent estimator instances and serial model selection.
- Inference singleton/batch equivalence and order preservation; no training-mode mutation during validation/prediction.
- Persistence tests include all fitted preprocessing, effective blend coefficients, labels, history, shapes, restricted loading, and failed-write preservation.

Use real scikit-learn cloning, Pipeline, and GridSearchCV. Run the applicable official estimator checks against pinned supported versions and document limitations honestly. Do not catch all check failures or broadly mark them expected.

## 3. CPU and CUDA evidence

CPU correctness is always required. GPU tests may be marked/skipped on CPU hosts, but a CUDA support claim requires a recent run on actual CUDA hardware.

Record Python, PyTorch build, NumPy, scikit-learn, OS, GPU model, CUDA runtime reported by Torch, driver information where available, and test commands. Never confuse installed CUDA tooling with a successfully exercised CUDA training path.

Required CUDA evidence: activation forward/backward, isolated coefficient update, regression fit, classification fit/probabilities, CPU/GPU inference parity from identical state, and GPU-trained snapshot loaded on CPU. Compare inference values with tolerances, not bitwise training equality. The test runner should assert the resolved device so silent CPU execution cannot pass a GPU check.

Time CUDA work with explicit synchronization around measured sections and a warmup. Report end-to-end training separately from any isolated forward/backward timing. Run GPU cross-validation serially to avoid accidental memory competition.

### Early activation timing protocol

Before the full training benchmark, time native ReLU, `u*u` with `u=relu(x)` computed once, plain sine, and the accepted learned blend. Sine is a timing reference only, not another supported ExpoNet activation or a recreation of the complete PSANN activation. Use matched shapes/dtypes and include batch sizes 1 and 128 and hidden widths 64 and 256. Record forward-only inference separately from forward plus backward, including gradients of learned coefficients where applicable. Do not introduce a supported power-activation mode for benchmarking.

Measure the sigmoid parameter mapping separately and inside the complete learned activation. Compare a minimal dense block with and without normalization as well as isolated activations. Warm up, synchronize CUDA timings, repeat enough iterations to report a median and variability, and clear gradients consistently between training measurements. Keep setup, random input creation, and device transfers outside isolated kernel timing; report any end-to-end measurement separately. No timing result or accepted overhead budget exists yet.

## 4. Main research question

Does learning the linear/quadratic blend improve held-out predictions relative to a fixed activation under comparable capacity and optimization budgets? A changed coefficient or lower training loss alone does not establish usefulness.

### Required comparison matrix

| Variant | Blend coefficient | Purpose |
| --- | --- | --- |
| Native PyTorch ReLU | Equivalent to a=0 | Conventional accuracy and realistic runtime baseline. |
| Native squared ReLU (`u*u`) | Equivalent to a=1 | Quadratic endpoint accuracy and runtime baseline. |
| Fixed ExpoActivation | a=0 | Numerical/architecture equivalence check against ReLU; not the primary runtime baseline. |
| Fixed ExpoActivation | a=0.5 | Isolate the benefit of learning from the benefit of choosing a different fixed curve. |
| Fixed ExpoActivation | a=1 | Numerical/architecture equivalence check against squared ReLU. |
| Learned per layer | Initial a=0.5 | Proposed experimental control with fewer adaptive coefficients. |
| Learned per neuron | Initial a=0.5 | Test the accepted primary granularity. |

Run every main variant with both no hidden normalization and LayerNorm. Keep feature standardization identical within each comparison. Avoid a full Cartesian sweep of all future knobs; only after this matrix, test initial coefficients 0.1/0.9 and blend learning-rate sensitivity on a limited selection of tasks. The per-layer row remains contingent on approving that experimental control.

### Workloads

1. Isolated known-blend recovery, as a mathematical control rather than an accuracy benchmark.
2. Synthetic regression with linear, rectified linear/quadratic blends, and mixed smooth targets, including noise. Include tasks where magnitude matters and where the target is not specifically generated by the accepted activation.
3. Synthetic binary/multiclass classification to exercise task contracts.
4. At least one real numeric regression dataset and one real numeric classification dataset. Choose and record dataset source, license, version/checksum, preprocessing, and splits before running the comparison. Small bundled datasets can keep the first run inexpensive.

Time-series models and large external dataset pipelines are not needed to answer the initial question.

### Fairness and reporting

- Reuse exact train/validation/test partitions and paired random seeds across variants. Keep test data out of hyperparameter and default selection.
- Match widths, depths, ordinary-weight initialization, minibatches, maximum epochs, optimizer settings, preprocessing, early-stopping rule, and main learning rate in the primary activation ablation. Record the extra blend parameter count rather than pretending all capacities are identical.
- Initialize ordinary weights independently of activation parameter allocation so adding coefficients cannot silently change the baseline weights. Native ReLU uses the same blocks and normalization.
- Equal epoch/step limits are not equal wall time; report both actual steps and elapsed time. If tuning is added, provide comparable search budgets to baselines, including their initialization choices, and keep that analysis separate from the controlled ablation.
- Use at least five seeds for reported comparisons. Report per-seed results, mean, standard deviation, and paired metric differences. Small samples do not support sweeping statistical claims.
- Regression: RMSE, MAE, and R2 in original target units. Classification: accuracy and log loss. Report time, parameter count, and CUDA peak memory where available.
- Record per-layer blend coefficient min/mean/max and fraction within 0.01 of 0 or 1. For diagnostic runs, also inspect positive-activation fraction, activation magnitude quantiles, and gradient norms. Low activity or saturation can explain a coefficient that barely moves.
- Report all nonfinite failures and completed/attempted seed counts, not just successful averages. Accuracy improvements that rely on unstable runs or much longer training require that context.
- Learned blend coefficients depend on scaling and surrounding weights. They are activation-shape measurements, not direct input-feature importance or proof that the fitted coefficient is uniquely identifiable.

## 5. Outputs and completion

One runner under `benchmarks/` writes raw configurations, results, and logs to ignored `reports/` or `runs/`. Promote only a compact reviewed Markdown summary under `docs/` with reproduction commands and data provenance. Do not copy PSANN's generated datasets or experiment trees.

Separate three conclusions: **implementation correctness**, **numerical stability in tested settings**, and **empirical predictive benefit**. The first release can be useful even if the learned activation does not beat ReLU. A result should lead to a documented next decision, not automatic growth of the library.
