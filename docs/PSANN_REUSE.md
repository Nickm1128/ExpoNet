# PSANN source review and reuse plan

Review date: 2026-08-28. Licensing status updated: 2026-08-30.

Source checkout: `C:/Users/milin/Documents/psann`.
Source repository: [Nickm1128/psann](https://github.com/Nickm1128/psann).
Observed HEAD: `5878d66a498e9c4ac3e417f620276826977a72d3`.

The source working tree had existing modifications to `src/psann/_version.py` and `psannlm/_version.py`. Those files were not modified or used as a version baseline for ExpoNet. This review is not a certification of all PSANN implementations or results.

## What PSANN sets out to achieve

PSANN combines learnable sine-based activations with PyTorch networks exposed through scikit-learn style estimators. Its original activation learns amplitude, frequency, and damping, generally per output feature. It explores whether trainable activation behavior and specialized architectures improve function approximation and supervised prediction.

The maintained project now extends far beyond that original experiment: dense/residual/convolutional and spectral architectures, learned sparse expansion, episodic training, streaming state, multiple task adapters, model artifacts, deployment, and explanations. Its implementation and compatibility obligations reflect that breadth.

ExpoNet should preserve the convenient route from arrays to a trained PyTorch model and the useful correctness checks. It should not inherit the larger architecture inventory or the public API compatibility burden. Historical PSANN benchmark results do not establish that ExpoNet will improve on ReLU.

## Documents reviewed

Paths below are relative to the source repository, not ExpoNet:

- `README.md`, `docs/PROJECT_MAP.md`, and `docs/REPO_STRUCTURE.md`: current scope, supported surfaces, and output conventions.
- `docs/architecture.md` and `docs/training_core.md`: estimator flow, validation, shared optimization, device behavior, and checkpoint responsibilities.
- `docs/public_api.md`, `docs/quickstarts/regression.md`, and `docs/quickstarts/classification.md`: user-facing contracts and preprocessing/task semantics.
- `TECHNICAL_DETAILS.md`: activation mathematics and original model/scaling organization; some descriptions predate the maintained training contract.
- `docs/PSANN_Results_Compendium.md` and the explicitly historical `docs/ResearchFindings_and_NextSteps.md`: prior experimental aims and lessons about controlled comparisons.
- `pyproject.toml` and `LICENSE`: package/dependency boundaries and source notice.

The selected source inspection covered activation parameter registration, the estimator facade, scaling mixins, training configuration, fit helpers/contracts, and representative activation tests. It was targeted, not a line-by-line audit of the whole repository.

## Reuse inventory

| PSANN source area | Useful part | ExpoNet treatment |
| --- | --- | --- |
| `src/psann/activations.py` | Registered constrained parameters and feature-axis broadcasting. | Write the ReLU/squared-ReLU blend independently; adapt a small broadcasting/validation helper only if it is simpler than new code. Do not carry sine/decay configuration. |
| `src/psann/training.py` | Shared mini-batch loop, finite-state checks, history, and best-state restoration ideas. | Extract/rewrite a compact supervised loop; omit callbacks, compile/AMP fallbacks, runtime state, schedules, and resume machinery. |
| `src/psann/estimators/_fit_contracts.py` | Device validation, positive hyperparameter checks, exact target/output shapes. | Adapt narrowly scoped checks with new tests; no permissive fallback policies or variant signatures. |
| `src/psann/estimators/_fit_inputs.py` and `_fit_utils.py` | Split between data preparation, model construction, and optimization. | Use as architectural reference; these helpers contain unrelated hooks and should not be copied wholesale. |
| `src/psann/_sklearn/scaling.py` | Persist preprocessing consistently across training and prediction. | Use StandardScaler directly; avoid legacy aliases, custom scaler protocols, and streaming updates. |
| `src/psann/_sklearn/base.py` and `classifier.py` | Estimator lifecycle and label/probability handling. | Implement small wrappers around one common trainer. Inspect specific source paths again before adapting behavior. |
| `src/psann/sklearn.py` | Convenient public names. | New names only; do not copy serialization compatibility aliases or rewrite class module identities. |
| `tests/test_activation.py` | Shape, initialization, and parameter-gradient test patterns. | Adapt relevant patterns and add blend-specific mathematical tests. Do not import PSANN or its add-on package. |
| `tests/test_training_contracts.py`, `test_training_loop.py`, `test_device_dtype_policy.py` | Candidate regression cases for errors, training, and device policy. | Review and port only tests for selected ExpoNet contracts; test filenames alone are not evidence of behavior. |
| `training_checkpoint.py` / `platform/` | Separation of inference artifacts and resumable state. | Keep the distinction but design a much smaller inference-only format. Do not copy the platform. |
| Generated runs, datasets, release tooling, compatibility snapshots | Historical context. | Leave in PSANN; no bulk copy. |

## Adaptation procedure

1. Select an approved roadmap task and the smallest relevant source function/test.
2. Record its source commit and source/destination paths in a short entry below.
3. Verify source licensing and preserve required notices when copying substantive code. The inspected source includes an MIT notice naming Nicholas Milinkovich, 2025. ExpoNet is separately licensed under MIT with `Copyright (c) 2026 ExpoNet Contributors`.
4. Remove unrelated dependencies and legacy behavior. Do not create a runtime bridge to the source checkout.
5. Add tests for the intended ExpoNet contract, including any intentional difference from PSANN.
6. Run the focused tests and the relevant integration checks, then record results in the roadmap.

## Actual source imports

None. ExpoActivation was implemented independently without importing or depending on PSANN. Record future adaptations here, including source revision, destination, retained notice, and validating tests.
