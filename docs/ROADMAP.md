# ExpoNet implementation roadmap

Updated: 2026-08-30. Current stage: **P7 first release readiness complete; publication is intentionally out of scope**.

This is the single progress tracker. [DECISIONS.md](DECISIONS.md) owns accepted choices, [DESIGN.md](DESIGN.md) owns mathematical behavior, [API_CONTRACT.md](API_CONTRACT.md) owns interfaces, and [VALIDATION.md](VALIDATION.md) owns evaluation expectations.

ExpoActivation and the direct-PyTorch ExpoMLP are the implemented runtime components. Unchecked tasks are incomplete. The per-neuron ReLU/squared-ReLU blend is accepted and supersedes the original power proposal. Other proposed defaults must not be treated as approved simply because they appear in this roadmap.

## Working method

Take one bounded task at a time. Read its dependencies and contract, implement the smallest coherent change, run the named checks, and update its checkbox only when its acceptance criteria are met. Do not begin by copying PSANN's package tree.

For each completed task, add a compact entry under **Execution log** containing the task ID, changed paths, exact validation commands, outcome, and any limitation. If a dependency or design is unresolved, record it there and leave the task open. Do not substitute a mock for required CUDA evidence.

Future work requests can use this format:

```text
Implement task Pn.nn from docs/ROADMAP.md.
Follow the accepted decisions and referenced contracts.
Keep changes within that task; do not add deferred features.
Run its acceptance checks and update the execution log with actual outcomes.
```

## P0 — Resolve the design

Dependencies: none. No implementation is authorized by completing the documentation draft alone.

- [x] **P0.01 — Review PSANN and draft a focused plan.** Acceptance: source review, mathematical design, API proposal, validation plan, and phased tasks are linked from the root README.
- [x] **P0.02 — Confirm D01-D09.** Acceptance: record decisions, including coefficient mapping, computational-cost priorities, first workloads, dependencies, license, and initial CUDA scope. The 2026-08-30 decision entry accepts D02-D07 alongside the previously resolved choices.
- [x] **P0.03 — Reconcile the contracts.** Acceptance: all documents agree with the confirmed decisions; decide which initial defaults are provisional pending experiments. The accepted defaults are reflected in the contracts before P3 implementation.

Exit: no unresolved decision blocks activation semantics or the first package/API shape.

## P1 — Package skeleton and activation

Dependencies: P0 complete.

- [x] **P1.01 — Establish a small installable package.** Add `pyproject.toml`, `src/exponet/`, a confirmed license, `.gitignore`, and minimal test/lint configuration. Ignore environments, caches, checkpoints, datasets, and generated runs. Acceptance: editable install and a wheel import work without PSANN installed; record actual Python/Torch/NumPy/sklearn versions. Start with one validated Python version (proposed 3.11), then widen only with tests. Completed with the MIT License and an inspected wheel installed and exercised outside the source tree; see the 2026-08-30 execution entry.
- [x] **P1.02 — Implement ExpoActivation.** Follow the exact equation, safe evaluation, sharing, initialization, fixed mode, and broadcasting contracts. Acceptance: analytic forward/gradient tests, zero/negative behavior, invalid configuration tests, parameter/buffer registration, and a state_dict round trip pass on CPU.
- [x] **P1.03 — Prove coefficient learning in isolation (CPU scope).** A deterministic CPU test starts at `a=0.5`, optimizes only `theta`, and recovers the known target `a=0.7` within a declared tolerance; float64 gradcheck varies both the input and raw coefficient away from the origin. The deterministic CUDA forward/backward/update test now passes; see the 2026-08-30 CUDA execution entry.
- [x] **P1.04 — Measure activation overhead before expanding the library (CPU scope).** Used the timing protocol in VALIDATION.md to separate coefficient mapping, isolated activation inference/forward-backward, and minimal dense blocks with and without LayerNorm. Native ReLU, native squared ReLU, sine, and the real learned blend were measured across the required CPU shapes in float32 with recorded warmup, median, IQR, environment, and limitations. [The reviewed report](ACTIVATION_BENCHMARK.md) records significant isolated eager-CPU overhead that narrows inside dense blocks; no speed threshold or advantage is claimed. The CUDA benchmark smoke invocation succeeds, but a reviewed CUDA timing report still requires the full timing protocol.

Exit: the activation is correct independently of estimator preprocessing or architecture effects. No accuracy advantage is claimed.

## P2 — Dense model

Dependencies: P1.

- [x] **P2.01 — Compose ExpoMLP.** Built explicit Linear -> optional LayerNorm -> activation blocks and a linear readout. CPU tests cover hidden-width/output shapes, independent per-layer parameters, Xavier/zero-bias initialization, normalization-off behavior, width-one rejection with LayerNorm, and signed regression outputs.
- [x] **P2.02 — Verify direct PyTorch use.** Added `examples/direct_torch.py` with a caller-owned optimizer. CPU tests verify ordinary-weight/raw-blend updates, `.to()` parameter/buffer dtype and device behavior, strict state_dict reload of outputs and coefficients, and singleton/batched inference agreement.

Exit: one model backbone is usable through ordinary PyTorch without estimator state.

## P3 — Shared training and regression

Dependencies: P2.

- [x] **P3.01 — Implement input preparation and device resolution.** Valid dense features and regression targets convert to finite float32 arrays; malformed/nonfinite/complex/sparse inputs fail; explicit CUDA never silently falls back; rank metadata survives conversion; splitting precedes scaler fitting and validation cannot affect fitted scaler statistics.
- [x] **P3.02 — Implement the shared training loop.** AdamW uses separate non-overlapping linear-weight, blend, and no-decay groups; it provides sample-weighted mini-batch loss, optional clipping, validation, history, finite-state failures, and best-state restoration.
- [x] **P3.03 — Implement ExpoRegressor.** `fit`, `predict`, `score`, target scaling, blend inspection, lifecycle invalidation, cloning, Pipeline, and GridSearchCV smoke coverage are implemented and tested for one- and multi-output regression.
- [x] **P3.04 — Validate estimator device behavior.** CUDA regression fit/predict is finite and updates blend coefficients; identical state gives CPU/CUDA inference parity within recorded tolerances. Source arrays stay on CPU and only bounded mini-batches move to the selected device.

Exit: a usable regression workflow, including actual CUDA evidence. If hardware is unavailable, mark P3.04 blocked and continue independent CPU tasks without marking the phase complete.

## P4 — Classification and estimator integration

Dependencies: P3.01-P3.03; CUDA checks can follow P3.04 availability.

- [x] **P4.01 — Implement ExpoClassifier with the same trainer.** Integer and string label encoding, binary/multiclass K-logit loss, ordered probability columns, label reconstruction, and accuracy scoring pass; unseen validation labels and invalid target kinds fail clearly.
- [x] **P4.02 — Test estimator lifecycle and compatibility.** Repeated fits with changed classes reset state; stratified holdout errors are actionable; regression/classification cloning, parameter changes, Pipeline, and cross-validation smoke coverage pass. The official `check_estimator` suite was exercised: it reaches the intentionally unsupported continuous-label case because ExpoClassifier accepts only integer/string labels; no full-estimator-check compliance is claimed.
- [x] **P4.03 — Add small usage examples.** Runnable regression and multiclass-classification examples include fixed-blend control and explicit CPU/CUDA selection. CUDA classification gets a real training/inference smoke test.

Exit: two task wrappers share one trainer and preserve documented output semantics.

## P5 — Inference persistence

Dependencies: P4.

- [x] **P5.01 — Define and implement versioned snapshots.** Tensor-and-primitive metadata reconstructs only built-in ExpoNet models; fitted preprocessing and label/rank metadata are included; writes use a temporary sibling plus atomic replacement; loading uses `weights_only=True` with no pickle fallback.
- [x] **P5.02 — Verify portability and failure cases.** Trained/fixed blend coefficients, both normalization modes, regression target ranks/scalers, and classifier string labels survive save/load. CUDA-to-CPU loading, incompatible kind/version, damaged state, wrong shapes, failed-write preservation, fresh subsequent fit, and source/loaded inference parity are covered.

Exit: users can retain fitted models without introducing a general artifact/deployment platform.

## P6 — Controlled evaluation and default selection

Dependencies: P4 for experiments; P5 before publishing reusable fitted snapshots.

- [x] **P6.01 — Build one reproducible benchmark runner.** `benchmarks/initial_evaluation.py` implements the activation matrix and records paired splits/seeds, configuration, environment, native baselines, fixed/learned blends, parameter counts, metrics, elapsed time, coefficients, and failures in ignored JSON reports.
- [x] **P6.02 — Execute the initial matrix.** The 280-run five-seed matrix completed with both normalization modes on synthetic regression/multiclass controls and bundled real Diabetes/Iris datasets. Provenance, checksums, and attempted/completed counts are recorded in [the reviewed report](INITIAL_EVALUATION.md).
- [x] **P6.03 — Review findings and choose defaults.** The report summarizes held-out metrics, paired differences, stability, runtime, and coefficient behavior. Results do not support changing the initial blend or normalization defaults; no superiority claim is made.

Exit: a compact tracked report distinguishes mathematical correctness from empirical usefulness. A negative result is an acceptable outcome.

## P7 — First release readiness

Dependencies: P1-P6 complete; no actual publication is implied by readiness.

- [x] **P7.01 — Finalize documentation.** README/API documentation now uses verified examples and covers local installation, device selection, actual Windows CPU/CUDA coverage, numeric limits, output shapes, the four-export public API, and deferred scope. Roadmap and PSANN source-adaptation notices remain accurate.
- [x] **P7.02 — Validate a built distribution.** Built wheel/sdist were installed into a clean short-path Windows environment and tested outside the source tree: CPU artifact tests passed with CUDA-only tests skipped, lint passed, and all examples ran. The same wheel was installed into the CUDA environment and passed the complete GPU suite. Linux remains unclaimed.
- [x] **P7.03 — Review repository hygiene and release scope.** Git ignore/status checks, archive inspection, credential-pattern scanning, dependency metadata inspection, and source search confirm ignored generated artifacts, no PSANN runtime dependency, MIT metadata, and the recorded deferred scope. Committing, pushing, and publishing remain separate actions to request explicitly.

Exit: a small, validated initial release candidate with documented limitations.

## Deferred work: no automatic expansion

Residual architectures; convolutional or sequence estimators; BatchNorm/RMSNorm; dropout; mixed precision; compile/export tooling; custom kernels; distributed training; other accelerators; sample/class weights; dataframe schema management; missing-value handling; streaming; warm start; exact training resume; custom loss/optimizer/callback frameworks; serving; explanation frameworks; automatic search.

Reconsider only with a concrete user need or measured limitation. If a feature becomes necessary, add its contract, validation requirements, and bounded tasks before implementation.

## Execution log

| Date | Task | Changes and evidence | Remaining limitation |
| --- | --- | --- | --- |
| 2026-08-28 | P0.01 | Reviewed PSANN maintained docs and selected source/tests; added this linked planning set. Local Markdown links, code fences, trailing whitespace, and terminology checks passed. `git status --short` confirms only the new planning files in ExpoNet; source checkout status is unchanged. | Design questions await confirmation. No ExpoNet code, training, runtime test, commit, or push performed. |
| 2026-08-28 | P0.02 (partial) | Recorded per-neuron exponents and optional endpoint attainment. Added computational-cost discussion and early timing check. | Exact-power cost tradeoff, constraint mapping, and remaining scope choices still need review. No benchmark executed. |
| 2026-08-28 | P0.02 / P0.03 (partial) | Accepted the per-neuron ReLU/squared-ReLU blend; replaced active power mathematics, parameter names, diagnostics, and tests throughout the planning set. Document link/fence/whitespace/terminology checks passed; scalar arithmetic checks verified example values and input/raw-coefficient derivatives at 12 input/coefficient pairs away from zero. Historical entries above describe superseded proposals. | Coefficient mapping and remaining scope/default decisions are still pending. Documentation checks are not PyTorch validation; no implementation or benchmark run. |
| 2026-08-28 | P1.01 (superseded status) | Historical package-scaffolding entry: added the src layout, `pyproject.toml`, `.gitignore`, and test/lint configuration. The 2026-08-29 correction below supersedes its completion and license claims. | P1.01 remains open because no license has been selected and no wheel validation has been performed. |
| 2026-08-28 | P1.02 | Implemented ExpoActivation with all contracts: per-neuron and per-layer modes, `num_features`, `blend_init`, `trainable`, `feature_dim`. Forward uses `u * ((1 - a) + a * u)` with `a = sigmoid(theta)`. Gradient tests pass with analytic values at zero/negative/positive inputs and interior. Fixed mode stores `a_fixed` as buffer; trainable mode uses `theta` as Parameter. State_dict round trips correctly. CUDA test skipped (no CUDA available). CPU float64 validation included. | No PSANN coupling; no MLP, estimators, shared training loop, persistence, or benchmark suite implemented. |
| 2026-08-28 | P1.02 (superseded verification) | Historical verification entry. Its recovery description and test-count claim were incorrect; the 2026-08-29 correction below is authoritative. Other listed activation behaviors remain covered by the corrected suite. | This entry must not be used as current evidence. |
| 2026-08-29 | P1.02 / P1.03 correction | Corrected constructor validation, removed duplicate/misplaced tests, strengthened noncontiguous, distinct-coefficient, fixed-optimizer, state-dict, CUDA, and recovery coverage, removed the placeholder license field, and reconciled repository hygiene and status documentation. Exact verification: `python -B -m pytest -q -p no:cacheprovider -k "not cuda"` -> `62 passed, 1 deselected in 4.05s`; `python -B -m ruff check --no-cache --no-respect-gitignore src tests` -> `All checks passed!`; `python -B -m ruff format --check --no-cache --no-respect-gitignore src tests` -> `3 files already formatted`; `python -c "from exponet import ExpoActivation; print(ExpoActivation)"` -> `<class 'exponet.activations.ExpoActivation'>`; `git check-ignore -v src/exponet/activations.py tests/test_activations.py README.md pyproject.toml` -> no output (exit 1 because none of the paths are ignored); `git status --short` -> `.gitignore`, `README.md`, `docs/`, `pyproject.toml`, `src/`, and `tests/` are all untracked and eligible for addition. Focused command `python -B -m pytest -q -p no:cacheprovider tests/test_activations.py::TestExpoActivationConfiguration::test_per_neuron_rejects_non_integer_features tests/test_activations.py::TestExpoActivationConfiguration::test_rejects_non_integer_feature_dim tests/test_activations.py::TestExpoActivationConfiguration::test_noncontiguous_input tests/test_activations.py::TestIsolatedRecovery::test_recovery_simple` -> `11 passed in 3.66s`; this covers the requested invalid values, real noncontiguous fixture, and learning assertions. A direct recovery probe reported `initial=0.500000000 target=0.699999988 final=0.699999988 movement=0.199999988 abs_error=0.000000000`. | `torch.cuda.is_available()` reported `False`; the prepared CUDA test was deselected and not executed, so no CUDA result is claimed. License selection, wheel/distribution validation, MLP, estimators, training loop, persistence, benchmarks, and later phases remain unresolved or unimplemented. No commit, staging, or push was performed. |

### 2026-08-30 — P1.01 license and built-wheel validation

Confirmed license: MIT License, `Copyright (c) 2026 ExpoNet Contributors`.

Changed paths: `LICENSE`, `pyproject.toml`, `README.md`, `docs/DECISIONS.md`, `docs/PSANN_REUSE.md`, and `docs/ROADMAP.md`. The package metadata uses the SPDX expression `MIT`, explicitly includes `LICENSE`, and requires `setuptools>=77.0.0` for PEP 639 support.

Validation environment: Python 3.11.9, PyTorch 2.7.1+cu118, NumPy 2.4.6, and scikit-learn 1.4.2 on Windows CPU. `torch.cuda.is_available()` returned `False`; the CUDA test remained deselected, no CUDA operation was run, and no CUDA evidence is claimed.

Source checks:

```powershell
python -B -m pytest -q -p no:cacheprovider -k "not cuda"
```

Result: `62 passed, 1 deselected in 3.20s`.

```powershell
python -B -m ruff check --no-cache --no-respect-gitignore src tests
```

Result: `All checks passed!`.

```powershell
python -B -m ruff format --check --no-cache --no-respect-gitignore src tests
```

Result: `3 files already formatted`.

Build command using the declared PEP 517 backend:

```powershell
python -B -m build --wheel --outdir "C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\wheel" .
```

Result: `Successfully built exponet-0.1.0-py3-none-any.whl` in an isolated build environment after installing the declared `setuptools>=77.0.0` build requirement.

Wheel inspection used Python's standard `zipfile` and email metadata parsers to assert the exact seven wheel members, package name/version, `License-Expression: MIT`, `License-File: LICENSE`, byte-for-byte license inclusion, and the absence of any PSANN dependency. Result: `files=7 exact_contents=True license_match=True psann_dependency=False`. Importable package contents were exactly `exponet/__init__.py` and `exponet/activations.py`; tests, docs, source-layout paths, caches, environments, and generated output were not packaged as importable contents.

Temporary environment and installation commands:

```powershell
python -B -m venv --system-site-packages "C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\venv"
& "C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\venv\Scripts\python.exe" -B -m pip install --no-deps --no-cache-dir "C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\wheel\exponet-0.1.0-py3-none-any.whl"
```

Results: the virtual environment was created successfully and pip reported `Successfully installed exponet-0.1.0`. The environment deliberately reused existing system packages and installed the wheel with `--no-deps` to avoid downloading another PyTorch build; this was not a completely isolated dependency installation.

The first smoke invocation added Python's `-I` flag, which hid the Windows user-level Torch installation and failed before ExpoNet import with `ModuleNotFoundError: No module named 'torch'`. The affected validation was rerun without `-I`, still from the temporary working directory outside the repository, with explicit path and dependency assertions:

```powershell
& "C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\venv\Scripts\python.exe" -B -c "from pathlib import Path; import importlib.metadata as md; import sys; import torch; import exponet; from exponet import ExpoActivation; installed=Path(exponet.__file__).resolve(); repo=Path(r'C:\Users\milin\Documents\ExpoNet').resolve(); venv=Path(r'C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\venv').resolve(); assert venv in installed.parents; assert repo not in installed.parents; x=torch.tensor([[0.5,2.0]],requires_grad=True); act=ExpoActivation(num_features=2,blend_init=0.5); out=act(x); out.sum().backward(); assert torch.isfinite(out).all(); assert x.grad is not None and torch.isfinite(x.grad).all(); assert act.theta.grad is not None and torch.isfinite(act.theta.grad).all(); requires=md.requires('exponet') or []; assert not any('psann' in item.lower() for item in requires); assert not any(name=='psann' or name.startswith('psann.') for name in sys.modules); print(f'exponet_file={installed}'); print(f'outside_repo={repo not in installed.parents} inside_temp_venv={venv in installed.parents}'); print(f'output={out.detach().tolist()} theta_grad={act.theta.grad.detach().tolist()} finite=True'); print('psann_loaded=False psann_dependency=False')"
```

Result: `exponet_file=C:\Users\milin\AppData\Local\Temp\exponet-wheel-validation-5d4ca65dca5a4348b0f1aa25f2853bc3\venv\Lib\site-packages\exponet\__init__.py`, `outside_repo=True`, `inside_temp_venv=True`, deterministic output `[[0.375, 3.0]]`, theta gradient `[-0.0625, 0.5]`, `finite=True`, `psann_loaded=False`, and `psann_dependency=False`.

Remaining P1 limitations: P1.04 activation-overhead measurement is pending, and CUDA execution remains deferred for unavailable hardware. MLP, estimators, training, persistence, benchmarks, and later phases remain unimplemented.

### 2026-08-30 — P1.04 CPU activation-overhead measurement

Changed paths: `benchmarks/__init__.py`, `benchmarks/benchmark_activation.py`, `docs/ACTIVATION_BENCHMARK.md`, `README.md`, and `docs/ROADMAP.md`.

The benchmark used the current checkout (`C:\Users\milin\Documents\ExpoNet\src\exponet\__init__.py`) and the real `ExpoActivation`. It measured standalone sigmoid coefficient mapping, isolated activation inference and forward/backward, and minimal dense blocks with Identity or LayerNorm across batch sizes 1 and 128 and widths 64 and 256. References were native ReLU, native squared ReLU, and plain sine. Setup, input construction, and transfers were outside timed regions; gradients were cleared consistently inside forward/backward timing.

Environment: Windows 10 build 26200; Intel64 Family 6 Model 186 Stepping 3; Python 3.11.9; PyTorch 2.7.1+cu118; NumPy 2.4.6; scikit-learn 1.4.2; CPU float32; one intra-operation thread; seed 20260830. Each of 100 final cases used 20 explicit warmup iterations and `torch.utils.benchmark.Timer.blocked_autorange` with a 0.5-second minimum. Every table entry in `docs/ACTIVATION_BENCHMARK.md` records median and IQR microseconds; no final result had IQR greater than 25% of its median.

Exact timing command:

```powershell
python -B -m benchmarks.benchmark_activation --device cpu --num-threads 1 --warmup 20 --min-run-time 0.5 --output "$env:TEMP\exponet-activation-benchmark-cpu-20260830-final.json"
```

Result: `wrote=C:\Users\milin\AppData\Local\Temp\exponet-activation-benchmark-cpu-20260830-final.json` and `measurements=100`.

Review outcome: learned-blend isolated inference was 3.15-3.76 times the matched ReLU median and isolated forward/backward was 2.61-3.12 times. In dense blocks, ratios narrowed to 1.25-2.18 times for inference and 1.31-1.76 times for forward/backward without normalization, and 1.19-1.94 times and 1.17-1.60 times respectively with LayerNorm. Standalone sigmoid mapping represented 15.3-30.2% of the complete learned-activation median. The measurement shows eager-CPU overhead rather than a speed advantage; no threshold was predeclared, so the result is a baseline and limitation, not a performance pass claim.

CUDA availability and guarded command:

```powershell
python -B -m benchmarks.benchmark_activation --device cuda --num-threads 1 --warmup 1 --min-run-time 0.001 --output "$env:TEMP\exponet-activation-benchmark-cuda-should-not-exist.json"
```

Result: exit 2 with `CUDA was requested, but torch.cuda.is_available() is False`; no output file was created and no CUDA operation ran. This validates explicit rejection only and is not CUDA timing evidence.

Final checks:

```powershell
python -B -m pytest -q -p no:cacheprovider -k "not cuda"
python -B -m ruff check --no-cache --no-respect-gitignore src tests benchmarks
python -B -m ruff format --check --no-cache --no-respect-gitignore src tests benchmarks
```

Results: `62 passed, 1 deselected in 2.69s`; `All checks passed!`; `5 files already formatted`.

Remaining limitations: CUDA timing and validation require actual GPU hardware. These measurements cover one Windows CPU, eager float32, one thread, one PyTorch build, and small dense shapes; they are not end-to-end training evidence. P2-P7 remain unimplemented. No commit or push was performed for this task.

### 2026-08-30 — P2.01/P2.02 ExpoMLP and direct-PyTorch validation

Changed paths: `src/exponet/nn.py`, `src/exponet/__init__.py`, `tests/test_nn.py`, `examples/direct_torch.py`, `README.md`, and `docs/ROADMAP.md`.

Implemented `ExpoMLP` as direct PyTorch composition only: each hidden block is `Linear -> LayerNorm or Identity -> ExpoActivation`, with a final linear-only readout. Constructor and forward validation require positive non-boolean integer input/output widths, a nonempty tuple of positive non-boolean hidden widths, supported normalization, rank-two inputs, and matching input features. The direct example owns its `AdamW` optimizer and performs one update. No estimator, shared training, persistence, or additional benchmark functionality was added.

Exact CPU verification commands:

```powershell
python -B -m pytest -q -p no:cacheprovider -k "not cuda"
python -B -m ruff check --no-cache --no-respect-gitignore src tests benchmarks examples
python -B -m ruff format --check --no-cache --no-respect-gitignore src tests benchmarks examples
python -c "from exponet import ExpoMLP; print(ExpoMLP)"
python -B examples/direct_torch.py
git diff --check
git status --short
```

Results: `110 passed, 1 deselected in 3.71s`; `All checks passed!`; `8 files already formatted`; import printed `<class 'exponet.nn.ExpoMLP'>`; and the example printed `prediction_shape=(1, 1)`. `git diff --check` exited 0 with only existing line-ending warnings for `README.md`, `docs/ROADMAP.md`, and `src/exponet/__init__.py`. Status showed the intentional P1.04 changes in `README.md`, `docs/ROADMAP.md`, `benchmarks/`, and `docs/ACTIVATION_BENCHMARK.md`, together with the P2 paths listed above.

CUDA limitation at the time of this CPU-only entry: CUDA tests were deliberately deselected, and no CUDA operation or validation was claimed. Subsequent entries record the later CUDA setup and validation. No staging, commit, or push was performed.

### 2026-08-30 — CUDA environment and activation smoke validation

Installed Python 3.12.10 and an isolated `.venv-cuda` environment in the repository, then installed PyTorch 2.11.0+cu128, NumPy 2.5.2, scikit-learn 1.9.0, pytest 9.1.1, Ruff 0.16.5, and ExpoNet in editable mode. PyTorch reports CUDA runtime 12.8, one available device, and `NVIDIA GeForce RTX 5060`; `nvidia-smi` reports driver 616.56 and 8151 MiB of GPU memory. The CUDA toolkit compiler (`nvcc`) is not installed, which is not required for this prebuilt-PyTorch test path.

Validation commands:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_activations.py::TestExpoActivationDeviceAndDtype::test_cuda_float32
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv-cuda\Scripts\python.exe -B -m ruff check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m ruff format --check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m benchmarks.benchmark_activation --device cuda --num-threads 1 --warmup 1 --min-run-time 0.001 --output "$env:TEMP\exponet-activation-benchmark-cuda-20260830-smoke.json"
```

Results: the CUDA activation forward/backward/update test passed (`1 passed in 8.04s`); the full suite passed (`111 passed in 4.03s`); Ruff reported no lint errors and five formatted files; the CUDA benchmark smoke wrote 100 measurements. The benchmark smoke only proves GPU execution and synchronization paths, not performance: it uses one warmup and a 0.001-second minimum run time rather than the documented timing protocol.

At the time of this entry, P3/P4 estimators, training, CPU/GPU state parity, and CUDA persistence were unavailable. The historical CPU-only execution entries above remain accurate descriptions of their recorded environments. No staging, commit, or push was performed.

### 2026-08-30 — P3 shared training and regression

Changed paths: `src/exponet/_validation.py`, `src/exponet/_training.py`, `src/exponet/estimators.py`, `src/exponet/__init__.py`, `tests/test_estimators.py`, and the P3 documentation.

Implemented `ExpoRegressor` as a scikit-learn `BaseEstimator`/`RegressorMixin` wrapper around `ExpoMLP`. Dense finite numeric arrays are validated and converted to float32; sparse, complex, nonfinite, tensor, malformed, and empty inputs fail before model allocation. CPU arrays remain resident on CPU, with only mini-batches moved to the resolved device. Optional feature/target scalers fit only on training rows after splitting. The shared loop uses AdamW with non-overlapping linear-weight, blend, and no-decay parameter groups; sample-weighted epoch loss; optional clipping; finite-state checks; validation history; and early-stopping best-state restoration.

Validation commands:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m ruff format --check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m ruff check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Results: `132 passed in 6.83s`; Ruff reported no lint errors and nine formatted files. P3 tests cover rank preservation, target inverse scaling, validation-scaler isolation, fixed and learned blends, optimizer grouping, early stopping, repeated/failed-fit lifecycle, cloning, Pipeline, GridSearchCV, real CUDA regression fit/predict, CUDA blend updates, and CPU/GPU inference parity from identical model state. No staging, commit, or push was performed.

### 2026-08-30 — P4 classification and estimator integration

Changed paths: `src/exponet/_training.py`, `src/exponet/_validation.py`, `src/exponet/estimators.py`, `src/exponet/__init__.py`, `tests/test_classifier.py`, `examples/regression.py`, `examples/classification.py`, and the P4 documentation.

Implemented `ExpoClassifier` with the shared bounded-batch trainer used by regression. It validates homogeneous one-dimensional integer or string labels, sorts and preserves `classes_`, trains `K` logits with cross-entropy for both binary and multiclass tasks, returns ordered softmax probabilities, and maps argmax predictions back to original labels. Explicit validation rejects labels outside the fitted class set; internal early stopping uses a seeded stratified split. The classifier owns fresh model/scaler/history state per fit and supports cloning, Pipeline, and cross-validation.

Validation commands:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m ruff format --check --no-cache --no-respect-gitignore src tests examples
.\.venv-cuda\Scripts\python.exe -B -m ruff check --no-cache --no-respect-gitignore src tests examples
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv-cuda\Scripts\python.exe -B examples\regression.py
.\.venv-cuda\Scripts\python.exe -B examples\classification.py
```

Results: `146 passed in 9.00s`; Ruff reported no lint errors and thirteen formatted files. Both examples ran on CUDA with synthetic data (`r2=0.998` for fixed-blend regression and `accuracy=1.000` for multiclass classification). The CUDA classifier test verifies finite probabilities and blend movement. `sklearn.utils.estimator_checks.check_estimator` was also exercised with a small CPU configuration; it reaches the intentionally unsupported continuous-label case because ExpoClassifier accepts only integer/string labels, so full official-estimator-check compliance is not claimed. No staging, commit, or push was performed.

### 2026-08-30 — P5 restricted inference persistence

Changed paths: `src/exponet/_persistence.py`, `src/exponet/estimators.py`, `tests/test_persistence.py`, and the P5 documentation.

Implemented version-1 inference snapshots for both estimators. Snapshots contain primitive constructor/fitted metadata and CPU tensor copies of the strict `ExpoMLP` state, along with scaler arrays, regression output-rank metadata, or classifier labels. Saving does not move the live model; it writes a temporary sibling and atomically replaces the destination. Loading uses `torch.load(..., weights_only=True, map_location="cpu")`, validates the kind/version/metadata, reconstructs only `ExpoMLP`, strictly loads state, and then honors the requested load device.

Validation commands:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m ruff format --check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m ruff check --no-cache --no-respect-gitignore src tests
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_persistence.py
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Results: persistence coverage passed (`10 passed in 6.25s`); the full suite passed (`156 passed in 10.23s`); Ruff reported no lint errors and twelve formatted files. Tests cover trained/fixed blends, both normalizations, one/multi-output target ranks, enabled/disabled scalers, classifier string labels, incompatible kind/version, damaged snapshots, wrong tensor shapes, failed-write preservation, fresh fit after load, and a CUDA-trained regression snapshot loaded on CPU without moving the live CUDA model. CPU/GPU output differences are compared with float32 tolerances rather than bitwise equality. No staging, commit, or push was performed.

### 2026-08-30 — P6 initial controlled evaluation

Changed paths: `benchmarks/initial_evaluation.py`, `docs/INITIAL_EVALUATION.md`, `README.md`, `docs/DECISIONS.md`, `docs/VALIDATION.md`, and `docs/ROADMAP.md`. Raw output: ignored `reports/initial-evaluation-20260830.json`.

Implemented a reproducible runner that evaluates native ReLU/squared-ReLU, fixed Expo blends at 0/0.5/1, and learned per-layer/per-neuron blends. It uses paired 64/16/20 train/validation/test splits, five fixed seeds, both LayerNorm modes, matched `(16,)` blocks and optimizer budgets, train-only feature scaling, and train-only regression target scaling. It records per-seed metrics, parameters, runtime, coefficients, environment, dataset provenance/checksums, and failures.

Execution command:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m benchmarks.initial_evaluation --device cpu --epochs 35 --seeds 11,23,37,53,71 --output reports\initial-evaluation-20260830.json
```

Results: `280` of `280` runs completed. The reviewed [initial evaluation report](INITIAL_EVALUATION.md) contains the full aggregate metrics, paired learned-vs-ReLU differences, coefficient behavior, data provenance, limitations, and decision. On this small matrix learned blends did not consistently improve held-out results over ReLU; no default change is justified. No staging, commit, or push was performed.

### 2026-08-30 — P7 first release readiness

Changed paths: `README.md`, `docs/API_CONTRACT.md`, `docs/DESIGN.md`, `docs/DECISIONS.md`, `docs/VALIDATION.md`, `docs/ROADMAP.md`, and `tests/test_estimators.py`.

The README is now a release guide rather than a planning notice: it contains a
verified regression quick start, local install/device guidance, the small public
API, prediction shapes, actual Windows coverage, numeric/persistence limits, and
explicitly deferred features. API and design documents now identify the P1-P6
surface as implemented. The device-index test accepts both valid explicit-CUDA
rejection paths: CUDA unavailable on a CPU-only build, or an out-of-range CUDA
device when CUDA is available.

Built artifacts: `dist/exponet-0.1.0-py3-none-any.whl` and
`dist/exponet-0.1.0.tar.gz` (both ignored). The wheel contains 12 members; the
sdist contains 27 members. Metadata reports `Name: exponet`, `Version: 0.1.0`,
`License-Expression: MIT`, and runtime requirements only for Torch, NumPy, and
scikit-learn. Neither archive contains datasets, reports/runs, checkpoints, or
virtual environments, and neither declares a PSANN dependency.

Validation commands:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m ruff format --check --no-cache --no-respect-gitignore src tests benchmarks examples
.\.venv-cuda\Scripts\python.exe -B -m ruff check --no-cache --no-respect-gitignore src tests benchmarks examples
.\.venv-cuda\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv-cuda\Scripts\python.exe -B -m build
```

Source validation passed with `156 passed in 16.67s`. A clean Python 3.12
environment at `C:\e7v\v` installed the built wheel with its declared
dependencies (Torch 2.13.0 CPU build, NumPy 2.5.2, and scikit-learn 1.9.0).
From the extracted sdist and outside the original source tree, its test suite
passed `152 passed, 4 skipped`; all four skips were CUDA-only tests, lint passed,
and direct-PyTorch/regression/classification examples completed on CPU. A longer
temporary path initially exposed Windows' Torch path-length limit; the clean
validation was repeated successfully from the short path above.

The same built wheel was then installed non-editably into the CUDA environment.
It imported from `site-packages` rather than `src`, saw `NVIDIA GeForce RTX 5060`
with CUDA runtime 12.8, passed `156 passed in 16.08s`, and all three examples
completed; regression and classification resolved to CUDA. Git status confirms
that `.venv-cuda/`, `dist/`, `reports/`, and generated egg metadata are ignored.
A credential-pattern scan and source/metadata scan found no credential-like
assignments or PSANN runtime import/dependency. Linux CPU validation has not
been run and no Linux support claim is made. No staging, commit, push, or
publication was performed.
