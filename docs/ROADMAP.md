# ExpoNet implementation roadmap

Updated: 2026-08-30. Current stage: **P1 and P2 CPU scope complete; CUDA validation deferred for unavailable hardware**.

This is the single progress tracker. [DECISIONS.md](DECISIONS.md) owns accepted choices, [DESIGN.md](DESIGN.md) owns mathematical behavior, [API_CONTRACT.md](API_CONTRACT.md) owns interfaces, and [VALIDATION.md](VALIDATION.md) owns evaluation expectations.

ExpoActivation is the only implemented runtime component. Unchecked tasks are incomplete. The per-neuron ReLU/squared-ReLU blend is accepted and supersedes the original power proposal. Other proposed defaults must not be treated as approved simply because they appear in this roadmap.

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
- [ ] **P0.02 — Confirm D01-D09.** Acceptance: record decisions, including coefficient mapping, computational-cost priorities, first workloads, dependencies, license, and initial CUDA scope. The per-neuron linear/quadratic blend and no exact-endpoint requirement are confirmed; remaining choices stay open.
- [ ] **P0.03 — Reconcile the contracts.** Acceptance: all documents agree with the confirmed decisions; decide which initial defaults are provisional pending experiments. Obtain the go-ahead to begin implementation.

Exit: no unresolved decision blocks activation semantics or the first package/API shape.

## P1 — Package skeleton and activation

Dependencies: P0 complete.

- [x] **P1.01 — Establish a small installable package.** Add `pyproject.toml`, `src/exponet/`, a confirmed license, `.gitignore`, and minimal test/lint configuration. Ignore environments, caches, checkpoints, datasets, and generated runs. Acceptance: editable install and a wheel import work without PSANN installed; record actual Python/Torch/NumPy/sklearn versions. Start with one validated Python version (proposed 3.11), then widen only with tests. Completed with the MIT License and an inspected wheel installed and exercised outside the source tree; see the 2026-08-30 execution entry.
- [x] **P1.02 — Implement ExpoActivation.** Follow the exact equation, safe evaluation, sharing, initialization, fixed mode, and broadcasting contracts. Acceptance: analytic forward/gradient tests, zero/negative behavior, invalid configuration tests, parameter/buffer registration, and a state_dict round trip pass on CPU.
- [x] **P1.03 — Prove coefficient learning in isolation (CPU scope).** A deterministic CPU test starts at `a=0.5`, optimizes only `theta`, and recovers the known target `a=0.7` within a declared tolerance; float64 gradcheck varies both the input and raw coefficient away from the origin. A deterministic CUDA forward/backward/update test is prepared, but CUDA execution is deferred because this machine has no GPU; no CUDA pass is claimed.
- [x] **P1.04 — Measure activation overhead before expanding the library (CPU scope).** Used the timing protocol in VALIDATION.md to separate coefficient mapping, isolated activation inference/forward-backward, and minimal dense blocks with and without LayerNorm. Native ReLU, native squared ReLU, sine, and the real learned blend were measured across the required CPU shapes in float32 with recorded warmup, median, IQR, environment, and limitations. [The reviewed report](ACTIVATION_BENCHMARK.md) records significant isolated eager-CPU overhead that narrows inside dense blocks; no speed threshold or advantage is claimed. CUDA timing remains deferred because no CUDA device is available.

Exit: the activation is correct independently of estimator preprocessing or architecture effects. No accuracy advantage is claimed.

## P2 — Dense model

Dependencies: P1.

- [x] **P2.01 — Compose ExpoMLP.** Built explicit Linear -> optional LayerNorm -> activation blocks and a linear readout. CPU tests cover hidden-width/output shapes, independent per-layer parameters, Xavier/zero-bias initialization, normalization-off behavior, width-one rejection with LayerNorm, and signed regression outputs.
- [x] **P2.02 — Verify direct PyTorch use.** Added `examples/direct_torch.py` with a caller-owned optimizer. CPU tests verify ordinary-weight/raw-blend updates, `.to()` parameter/buffer dtype and device behavior, strict state_dict reload of outputs and coefficients, and singleton/batched inference agreement.

Exit: one model backbone is usable through ordinary PyTorch without estimator state.

## P3 — Shared training and regression

Dependencies: P2.

- [ ] **P3.01 — Implement input preparation and device resolution.** Acceptance: valid dense inputs convert to float32; malformed/nonfinite/complex/sparse inputs fail; explicit CUDA never silently falls back; regression rank metadata survives conversion; splitting precedes scaler fitting and validation cannot affect fitted scaler statistics.
- [ ] **P3.02 — Implement the shared training loop.** AdamW, blend parameter group, sample-weighted epoch loss, optional clipping, validation, history, and best-state restoration. Acceptance: final short batch is included; fixed coefficients stay fixed; every trainable parameter appears in exactly one optimizer group; finite-state failures raise; early-stopping boundary cases pass.
- [ ] **P3.03 — Implement ExpoRegressor.** Acceptance: fit/predict/score, one- and multiple-output shapes, target inverse scaling, get_blend_weights, fit reset, fitted-state invalidation, clone, and a small Pipeline/GridSearchCV smoke test pass. A small synthetic regression problem overfits with a fixed seed without requiring an improvement over ReLU.
- [ ] **P3.04 — Validate estimator device behavior.** Acceptance: real CUDA regression fit/predict produces finite results and learns blend coefficients; identical saved model state gives CPU/CUDA inference parity within recorded tolerances; batching and CPU staging avoid requiring the entire dataset on the GPU.

Exit: a usable regression workflow, including actual CUDA evidence. If hardware is unavailable, mark P3.04 blocked and continue independent CPU tasks without marking the phase complete.

## P4 — Classification and estimator integration

Dependencies: P3.01-P3.03; CUDA checks can follow P3.04 availability.

- [ ] **P4.01 — Implement ExpoClassifier with the same trainer.** Acceptance: integer and string label encoding, binary/multiclass K-logit loss, probability column ordering, label reconstruction, and accuracy scoring pass; unseen validation labels and invalid target kinds fail clearly.
- [ ] **P4.02 — Test estimator lifecycle and compatibility.** Acceptance: repeated fits with changed classes reset state; stratified holdout errors are actionable; regression/classification cloning, parameter changes, Pipeline, cross-validation, and relevant official estimator checks are exercised. Record any unsupported checks with a narrow contract explanation; do not claim full compliance from smoke tests alone.
- [ ] **P4.03 — Add small usage examples.** Acceptance: regression, multiclass classification, fixed-blend controls, and explicit CPU/CUDA selection examples run from the installed package. No network dataset download is needed for default examples. CUDA classification gets a real training/inference smoke test.

Exit: two task wrappers share one trainer and preserve documented output semantics.

## P5 — Inference persistence

Dependencies: P4.

- [ ] **P5.01 — Define and implement versioned snapshots.** Acceptance: primitive metadata and tensors reconstruct only built-in model types; all fitted preprocessing and label/rank metadata are included; writes are atomic; loading is explicitly restricted with no arbitrary pickle fallback.
- [ ] **P5.02 — Verify portability and failure cases.** Acceptance: trained/fixed blend coefficients, all normalization modes, regression target ranks/scalers, and classifier string labels survive save/load. Test CUDA-to-CPU loading, incompatible kind/version, damaged state, wrong shapes, and failed-write preservation. Loaded inference matches the source model; a subsequent fit starts fresh.

Exit: users can retain fitted models without introducing a general artifact/deployment platform.

## P6 — Controlled evaluation and default selection

Dependencies: P4 for experiments; P5 before publishing reusable fitted snapshots.

- [ ] **P6.01 — Build one reproducible benchmark runner.** Acceptance: implement the experiment matrix in VALIDATION.md; record split/seed/configuration/environment, native ReLU baseline, fixed blends, learned coefficients, actual parameter counts, metrics, elapsed time, and failures. Generated outputs go to ignored `runs/` or `reports/`.
- [ ] **P6.02 — Execute the initial matrix.** Acceptance: paired runs use the same data partitions and budgets, at least five seeds for reported comparisons, and both normalized/unnormalized blocks. Include at least one real regression and one real classification dataset with recorded provenance, alongside synthetic controls. Failures remain visible in aggregate summaries.
- [ ] **P6.03 — Review findings and choose defaults.** Acceptance: summarize per-dataset accuracy, stability, runtime, coefficient behavior, and seed variability. Compare learned blends against fixed `blend_init` as well as ReLU. Record whether evidence supports changing normalization or initialization defaults. No selected best seed or universal superiority claim.

Exit: a compact tracked report distinguishes mathematical correctness from empirical usefulness. A negative result is an acceptable outcome.

## P7 — First release readiness

Dependencies: P1-P6 complete; no actual publication is implied by readiness.

- [ ] **P7.01 — Finalize documentation.** Acceptance: replace planned snippets with verified examples, document install/device guidance, actual version/platform coverage, numeric limitations, output shapes, and the small public API. Keep roadmap history and source-adaptation notices accurate.
- [ ] **P7.02 — Validate a built distribution.** Acceptance: build wheel/sdist, install in a clean environment, run relevant tests/examples outside the source tree, and run lint checks. Record Windows CPU and CUDA evidence; add Linux CPU checks before claiming Linux support. GPU skips do not count as CUDA passes.
- [ ] **P7.03 — Review repository hygiene and release scope.** Acceptance: no datasets, virtual environments, secrets, or generated checkpoints in tracked files; no undeclared PSANN dependency; reviewed license and dependency metadata; all deferred features remain out. Committing, pushing, and publishing are separate actions to request explicitly.

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

CUDA limitation: CUDA tests were deliberately deselected by the CPU command; no CUDA operation or validation is claimed because `torch.cuda.is_available()` is false on this machine. Current evidence covers CPU direct-module behavior only. P3.01 is the next roadmap task. No staging, commit, or push was performed.
